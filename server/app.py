import asyncio
import json
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig
from engine.model import GPT
from engine.scheduler import Engine, Request
from server.engine_loop import EngineLoop

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()
eloop = EngineLoop(Engine(model=model, n_slots=4, max_len=512))


#the three blocking mechanisms, for the dashboard race (run via to_thread; each returns full text at once)
@torch.no_grad()
def _naive(ids, n_new):
    for _ in range(n_new):
        logits, _ = model(ids)
        ids = torch.cat([ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
    return ids

BLOCKING_MODES = {
    "naive": lambda ids, n: _naive(ids, n),
    "cached": lambda ids, n: model.generate(ids, max_new_tokens=n),
    "static": lambda ids, n: model.generate_batch([ids], max_new_tokens=n),
}

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(eloop.run())  # start the loop when the server boots
    yield # server runs here
    task.cancel()  # and stops it on shutdown

app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(body: dict):
    ids = tok(body["prompt"], return_tensors="pt").input_ids.to(DEVICE)
    n_new = body.get("max_new_tokens", 50)
    mode = body.get("mode", "continuous")

    if mode == "static":
        #static batching: ALL prompts in one padded generate_batch call; results burst together
        all_ids = [tok(p, return_tensors="pt").input_ids.to(DEVICE)
                   for p in body.get("prompts", [body["prompt"]])]
        async def sse_static():
            t0 = time.perf_counter()
            out = await asyncio.to_thread(model.generate_batch, all_ids, n_new)
            total = time.perf_counter() - t0
            texts = [tok.decode(out[i, -n_new:]) for i in range(len(all_ids))]
            yield f"data: {json.dumps({'texts': texts, 'done': True, 'n_tokens': n_new * len(all_ids), 'total_s': round(total, 3)})}\n\n"
        return StreamingResponse(sse_static(), media_type="text/event-stream")

    if mode in BLOCKING_MODES:
        #blocking mechanisms: run off-loop, then emit the whole result as one event
        async def sse_burst():
            t0 = time.perf_counter()
            out = await asyncio.to_thread(BLOCKING_MODES[mode], ids, n_new)
            total = time.perf_counter() - t0
            text = tok.decode(out[0, -n_new:])
            yield f"data: {json.dumps({'token': text, 'done': True, 'n_tokens': n_new, 'total_s': round(total, 3)})}\n\n"
        return StreamingResponse(sse_burst(), media_type="text/event-stream")

    #continuous: the real serving path, token by token
    req = Request(prompt_ids=ids, max_new_tokens=n_new)
    outbox = eloop.register(req)

    async def sse(): # async generator: one SSE line per token
        while True:
            token, done = await outbox.get()
            yield f"data: {json.dumps({'token': tok.decode([token]), 'done': done})}\n\n"
            if done:
                break

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/")
async def home():
    return HTMLResponse("""<!doctype html>
<title>llm-inference</title>
<style>
  body { font-family: ui-monospace, "SF Mono", Menlo, monospace; background: #fdfdfc;
         color: #1a1a1a; max-width: 880px; margin: 72px auto; padding: 0 24px;
         font-size: 13px; line-height: 1.6; }
  h1 { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 4px; }
  .sub { color: #8b8a85; margin-bottom: 28px; }
  .pin { width: 100%; box-sizing: border-box; font: inherit; color: inherit;
         background: transparent; border: 1px solid #e4e3dd; padding: 8px 10px;
         outline: none; }
  .pin:focus { border-color: #b8b6ae; }
  .pin::placeholder { color: #b8b6ae; }
  button { font: inherit; background: transparent; border: 1px solid #e4e3dd;
           padding: 6px 16px; margin: 10px 0 32px; cursor: pointer; color: #1a1a1a; }
  button:hover { background: #f3f2ee; }
  #panes { display: grid; grid-template-columns: 1fr 1fr; gap: 28px 36px; }
  .pane { border-top: 1px solid #e4e3dd; padding-top: 10px; min-height: 130px; }
  .stat { color: #8b8a85; font-size: 12px; }
  .out { white-space: pre-wrap; margin-top: 8px; color: #3d3c39; }
</style>
<body>
<h1>llm-inference</h1>
<div class="sub">gpt-2 124m served from this machine &middot;
mechanisms run one at a time so timings are fair (no gpu contention)</div>
<div id="inputs" style="display:grid;gap:6px">
  <input class="pin" value="The meaning of life is" placeholder="prompt 1">
  <input class="pin" value="Once upon a time" placeholder="prompt 2">
  <input class="pin" value="def fibonacci(n):" placeholder="prompt 3">
  <input class="pin" placeholder="prompt 4">
</div>
<button onclick="race()">run</button>
<div id="table"></div>
<div id="stream"></div>
<script>
const N_NEW = 40;
const OTHERS = [["naive", "#b06456", "no kv cache, one prompt at a time"],
                ["cached", "#6b8f5e", "kv cache, one prompt at a time"],
                ["static batching", "#a89354", "all prompts in one padded batch"]];
const CONT_COLOR = "#7a72b5";

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

async function race() {
  const prompts = [...document.querySelectorAll('.pin')]
                  .map(el => el.value.trim()).filter(Boolean);
  if (!prompts.length) return;

  //the one visible output: continuous, streaming live
  const stream = document.getElementById('stream');
  stream.innerHTML = `<div class="pane" style="margin-top:28px"><span style="color:${CONT_COLOR}">continuous batching</span>
    <span class="stat">all prompts as live streams, one shared batch</span>
    <span id="sum-continuous" class="stat" style="float:right"></span>` +
    prompts.map((p, i) => `<div style="margin-top:10px"><span class="stat">${esc(p.slice(0,60))}
      &middot; <span id="s-c${i}"></span></span><div id="o-c${i}" class="out"></div></div>`).join('') +
    `</div>`;

  //the others: timing rows only; their text is checked against continuous, not shown
  const table = document.getElementById('table');
  table.innerHTML = `<div class="pane">` +
    OTHERS.map(([label, c, desc]) =>
      `<div style="margin-top:6px"><span style="color:${c}">${label}</span>
       <span class="stat">${desc} &middot; <span id="row-${label}">waiting</span></span></div>`).join('') +
    `</div>`;

  //story order: each blocking mechanism in turn (whole GPU to itself), continuous last
  const results = {};
  for (const [label] of OTHERS) {
    const row = document.getElementById('row-' + label);
    row.textContent = 'running...';
    const t0 = performance.now();
    const mode = label.split(' ')[0];
    const texts = mode === 'static' ? await runStatic(prompts) : await runSequential(mode, prompts);
    results[label] = texts;
    row.textContent = `all ${prompts.length} in ${((performance.now() - t0) / 1000).toFixed(2)}s | checking...`;
  }

  //continuous last: streams live, and its texts are the reference for the identity checks
  const sum = document.getElementById('sum-continuous');
  sum.textContent = 'running...';
  const t0 = performance.now();
  const ref = await Promise.all(prompts.map((p, i) => runContinuous(p, i)));
  sum.textContent = `all ${prompts.length} in ${((performance.now() - t0) / 1000).toFixed(2)}s`;
  for (const [label] of OTHERS) {
    const row = document.getElementById('row-' + label);
    const texts = results[label];
    const same = texts.length === ref.length && texts.every((t, i) => t === ref[i]);
    row.textContent = row.textContent.replace('checking...',
      same ? 'output identical \u2713' : 'OUTPUT DIVERGED \u2717');
  }
}

async function runContinuous(prompt, i) {
  const out = document.getElementById('o-c' + i);
  const stat = document.getElementById('s-c' + i);
  const t0 = performance.now();
  let ttft = null, text = '', n = 0;
  stat.textContent = '...';
  const resp = await fetch('/generate', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt, max_new_tokens: N_NEW})});
  await readSSE(resp, ev => {
    if (ttft === null) ttft = (performance.now() - t0) / 1000;
    text += ev.token; n++;
    out.textContent = text;
    if (ev.done) {
      const total = (performance.now() - t0) / 1000;
      stat.textContent = `ttft ${ttft.toFixed(2)}s, ${total.toFixed(2)}s, ${(n / total).toFixed(0)} tok/s`;
    }
  });
  return text;
}

async function runSequential(mode, prompts) {
  const texts = [];
  for (const prompt of prompts) {
    const resp = await fetch('/generate', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, max_new_tokens: N_NEW, mode})});
    const ev = await readSSE(resp, () => {});
    texts.push(ev.token);
  }
  return texts;
}

async function runStatic(prompts) {
  const resp = await fetch('/generate', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: prompts[0], prompts, max_new_tokens: N_NEW, mode: 'static'})});
  const ev = await readSSE(resp, () => {});
  return ev.texts;
}

async function readSSE(resp, onEvent) {
  const reader = resp.body.getReader(), dec = new TextDecoder();
  let buf = '', last = null;
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value);
    let i;
    while ((i = buf.indexOf('\\n\\n')) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 2);
      if (!line.startsWith('data: ')) continue;
      last = JSON.parse(line.slice(6));
      onEvent(last);
    }
  }
  return last;
}
</script>
</body>""")