import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from engine.config import DEVICE, MODEL
from engine.load import LABELS, load_models
from engine.scheduler import Engine, Request
from server.engine_loop import EngineLoop

#every served model gets its own engine loop (own scheduler, own kv cache); requests pick one by name.
#MODELS=gpt2,qwen (default both) chooses what loads; MODEL= picks the dashboard's default
SERVED = [m.strip() for m in os.environ.get("MODELS", "gpt2,qwen").split(",")]
DEFAULT = MODEL if MODEL in SERVED else SERVED[0]
MODELS = {name: (model, tok, EngineLoop(Engine(model=model, n_slots=4, max_len=512)))
          for name, (model, tok) in load_models(SERVED).items()}


#the three blocking mechanisms, for the dashboard race. each takes an on_step(i, ctx_len)
#progress hook so the page can draw the work per step while it runs
@torch.no_grad()
def _naive(model, ids, n_new, sampling, on_step):
    for i in range(n_new):
        logits, _ = model(ids) #no cache: the whole context is recomputed every step (the baseline stays greedy)
        ids = torch.cat([ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        on_step(i, ids.shape[1])
    return ids

BLOCKING_MODES = {
    "naive":  lambda model, ids, n, s, cb: _naive(model, ids, n, s, cb),
    "cached": lambda model, ids, n, s, cb: model.generate(ids, max_new_tokens=n, on_step=cb, **s),
}
#tokens touched per step, for the work strip: naive redoes the whole context, cached does 1
WORK = {"naive": lambda ctx, b: ctx, "cached": lambda ctx, b: 1, "static": lambda ctx, b: b}


@asynccontextmanager
async def lifespan(app):
    tasks = [asyncio.create_task(eloop.run()) for _, _, eloop in MODELS.values()]  # one loop per model, started at boot
    yield # server runs here
    for t in tasks:
        t.cancel()  # and stopped on shutdown

app = FastAPI(lifespan=lifespan)


async def _stream_blocking(run, mode, n_new, b, finish):
    """run a blocking generation off-loop while relaying its on_step progress as SSE events;
    `run(on_step)` does the work, `finish(out)` builds the final event's payload"""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    def on_step(i, ctx): #called on the worker thread -> hand the event to the loop thread-safely
        loop.call_soon_threadsafe(q.put_nowait, {"step": i + 1, "of": n_new, "ctx": ctx, "work": WORK[mode](ctx, b)})
    t0 = time.perf_counter()
    fut = asyncio.ensure_future(asyncio.to_thread(run, on_step))
    while not fut.done():
        try:
            ev = await asyncio.wait_for(q.get(), timeout=0.05)
            yield f"data: {json.dumps(ev)}\n\n"
        except asyncio.TimeoutError:
            pass
    while not q.empty():
        yield f"data: {json.dumps(q.get_nowait())}\n\n"
    out = fut.result()
    payload = finish(out)
    payload.update({"done": True, "total_s": round(time.perf_counter() - t0, 3)})
    yield f"data: {json.dumps(payload)}\n\n"


@app.post("/generate")
async def generate(body: dict):
    model, tok, eloop = MODELS[body.get("model", DEFAULT)]
    ids = tok(body["prompt"], return_tensors="pt").input_ids.to(DEVICE)
    n_new = body.get("max_new_tokens", 50)
    mode = body.get("mode", "continuous")
    #sampling is opt-in; greedy by default so the dashboard's identity checks hold
    sampling = {"do_sample": body.get("do_sample", False), "temperature": body.get("temperature", 1.0), "top_k": body.get("top_k")}

    if mode == "static":
        #static batching: ALL prompts in one padded generate_batch call; results burst together
        all_ids = [tok(p, return_tensors="pt").input_ids.to(DEVICE) for p in body.get("prompts", [body["prompt"]])]
        gen = _stream_blocking(
            lambda cb: model.generate_batch(all_ids, n_new, on_step=cb, **sampling), mode, n_new, len(all_ids),
            lambda out: {"texts": [tok.decode(out[i, -n_new:]) for i in range(len(all_ids))], "n_tokens": n_new * len(all_ids)})
        return StreamingResponse(gen, media_type="text/event-stream")

    if mode in BLOCKING_MODES:
        gen = _stream_blocking(
            lambda cb: BLOCKING_MODES[mode](model, ids, n_new, sampling, cb), mode, n_new, 1,
            lambda out: {"token": tok.decode(out[0, -n_new:]), "n_tokens": n_new})
        return StreamingResponse(gen, media_type="text/event-stream")

    #continuous: the real serving path, token by token
    req = Request(prompt_ids=ids, max_new_tokens=n_new, **sampling)
    outbox = eloop.register(req)

    async def sse(): # async generator: one SSE line per token
        #decode the whole output each time and send only what's new: a single token can be half
        #a multi-byte character, which decoded alone shows as \ufffd; the full decode merges it
        out_ids, sent = [], ""
        while True:
            token, done = await outbox.get()
            out_ids.append(token)
            text = tok.decode(out_ids)
            if text.endswith("\ufffd") and not done:
                continue  #half a character so far; wait for the rest
            yield f"data: {json.dumps({'token': text[len(sent):], 'done': done})}\n\n"
            sent = text
            if done:
                break

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/")
async def home():
    options = "".join(f'<option value="{n}"{" selected" if n == DEFAULT else ""}>{LABELS[n]}</option>' for n in MODELS)
    return HTMLResponse(PAGE.replace("__OPTIONS__", options))


PAGE = """<!doctype html>
<title>llm-inference</title>
<style>
  body { font-family: ui-monospace, "SF Mono", Menlo, monospace; background: #fdfdfc;
         color: #1a1a1a; max-width: 920px; margin: 64px auto; padding: 0 24px;
         font-size: 13px; line-height: 1.6; }
  h1 { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 18px; }
  .muted { color: #8b8a85; }
  .small { font-size: 12px; }
  .controls { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; margin-bottom: 14px; }
  .pin, .sel { font: inherit; color: inherit; background: transparent; border: 1px solid #e4e3dd; outline: none; }
  .pin { width: 100%; box-sizing: border-box; padding: 8px 10px; }
  .pin:focus, .sel:focus { border-color: #b8b6ae; }
  .pin::placeholder { color: #b8b6ae; }
  .sel { padding: 4px 8px; }
  button { font: inherit; background: #1a1a1a; color: #fdfdfc; border: 1px solid #1a1a1a; padding: 5px 18px; cursor: pointer; }
  button:hover { background: #3d3c39; }
  button:disabled { background: #b8b6ae; border-color: #b8b6ae; cursor: default; }
  label.opt { color: #8b8a85; font-size: 12px; display: flex; align-items: center; gap: 6px; }
  #prompts { display: grid; gap: 6px; margin-bottom: 28px; }
  h2 { font-size: 12px; font-weight: 600; color: #8b8a85; text-transform: uppercase; letter-spacing: 0.06em;
       margin: 28px 0 8px; border-bottom: 1px solid #e4e3dd; padding-bottom: 6px; }
  .mech { display: grid; grid-template-columns: 300px 1fr 150px; gap: 0 24px; align-items: start;
          padding: 12px 0; border-bottom: 1px solid #f0efea; }
  .mech .name { font-weight: 600; }
  .mech .desc { color: #8b8a85; font-size: 12px; }
  .mech .status { font-size: 12px; text-align: right; color: #3d3c39; white-space: pre-line; }
  .strip { display: flex; align-items: flex-end; gap: 2px; height: 34px; margin-top: 6px; }
  .strip i { display: block; width: 6px; background: #d9d8d2; height: 2px; }
  .strip i.on { background: var(--c); }
  .rows { display: grid; gap: 3px; margin-top: 6px; }
  .rows div { display: flex; gap: 2px; height: 6px; }
  .rows b { display: block; width: 6px; background: #ecebe7; }
  .rows b.on { background: var(--c); }
  .cap { color: #8b8a85; font-size: 11px; margin-top: 4px; min-height: 15px; }
  .card { margin-top: 10px; border: 1px solid #e4e3dd; }
  .card .head { display: flex; justify-content: space-between; gap: 12px; padding: 6px 10px; background: #f6f5f1; font-size: 12px; }
  .card .head .n { color: var(--c); font-weight: 600; margin-right: 8px; }
  .card .head .stat { color: #8b8a85; white-space: nowrap; }
  .card .out { white-space: pre-wrap; color: #3d3c39; padding: 8px 10px; min-height: 40px; }
</style>
<body>
<h1>llm-inference</h1>
<div class="controls">
  <span class="muted small">model</span> <select id="model" class="sel">__OPTIONS__</select>
  <label class="opt"><input type="checkbox" id="sample"> sample (temperature 0.9, top-k 50)</label>
  <button id="run" onclick="race()">run</button>
</div>
<div id="prompts">
  <input class="pin" value="The meaning of life is" placeholder="prompt 1">
  <input class="pin" value="Once upon a time" placeholder="prompt 2">
  <input class="pin" value="def fibonacci(n):" placeholder="prompt 3">
  <input class="pin" placeholder="prompt 4">
</div>
<div class="muted small">the four mechanisms run one after another so the timings are fair (no gpu contention). greedy repeats itself; sampling breaks the loop but then outputs can't be checked for identity.</div>

<div id="results" style="display:none">
<div id="mechs"></div>
<div id="stream" style="--c:#7a72b5"></div>
</div>

<script>
const N_NEW = 40;
const MECHS = [
  ["naive",  "#b06456", "no kv cache: every step recomputes the whole context, so the work per step grows"],
  ["cached", "#6b8f5e", "kv cache: every step processes one new token; the context is read from the cache"],
  ["static", "#a89354", "static batching: all prompts left-padded into one batch, every row advances in lockstep"],
];
const model = () => document.getElementById('model').value;
const sampling = () => document.getElementById('sample').checked ? {do_sample: true, temperature: 0.9, top_k: 50} : {};
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
const $ = id => document.getElementById(id);

function bars(n) { return Array.from({length: n}, () => '<i></i>').join(''); }

async function race() {
  const prompts = [...document.querySelectorAll('input.pin')].map(el => el.value.trim()).filter(Boolean);
  if (!prompts.length || $('run').disabled) return;
  $('run').disabled = true; //one race at a time: a second click mid-race would write into the same rows
  try { await raceBody(prompts); } finally { $('run').disabled = false; }
}

async function raceBody(prompts) {
  $('results').style.display = '';

  //one row per blocking mechanism: name | work strip | status
  $('mechs').innerHTML = MECHS.map(([m, c, desc]) => `
    <div class="mech" style="--c:${c}">
      <div><div class="name" style="color:${c}">${m}</div><div class="desc">${desc}</div></div>
      <div>${m === 'static'
        ? `<div class="rows" id="rows-${m}">${prompts.map(() => '<div>' + '<b></b>'.repeat(N_NEW) + '</div>').join('')}</div>`
        : `<div class="strip" id="strip-${m}">${bars(N_NEW)}</div>`}
        <div class="cap" id="cap-${m}"></div></div>
      <div class="status" id="status-${m}">waiting</div>
    </div>`).join('');

  //continuous: the fourth mechanism row, then one live card per prompt
  $('stream').innerHTML = `<div class="mech">
      <div><div class="name" style="color:#7a72b5">continuous</div><div class="desc">continuous batching: all prompts in one shared batch, each streamed live as its slot is scheduled</div></div>
      <div></div><div class="status" id="sum-continuous">waiting</div></div>` +
    prompts.map((p, i) => `<div class="card"><div class="head"><span><span class="n">${i + 1}</span>${esc(p.slice(0, 70))}</span><span class="stat" id="s-c${i}"></span></div><div id="o-c${i}" class="out"></div></div>`).join('');
  const sampled = !!sampling().do_sample; //read once, so a mid-run toggle can't change the verdict

  const results = {};
  for (const [m] of MECHS) {
    $('status-' + m).textContent = 'running';
    results[m] = m === 'static' ? await runStatic(prompts) : await runSequential(m, prompts);
  }
  $('sum-continuous').textContent = 'running';
  const t0 = performance.now();
  const ref = await Promise.all(prompts.map((p, i) => runContinuous(p, i)));
  $('sum-continuous').textContent = `all ${prompts.length} in ${((performance.now() - t0) / 1000).toFixed(2)}s`;
  for (const [m] of MECHS) {
    const same = results[m].length === ref.length && results[m].every((t, i) => t === ref[i]);
    $('status-' + m).textContent += '\\n' + (sampled ? 'sampled, not compared' : same ? 'output identical \\u2713' : 'OUTPUT DIVERGED \\u2717');
  }
}

//draw one step of work: naive/cached -> a bar whose height is the tokens touched; static -> the rows advance
function drawStep(m, ev, maxWork, promptLens) {
  if (m === 'static') {
    const rows = $('rows-' + m).children;
    for (let r = 0; r < rows.length; r++) rows[r].children[ev.step - 1].classList.add('on');
    $('cap-' + m).textContent = `step ${ev.step}/${ev.of}: ${ev.work} rows x 1 token, context ${ev.ctx} (left-padded)`;
  } else {
    const bar = $('strip-' + m).children[ev.step - 1];
    bar.style.height = Math.max(2, Math.round(34 * ev.work / maxWork)) + 'px';
    bar.classList.add('on');
    $('cap-' + m).textContent = m === 'naive'
      ? `step ${ev.step}/${ev.of}: recomputing ${ev.work} tokens`
      : `step ${ev.step}/${ev.of}: 1 new token, ${ev.ctx - 1} from the cache`;
  }
}

async function runSequential(mode, prompts) {
  const texts = [];
  const t0 = performance.now();
  for (const [k, prompt] of prompts.entries()) {
    const resp = await fetch('/generate', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, max_new_tokens: N_NEW, mode, model: model(), ...sampling()})});
    //the strip shows the current prompt's steps; reset it per prompt
    [...$('strip-' + mode).children].forEach(b => { b.className = ''; b.style.height = '2px'; });
    const ev = await readSSE(resp, ev => { if (ev.step) drawStep(mode, ev, (ev.ctx - ev.step) + N_NEW); });
    texts.push(ev.token);
    $('status-' + mode).textContent = `prompt ${k + 1}/${prompts.length} done, ${((performance.now() - t0) / 1000).toFixed(1)}s so far`;
  }
  $('status-' + mode).textContent = `all ${prompts.length} in ${((performance.now() - t0) / 1000).toFixed(2)}s`;
  return texts;
}

async function runStatic(prompts) {
  const t0 = performance.now();
  const resp = await fetch('/generate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: prompts[0], prompts, max_new_tokens: N_NEW, mode: 'static', model: model(), ...sampling()})});
  const ev = await readSSE(resp, ev => { if (ev.step) drawStep('static', ev); });
  $('status-static').textContent = `all ${prompts.length} in ${((performance.now() - t0) / 1000).toFixed(2)}s`;
  return ev.texts;
}

async function runContinuous(prompt, i) {
  const out = $('o-c' + i), stat = $('s-c' + i);
  const t0 = performance.now();
  let ttft = null, text = '', n = 0;
  stat.textContent = '...';
  const resp = await fetch('/generate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt, max_new_tokens: N_NEW, model: model(), ...sampling()})});
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
</body>"""
