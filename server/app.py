import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig
from engine.model import GPT
from engine.scheduler import Engine, Request
from server.engine_loop import EngineLoop

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()
eloop = EngineLoop(Engine(model=model, n_slots=4, max_len=512))

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(eloop.run())  # start the loop when the server boots
    yield # server runs here
    task.cancel()  # and stops it on shutdown

app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(body: dict):
    ids = tok(body["prompt"], return_tensors="pt").input_ids.to(DEVICE)
    req = Request(prompt_ids=ids, max_new_tokens=body.get("max_new_tokens", 50))
    outbox = eloop.register(req)

    async def sse(): # async generator: one SSE line per token
        while True:
            token, done = await outbox.get()
            yield f"data: {json.dumps({'token': tok.decode([token]), 'done': done})}\n\n"
            if done:
                break

    return StreamingResponse(sse(), media_type="text/event-stream")