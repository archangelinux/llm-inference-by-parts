import asyncio
from engine.scheduler import Engine, Request

class EngineLoop:

    def __init__(self, engine):
        self.engine = engine
        self.inbox = asyncio.Queue() #handlers put requests here
        self.outboxes = {} #req_id -> asyncio.Queue of (token, done) events; every request gets a private queue, created at register()

    def register(self, request) -> asyncio.Queue:
        outbox = asyncio.Queue()
        self.outboxes[request.req_id] = outbox
        self.inbox.put_nowait(request)
        return outbox

    async def run(self):
        while True:
            #idle -> sleep until a request arrives (costs nothing, wakes instantly)
            if not (self.engine.waiting or any(r is not None for r in self.engine.running)):
                req = await self.inbox.get()
                self.engine.submit(req)

            #drain inbox queue without blocking
            while True:
                try: #eafp, no race conditions between check and get
                    self.engine.submit(self.inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # take one lap off-loop on a side thread; event loop can run next stream/other tasks while GPU works
            events = await asyncio.to_thread(self.engine.step) #recall step() returns (req_id, token, done)

            # sort the mail (back on the event loop here - queues are safe again)
            for req_id, token, done in events:
                self.outboxes[req_id].put_nowait((token, done))
                if done:
                    del self.outboxes[req_id]  # handler keeps its own reference; drop ours

if __name__ == "__main__":
    #smoke test, no HTTP: two concurrent requests, tokens must print as generated but interleaved
    from transformers import GPT2Tokenizer
    from engine.config import DEVICE, GPTConfig, DTYPE
    from engine.model import GPT

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT.from_pretrained(GPTConfig()).to(DEVICE, DTYPE).eval()
    loop = EngineLoop(Engine(model=model, n_slots=2, max_len=128))

    async def stream(name, prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        outbox = loop.register(Request(prompt_ids=ids, max_new_tokens=15))
        while True:
            token, done = await outbox.get()
            print(f"[{name}] {tok.decode([token])!r}", flush=True)
            if done:
                break

    async def main():
        task = asyncio.create_task(loop.run())# start, loop implicit
        await asyncio.gather(stream("A", "Hello"),  # for loop to take on two "handlers" at once
                             stream("B", "The meaning of life is"))
        task.cancel()# crude shutdown, fine for a smoke test

    asyncio.run(main()) #event loop starts, runs main() as the first task