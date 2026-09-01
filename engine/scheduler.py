'''for continuous batching! based on ORCA's paper iteration-level scheduling, just without distributed workers, and pad align instead of selective batching'''

import torch 
from collections import deque 
from dataclasses import dataclass, field # for the Request class
from engine.model import PAD_TOKEN, GPT, GPTConfig, DEVICE

#postcodition of the whole system: when done=True and output_ids holds the same as what solo greedy generation would have produced, up to and including EOS or budget length (max_new_tokens)
@dataclass
class Request:
    prompt_ids: torch.Tensor #(1, P) on DEVICE
    max_new_tokens: int
    eos_id : int | None = None
    output_ids: list = field(default_factory=list)
    done: bool = False

class Engine:

    def __init__(self, model, n_slots, max_len):
        self.model = model
        self.cfg = model.config
        self.n_slots = n_slots
        self.max_len = max_len
        w = model.lm_head.weight #choose any for device and dtype
        head_size = self.cfg.n_embd // self.cfg.n_head

        #allocates workspace, ledger and counters
        self.waiting = deque()
        self.completed = list()
        self.running = [None] * n_slots
        self.frontier = 0 #columns filled so far (like the shared past_len in generate, except this one outlives any one batch); the frontier'th column is where the next write will go

        #pairs; the counter isn't a storable state here
        self.kv_buffers = [
            (torch.empty(n_slots, self.cfg.n_head, max_len, head_size, device=w.device, dtype=w.dtype),
            torch.empty(n_slots, self.cfg.n_head, max_len, head_size, device=w.device, dtype=w.dtype))
            for _ in range(self.cfg.n_layer)
        ]

        self.mask = torch.zeros((n_slots, max_len), dtype=torch.long, device=w.device) 
        self.next_id = torch.full((n_slots, 1), PAD_TOKEN, dtype=torch.long, device=w.device) #hold one token each slot will feed the model next

    def submit(self, request) -> None:
        self.waiting.append(request)

    def _evict(self) -> None:
        for slot, request in enumerate(self.running):
            if request is not None and request.done:
                self.completed.append(request)
                self.running[slot] = None
                self.next_id[slot] = PAD_TOKEN
                self.mask[slot] = 0 # fill row with zeros

    def _admit(self) -> None:
        isNewBatch = all(r is None for r in self.running) #new batch
        for slot in range(self.n_slots): 
            if self.running[slot] is None and self.waiting: 
                peek = self.waiting[0]
                if (peek.prompt_ids.shape[1] <= self.frontier or (isNewBatch)) and (peek.prompt_ids.shape[1] if isNewBatch else self.frontier) + peek.max_new_tokens + 1 <= self.max_len:
                    new_req = self.waiting.popleft()
                    p = new_req.prompt_ids.shape[1]
                    if isNewBatch: 
                        self.frontier = p
                        isNewBatch = False
                    slot_cache = [(k[slot:slot+1], v[slot:slot+1], self.frontier - p) #gets the whole k, v for each slot (slices out dim 0); self.frontier - p is the counter
                                for (k, v) in self.kv_buffers]
                    self.mask[slot, self.frontier - p : self.frontier] = 1 #squeeze and mask
                    logits, _ = self.model(new_req.prompt_ids, kv_past=slot_cache, attn_mask=self.mask[slot:slot+1, :self.frontier]) #prefill this slot
                    self._intake(slot, logits, new_req)
                
            
    #logit argmax selection + tokenization + recording to output + eos/budget detection atomically follows every model call (prefill + decode)
    def _intake(self, slot, logits, request):
        pick = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self.next_id[slot] = pick
        request.output_ids.append(pick.item()) #not the full tensor with device and stuff
        self.running[slot] = request
        #detect eos or budget
        if self.next_id[slot] == request.eos_id or len(request.output_ids) >= request.max_new_tokens:
            request.done = True

    def step(self) -> None:
        self._evict()
        self._admit()
        #grow mask
        for slot, request in enumerate(self.running):
            if request is not None:
                self.mask[slot, self.frontier] = 1

        # single batch decode call
        step_cache = [(k,v,self.frontier) for (k, v) in self.kv_buffers] #all rows, counter = frontier
        logits, _ = self.model(self.next_id, kv_past=step_cache, attn_mask=self.mask[:, :self.frontier + 1])  #(n_slots, 1)
        for slot, request in enumerate(self.running):
            if request is not None:
                self._intake(slot, logits[slot:slot+1], request)
        self.frontier += 1
        
    def run(self):
        while self.waiting or any(r is not None for r in self.running):
            self.step()
        return self.completed

if __name__ == "__main__":
    #smoke test: tokenization stays out here at the boundary - the Engine only ever sees Request objects holding id tensors
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()
    scheduler = Engine(model = model, n_slots = 2, max_len = 64)
    for s in ["Hello", "Today is a very"]:
        ids = tok(s, return_tensors="pt").input_ids.to(DEVICE)
        scheduler.submit(Request(prompt_ids=ids, max_new_tokens=8))
    for r in scheduler.run():
        print(r.output_ids, "->", repr(tok.decode(r.output_ids)))

