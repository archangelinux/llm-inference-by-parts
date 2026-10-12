import torch

def _new_cache(model, max_len):
    """Empty kv cache for one sequence; same layout as generate: one (k, v, counter) per layer"""
    cfg, w = model.config, model.lm_head.weight
    shape = (1, cfg.n_kv_head, max_len, cfg.head_dim)
    return [(torch.empty(shape, device=w.device, dtype=w.dtype),
             torch.empty(shape, device=w.device, dtype=w.dtype), 0)
            for _ in range(cfg.n_layer)]

def _set_len(cache, n):
    """Rollback: same buffers, counter moved to n. Columns past n become invisible and get overwritten later"""
    return [(k, v, n) for k, v, _ in cache]

@torch.no_grad()
def spec_generate(target, draft, ids, k, max_new_tokens, stats=None):
    """Greedy speculative decoding, b=1"""
    assert target.config.vocab_size == draft.config.vocab_size
    n_prompt = ids.shape[1]
    max_len = n_prompt + max_new_tokens + k      #verify writes k+1 columns past the committed text
    t_cache, d_cache = _new_cache(target, max_len), _new_cache(draft, max_len)

    #prefill: both models take the whole prompt; the target picks the first new token
    logits, t_cache = target(ids, kv_past=t_cache)
    _, d_cache = draft(ids, kv_past=d_cache)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1, vocab) -> (1, 1)
    ids = torch.cat((ids, next_id), dim=1)  # (1, t) -> (1, t+1) this new token is now "pending"

    while ids.shape[1] - n_prompt < max_new_tokens:
        feed = ids[:, d_cache[0][2]:]
        guesses = []
        for _ in range(k):
            logits, d_cache = draft(feed, kv_past=d_cache)
            g = logits[:, -1, :].argmax(dim=-1, keepdim=True) #shape (1, 1)
            guesses.append(g)
            feed = g
        guesses = torch.cat(guesses, dim=1)  #(1, k)

        #verify: one target forward over [pending, g1..gk]
        t_len = t_cache[0][2] #counter/past len before verify
            #run target on [pending, guesses] -> logits (1, k+1, vocab), t_cache
        logits, t_cache = target(torch.cat((ids[:, -1:], guesses), dim=1), kv_past=t_cache)        
        picks = logits.argmax(dim=-1)

        #compare: count accepted guesses, stop at the first mismatch
        i = 0
        while i < k and guesses[0, i] == picks[0, i]:
            i += 1

        #commit: accepted guesses + the target's own token at position i (correction, or bonus new token if i == k)
        ids = torch.cat((ids, guesses[:, :i], picks[:, i:i+1]), dim=1)

        #rollback
        keep = t_len + 1 + i
        t_cache = _set_len(t_cache, keep) 
        d_cache = _set_len(d_cache, min(d_cache[0][2], keep))  

    return ids[:, :n_prompt + max_new_tokens]  #trim because the last round can overshoot