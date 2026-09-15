#one next-token rule shared by generate_batch (both models) and the engine.
#greedy (argmax) is the default and is what every correctness test relies on;
#sampling is opt-in per call / per request
import torch
import torch.nn.functional as F


def pick_next(logits, do_sample=False, temperature=1.0, top_k=None):
    """logits: (b, vocab) for the last position -> (b, 1) token ids"""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature  # <1 sharpens, >1 flattens
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))  # only the k best can be drawn
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
