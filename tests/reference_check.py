from transformers import GPT2LMHeadModel, GPT2Tokenizer #from huggingface; lmhead is the final linear layer, tokenizer is subword via BPE (bypte-pair encoding)
import torch
from engine.config import DEVICE

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)

prompt = "The meaning of life is"
enc = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    out = model.generate(enc.input_ids.to(DEVICE), attention_mask=enc.attention_mask.to(DEVICE), max_new_tokens=20, do_sample=False)
print(tokenizer.decode(out[0]))