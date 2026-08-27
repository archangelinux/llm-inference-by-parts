from transformers import GPT2LMHeadModel, GPT2Tokenizer #from huggingface; lmhead is the final linear layer, tokenizer is subword via BPE (bypte-pair encoding)
import torch
from engine.config import DEVICE, GPTConfig
from engine.model import GPT

# HF
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)

prompt = "The meaning of life is"
enc = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    out = model.generate(enc.input_ids.to(DEVICE), attention_mask=enc.attention_mask.to(DEVICE), max_new_tokens=20, do_sample=False)
print(tokenizer.decode(out[0]))


#compare
#.embd intermediary module naming difference
#for register buffer, could set persistent = False to match HF but doenst matter for loading
#c_attn.weight: (768, 2304) --> OpenAI's Conv1D (in, out), mine is (2304, 768) Linear's (out, in) with transpose
mine = GPT(GPTConfig())

for k, v in model.state_dict().items():
    print(f"{k:50s} {tuple(v.shape)}")
print("----")
for k, v in mine.state_dict().items():
    print(f"{k:50s} {tuple(v.shape)}")