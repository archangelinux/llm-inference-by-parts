#turns the MODEL/DTYPE knobs into a (model, tokenizer) pair., for server and benching 
#tests stay model-specific (they're bound to a fixture set)
from transformers import AutoTokenizer

from engine.config import DEVICE, DTYPE, MODEL, GPTConfig, QwenConfig

HF_ID = {"gpt2": "gpt2", "qwen": "Qwen/Qwen3-0.6B-Base"}
MODEL_LABEL = {"gpt2": "gpt-2 124m", "qwen": "qwen3 0.6b"}[MODEL]


def load_model():
    if MODEL == "qwen":
        from engine.qwen import Qwen
        model = Qwen.from_pretrained(QwenConfig())
    else:
        from engine.model import GPT
        model = GPT.from_pretrained(GPTConfig())
    tok = AutoTokenizer.from_pretrained(HF_ID[MODEL])
    return model.to(DEVICE, DTYPE).eval(), tok
