#one place that turns the MODEL/DTYPE knobs into a (model, tokenizer) pair.
#tests stay model-specific (they're bound to a fixture set); server and benches use this.
from transformers import AutoTokenizer

from engine.config import DEVICE, DTYPE, MODEL, GPTConfig, QwenConfig

HF_ID = {"gpt2": "gpt2", "qwen": "Qwen/Qwen3-0.6B-Base"}
LABELS = {"gpt2": "gpt-2 124m", "qwen": "qwen3 0.6b"}
MODEL_LABEL = LABELS[MODEL]


def load_model(name=MODEL):
    if name == "qwen":
        from engine.qwen import Qwen
        model = Qwen.from_pretrained(QwenConfig())
    else:
        from engine.model import GPT
        model = GPT.from_pretrained(GPTConfig())
    tok = AutoTokenizer.from_pretrained(HF_ID[name])
    return model.to(DEVICE, DTYPE).eval(), tok


def load_models(names):
    return {n: load_model(n) for n in names}
