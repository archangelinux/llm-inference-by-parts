#Triton dev loop for Modal: ships kernels/ to an A10G and runs main() of the file specified
#each kernel file defines main(): build inputs, run the kernel, compare against a torch reference, print match + timing (see _template.py)
# usage: modal run kernels/dev.py --name vector_add   (runs kernels/vector_add.py)
import os

import modal

app = modal.App("llm-inference-kernels")

#pip line matches modal_bench.py so modal reuses that cached image build
#no need to pip install triton: the linux/CUDA torch build already includes it (MPS torch doesn't)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers")
         .env({"MODEL": os.environ.get("MODEL", "gpt2")})
         .add_local_dir("engine", remote_path="/root/engine")
         .add_local_dir("kernels", remote_path="/root/kernels")
         .add_local_dir("tests", remote_path="/root/tests"))


@app.function(gpu="A10G", image=image, timeout=600)
def run(name: str):
    import importlib
    mod = importlib.import_module(f"kernels.{name}")
    mod.main()

@app.function(gpu="A10G", image=image, timeout=1200)
def run_tests():
    #the quant test suite on cuda, fp16 -- the dtype the kernel path serves.
    #on cuda test_kernel actually runs and quantize_model takes the kernel path
    import os
    import subprocess
    env = {**os.environ, "DTYPE": "fp16"}
    subprocess.run(["python", "tests/test_quant.py"], check=True, cwd="/root", env=env)


@app.function(gpu="A10G", image=image, timeout=1800)
def run_engine_tests():
    #the continuous-batching gates on cuda, where Engine.step captures the decode forward
    #as a cuda graph: gpt-2 scheduler test (fp32, eager solo vs graphed engine), qwen engine
    #test, and the quantized-kernel path through the engine (fp16)
    import os
    import subprocess
    subprocess.run(["python", "tests/test_scheduler.py"], check=True, cwd="/root")
    subprocess.run(["python", "-c", "import tests.test_qwen as t; t.test_engine()"], check=True, cwd="/root")
    subprocess.run(["python", "-c", """
import torch
from engine.config import DEVICE
from engine.load import load_model
from engine.quant import quantize_model
from engine.scheduler import Engine, Request
model, tok = load_model()
quantize_model(model)
ids = [tok(p, return_tensors='pt').input_ids.to(DEVICE) for p in ['Hello', 'def fibonacci(n):', 'The meaning of life is']]
e = Engine(model=model, n_slots=2, max_len=128)
reqs = [Request(prompt_ids=i, max_new_tokens=20) for i in ids]
for r in reqs: e.submit(r)
e.run()
assert e.graph is not None, 'graph was not captured'
for r, i in zip(reqs, ids):
    solo = model.generate(i, max_new_tokens=20)[0, i.shape[1]:].tolist()
    assert r.output_ids == solo, tok.decode(r.output_ids)
print('int8-kernel engine: graph captured, all match solo')
"""], check=True, cwd="/root", env={**os.environ, "DTYPE": "fp16"})


@app.local_entrypoint()
def main(name: str = "vector_add"):
    #modal run kernels/dev.py --name matmul      -> runs kernels/matmul.py's main()
    #modal run kernels/dev.py --name tests       -> runs tests/test_quant.py (kernel path live)
    if name == "tests":
        run_tests.remote()
    elif name == "engine_tests":
        run_engine_tests.remote()
    else:
        run.remote(name)
