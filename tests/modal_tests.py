#qwen fixtures + test_qwen.py on an A10G, for sizes too big for local memory
    #(1.7b fp32 is ~7GB per copy; from_pretrained holds ours + HF's at once)
    #generates tests/fixtures/<checkpoint>/ remotely, runs the tests against them, copies the fixtures back

# usage: modal run tests/modal_tests.py --size 1.7b
from pathlib import Path

import modal

app = modal.App("llm-inference-tests")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers==4.56.2", "pytest") #match local: transformers 5 loads checkpoints in bf16 by default
         .add_local_dir("engine", remote_path="/root/engine")
         .add_local_dir("kernels", remote_path="/root/kernels")
         .add_local_dir("tests", remote_path="/root/tests", ignore=["fixtures"]))


@app.function(gpu="A10G", image=image, timeout=3600)
def run(size):
    import os
    import subprocess

    from engine.config import QWEN_PRESETS

    hf_id = QWEN_PRESETS[size].hf_id
    env = {**os.environ, "PYTHONPATH": "/root", "QWEN": size}
    subprocess.run(["python", "tests/make_fixtures.py", hf_id], cwd="/root", env=env, check=True)
    tests = subprocess.run(["python", "-m", "pytest", "-v", "-s", "tests/test_qwen.py"], cwd="/root", env=env,
                           capture_output=True, text=True)

    fixtures = Path("/root/tests/fixtures") / hf_id.split("/")[-1].lower()
    files = {p.name: p.read_bytes() for p in fixtures.iterdir()}
    return fixtures.name, files, tests.stdout + tests.stderr, tests.returncode


@app.local_entrypoint()
def main(size: str = "1.7b"):
    name, files, log, code = run.remote(size)
    out = Path(__file__).parent / "fixtures" / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, data in files.items():
        (out / fname).write_bytes(data)
    print(log)
    print(f"wrote {len(files)} fixture files to {out}; pytest exit code {code}")
