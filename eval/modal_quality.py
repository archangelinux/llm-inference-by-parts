#runs eval/quality.py on an A10G (the 0.6B model thrashes MPS memory at 512-token
#chunks on the M1) and copies the results json back
# usage: MODEL=qwen modal run eval/modal_quality.py
import os
from pathlib import Path

import modal

app = modal.App("llm-inference-quality")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "datasets")
         .env({"MODEL": os.environ.get("MODEL", "gpt2")})
         .add_local_dir("engine", remote_path="/root/engine")
         .add_local_dir("kernels", remote_path="/root/kernels")
         .add_local_dir("eval", remote_path="/root/eval"))


@app.function(gpu="A10G", image=image, timeout=1800)
def run() -> dict:
    import subprocess
    subprocess.run(["python", "eval/quality.py"], check=True, cwd="/root")
    return {p.name: p.read_text() for p in Path("/root/eval").glob("quality_results*.json")}


@app.local_entrypoint()
def main():
    for name, text in run.remote().items():
        (Path(__file__).parent / name).write_text(text)
        print(f"wrote eval/{name}")
