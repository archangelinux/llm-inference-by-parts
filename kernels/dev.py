#Triton dev loop for Modal: ships kernels/ to an A10G and runs main() of the file specified
#each kernel file defines main(): build inputs, run the kernel, compare against a torch reference, print match + timing (see _template.py)
# usage: modal run kernels/dev.py --name vector_add   (runs kernels/vector_add.py)
import modal

app = modal.App("llm-inference-kernels")

#pip line matches modal_bench.py so modal reuses that cached image build
#no need to pip install triton: the linux/CUDA torch build already includes it (MPS torch doesn't)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers")
         .add_local_dir("engine", remote_path="/root/engine")
         .add_local_dir("kernels", remote_path="/root/kernels"))


@app.function(gpu="A10G", image=image, timeout=600)
def run(name: str):
    import importlib
    mod = importlib.import_module(f"kernels.{name}")
    mod.main()

@app.local_entrypoint()
def main(name: str = "vector_add"):
    run.remote(name)
