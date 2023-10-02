import logging
from typing import Union
from modal import Image, Stub, method, gpu, Secret, Function

logging.basicConfig(level=logging.INFO)

image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.8",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "GitPython",
        "colorama",
        "fire",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers",
        "llama-recipes @ git+https://github.com/modal-labs/llama-recipes.git@6636910761b70ada964409960129c5a4e9c2c049",
        extra_index_url="https://download.pytorch.org/whl/nightly/cu118",
        pre=True,
    )
    .pip_install("huggingface_hub==0.17.1", "hf-transfer==0.1.3", "scipy")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)
stub = Stub()

@stub.function(gpu=gpu.A100(memory=80, count=2), image=image)
def train_on_modal(cmd: str):
    """
    this is literally a wrapper to just clone the code on github and then run it
    :param cmd: the command to run, e.g. torchrun --nproc-per-node 2 dummy_ddp.py
    """
    print("hello, world!")