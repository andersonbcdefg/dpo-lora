import fire
import torch
from typing import Literal, Union
from transformers import AutoModelForCausalLM
import numpy as np
from utils import (
    print_trainable_parameters,
    forward_batch
)
from data import get_dataloader
from models import (
    get_model_and_tokenizer,
    get_optimizer_for_model
)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def train(
    model_name: str,
    datasets: Union[str, list[str]], # path to dataset(s) on huggingface. must have (prompt, chosen, rejected)
    quantization: Literal["4bit", "8bit", None] = None,
    loss_fn: Literal["dpo", "sft"] = "dpo",
    batch_size: int = 16,
    accum_steps: int = 1,
    lr: float = 2.0e-5,
    num_workers: int = 4,
):

    # get LoRA model
    model, tokenizer = get_model_and_tokenizer(
        model_name=model_name,
        gradient_checkpointing=True,
        load_in_4bit=(quantization == "4bit"),
        load_in_8bit=(quantization == "8bit"),
        lora=True,
        lora_ckpt=None,
    )
        

    # get train dataloader
    datasets = [datasets] if isinstance(datasets, str) else datasets
    dataloader = get_dataloader(
        dataset_names=datasets,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # get optimizer
    optimizer = get_optimizer_for_model(
        model, model_name, max_lr=lr
    )

    # train -- uh oh what do????
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # set to float if not on cuda
    if device == torch.device("cpu"):
        model = model.float()
    model.to(device)
  
    for i, batch in enumerate(dataloader):
        loss, metrics = forward_batch(model, batch, device, loss_fn=loss_fn, train=True)
        print("Loss: ", loss)
        for metric in metrics:
            writer.add_scalar(metric, np.mean(metrics[metric]), i)
        (loss / accum_steps).backward()

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    
if __name__ == "__main__":
    fire.Fire(train)
