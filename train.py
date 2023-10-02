import os
import fire
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Literal, Union
from transformers import AutoModelForCausalLM
import numpy as np
from contextlib import nullcontext
from utils import (
    print_trainable_parameters,
)
from data import get_dataloader
from models import (
    get_model_and_tokenizer,
    get_optimizer_for_model
)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def init_distributed(rank: int):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    torch.cuda.set_device(f"cuda:{rank}")

    # return world size
    return torch.distributed.get_world_size()

def train(
    model_name: str,
    datasets: Union[str, list[str]], # path to dataset(s) on huggingface. must have (prompt, chosen, rejected)
    num_epochs: int = 1,
    n_samples: int = None,
    quantization: Literal["4bit", "8bit", None] = None,
    loss_fn: Literal["dpo", "sft"] = "dpo",
    batch_size: int = 16,
    accum_steps: int = 1,
    lr: float = 2.0e-5,
    num_workers: int = 4,
    save_dir: str = "checkpoint-final",
):

    # get LoRA model
    model, tokenizer = get_model_and_tokenizer(
        model_name=model_name,
        gradient_checkpointing=True,
        load_in_4bit=(quantization == "4bit"),
        load_in_8bit=(quantization == "8bit"),
        lora=True,
        lora_ckpt=None,
        device=None,
    )
        

    # get train dataloader
    datasets = [datasets] if isinstance(datasets, str) else datasets
    dataloader = get_dataloader(
        dataset_names=datasets,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=False,
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
    if quantization not in ["4bit", "8bit"]:
        # if using bitsandbytes the model will already be on the right device
        model.to(device)
    

    # train loop
    model.train()
    running_metrics = {}
    samples_so_far = 0
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch} ===")
        for i, batch in enumerate(dataloader):
            samples_so_far += batch["input_ids"].shape[0]
            loss, metrics = model(batch, loss_fn=loss_fn, train=True)
            for metric in metrics:
                if metric not in running_metrics:
                    running_metrics[metric] = []
                if isinstance(metrics[metric], list): 
                    running_metrics[metric].extend(metrics[metric])
                else:
                    running_metrics[metric].append(metrics[metric])
            (loss / accum_steps).backward()
            

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                # log the running stuff every optimizer step
                for metric in running_metrics:
                    avg = np.mean(running_metrics[metric])
                    writer.add_scalar(metric, avg, i)
                running_metrics = {}
            
            if n_samples is not None and samples_so_far >= n_samples:
                break

        # save model
        model.save_pretrained(save_dir + f"_epoch_{epoch + 1}")

        # break if we've seen enough samples
        if n_samples is not None and samples_so_far >= n_samples:
            break

def train_ddp(
    model_name: str,
    datasets: Union[str, list[str]], # path to dataset(s) on huggingface. must have (prompt, chosen, rejected)
    num_epochs: int = 1,
    quantization: Literal["4bit", "8bit", None] = None,
    loss_fn: Literal["dpo", "sft"] = "dpo",
    batch_size: int = 16,
    accum_steps: int = 1,
    lr: float = 2.0e-5,
    num_workers: int = 4,
    save_dir: str = "checkpoint-final",
    rank: int = None,
):
    # initialize distributed
    print("Environment world size: ", os.environ.get("WORLD_SIZE", None))
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", None))
        if rank is None:
            raise ValueError("Couldn't get rank.")
    print(f"Hello from device {rank}!")
    world_size = init_distributed(rank)
    assert world_size > 1, "Must have more than one GPU to use DDP"
    assert accum_steps % world_size == 0, "Accumulation steps must be divisible by world size"
    accum_steps = accum_steps // world_size # we want total accumulation steps to be the same no matter hardware

    # get LoRA model. do this on rank 0 first.
    if rank == 0:
        # Download model weights
        print("Downloading model weights on rank 0")
        model, tokenizer = get_model_and_tokenizer(
            model_name=model_name,
            gradient_checkpointing=True,
            load_in_4bit=(quantization == "4bit"),
            load_in_8bit=(quantization == "8bit"),
            lora=True,
            lora_ckpt=None,
            device=f"cuda:{rank}",
        )
        print("Done downloading model weights on rank 0")
    torch.distributed.barrier() # wait for rank 0 to finish downloading
    if rank != 0:
        model, tokenizer = get_model_and_tokenizer(
            model_name=model_name,
            gradient_checkpointing=False,
            load_in_4bit=(quantization == "4bit"),
            load_in_8bit=(quantization == "8bit"),
            lora=True,
            lora_ckpt=None,
            device=f"cuda:{rank}",
        )

    # wrap model in DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # model._set_static_graph() # this is necessary for gradient checkpointing to work

    # get train dataloader
    datasets = [datasets] if isinstance(datasets, str) else datasets
    dataloader = get_dataloader(
        dataset_names=datasets,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=True,
    )

    # get optimizer
    optimizer = get_optimizer_for_model(
        model, model_name, max_lr=lr
    )

    # train -- uh oh what do????
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    # set to float if not on cuda
    if device == torch.device("cpu"):
        model = model.float()
    if quantization not in ["4bit", "8bit"]:
        # if using bitsandbytes the model will already be on the right device
        model.to(device)
    

    # train loop
    model.train()
    running_losses = []
    running_metrics = {}
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        print(f"=== Epoch {epoch} ===")
        for i, batch in enumerate(dataloader):
            with model.no_sync() if (i + 1) % accum_steps != 0 else nullcontext():
                loss, metrics = model(batch, loss_fn=loss_fn, train=True)
                running_losses.append(loss.item())
                for metric in metrics:
                    if metric not in running_metrics:
                        running_metrics[metric] = []
                    if isinstance(metrics[metric], list): 
                        running_metrics[metric].extend(metrics[metric])
                    else:
                        running_metrics[metric].append(metrics[metric])
                print("about to backward...")
                (loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # only log on rank 0 every optimizer step
                loss_tensor = torch.tensor(running_losses).to(device)
                torch.distributed.all_reduce(loss_tensor)

                if rank == 0:
                    avg_loss = loss_tensor.mean().item() / world_size
                    writer.add_scalar("loss", avg_loss, i)
                running_losses = []

                for metric, values in running_metrics.items():
                    metric_tensor = torch.tensor(values).to(device)
                    torch.distributed.all_reduce(metric_tensor)
                    if rank == 0:
                        avg = metric_tensor.mean().item() / world_size
                        writer.add_scalar(metric, avg, i)
                
                running_metrics = {}


        # save model at the end, but only rank 0
        if torch.distributed.get_rank() == 0:
            model.module.save_pretrained(save_dir + f"_epoch_{epoch + 1}")

    
    
    
if __name__ == "__main__":
    fire.Fire()
