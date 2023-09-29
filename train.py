import fire
import torch
from typing import Literal, Union
from transformers import AutoModelForCausalLM
from utils import (
    print_trainable_parameters,
    forward_batch
)
from data import get_dataloader
from models import (
    get_model_and_tokenizer,
    get_optimizer_for_model
)
# automatic mixed precision
from torch.cuda.amp import GradScaler, autocast

def train(
    model_name: str,
    datasets: Union[str, list[str]], # path to dataset(s) on huggingface. must have (prompt, chosen, rejected)
    quantization: Literal["4bit", "8bit", None] = None,
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
        batch_size=16,
        num_workers=4,
    )

    # get optimizer
    optimizer = get_optimizer_for_model(
        model, model_name, max_lr=2.0e-5
    )

    # train -- uh oh what do????
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # set to float if not on cuda
    if device == torch.device("cpu"):
        model = model.float()
    model.to(device)
        
    for batch in dataloader:
        loss, metrics = forward_batch(model, batch, device)
        print("Loss: ", loss)
        print("Reward Accuracy: ", metrics["rewards_train/accuracies"])
        # print("Logps chosen: ", metrics["logps_train/chosen"])
        # print("Logps rejected: ", metrics["logps_train/rejected"])
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
if __name__ == "__main__":
    fire.Fire(train)
