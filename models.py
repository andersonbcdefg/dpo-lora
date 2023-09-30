from logger import logger

import fire
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from utils import print_trainable_parameters
from transformers.optimization import Adafactor
from torch.optim import AdamW

from config import TrainingConfig

from registry import MODEL_REGISTRY, LORA_MODULES

def get_quantization_config(load_in_4bit=False, load_in_8bit=False):
    if not (load_in_4bit or load_in_8bit):
        return None
    if load_in_4bit and load_in_8bit:
        logger.error("You can't load a model in both 4-bit and 8-bit precision.")
        raise ValueError("You can't load a model in both 4-bit and 8-bit precision.")
    if load_in_8bit:
        return BitsAndBytesConfig(
            load_in_4bit=False,
            load_in_8bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

## TODO: replace this with config: TrainingConfig (can't right now because export depends on being able to fiddle with it)
def get_model_and_tokenizer(
    model_name,
    gradient_checkpointing=True,
    load_in_4bit=False,
    load_in_8bit=False,
    lora=False,
    lora_ckpt=None,
):  
    model_type = "CausalLM"
    if model_type not in ["CausalLM", "Seq2SeqLM", "Classification"]:
        logger.error(f"Model type {model_type} not recognized.")
        raise ValueError(f"Model type {model_type} not recognized.")
    if load_in_4bit and load_in_8bit:
        logger.error("You can't load a model in both 4-bit and 8-bit precision.")
        raise ValueError("You can't load a model in both 4-bit and 8-bit precision.")
    
    if not lora and (load_in_4bit or load_in_8bit):
        logger.error("You can't load a model in 4-bit or 8-bit precision without LoRA.")
        raise ValueError("You can't load a model in 4-bit or 8-bit precision without LoRA.")
    
    logger.info(f"Getting model and tokenizer for {model_name}.")
    model_config = MODEL_REGISTRY[model_name]

    quantization_config = get_quantization_config(load_in_4bit, load_in_8bit)

    auto_config = AutoConfig.from_pretrained(model_config["path"], trust_remote_code=True)
    # if model_config["family"] == "GPTNeoX":
    #     auto_config.rope_scaling = {"type": "linear", "factor": 1.5}
    lora_config = None
    if lora:
        logger.info("LoRA is enabled; setting base model dropout to 0.0.")
        dropout_keys = [x for x in auto_config.__dict__.keys() if "dropout" in x or "pdrop" in x]
        for key in dropout_keys:
            auto_config.__dict__[key] = 0.0
        
    if lora and not lora_ckpt:
        task_type = None
        if model_type == "CausalLM":
            task_type = TaskType.CAUSAL_LM
        elif model_type == "Seq2SeqLM":
            task_type = TaskType.SEQ_2_SEQ_LM
        elif model_type == "Classification":
            task_type = TaskType.SEQ_CLS
        else:
            raise ValueError(f"No PEFT task type available for {model_type}.")

        lora_config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=LORA_MODULES[model_config["family"]], 
            lora_dropout=0.05, 
            bias="none", 
            task_type=task_type,
            layers_to_transform=list(range(36, 49)) if "CodeLlama" in model_name else None,
        )

    if "use_cache" in auto_config.__dict__:
        logger.info("Disabling KV cache for training.")
        auto_config.use_cache=False
    use_bf16 = False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("CUDA is available and supports BF16, using BF16.")
        use_bf16 = True
    model = AutoModelForCausalLM.from_pretrained(
        model_config["path"], 
        trust_remote_code=True,
        config=auto_config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        offload_folder="./offload"
    )

    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing.")
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            logger.warn(f"Unable to unable checkpointing for model {model_name}: {e}")

    if load_in_4bit or load_in_8bit:
        logger.info("Quantization is enabled, preparing model for k-bit training.")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing and model_config["family"] != "MPT")

    if lora and not lora_ckpt:
        model.add_adapter(lora_config)
        print_trainable_parameters(model)

    elif lora and lora_ckpt:
        model.load_adapter(lora_ckpt)

    elif lora and lora_ckpt:
        model = PeftModel.from_pretrained(
            model,
            lora_ckpt,
            offload_dir="./offload",
            offload_folder="./offload"
        )
        # mark the lora params as trainable
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True
    use_fast_tokenizer = model_config["fast_tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"], trust_remote_code=True, use_fast=use_fast_tokenizer)
    return model, tokenizer

def get_optimizer_for_model(model, model_name, max_lr=None):
    model_config = MODEL_REGISTRY[model_name]
    model_family = model_config["family"]

    # use adafactor for seq2seq (T5), adamw otherwise
    if model_family == "seq2seq":
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-3 if max_lr is None else max_lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=1e-5 if max_lr is None else max_lr,
        )
    return optimizer

def test_get_configs():
    for model_name in MODEL_REGISTRY:
        print()
        logger.info(f"\n=== Testing configs for {model_name} ===")
        model_config = MODEL_REGISTRY[model_name]
        auto_config = AutoConfig.from_pretrained(model_config["path"], trust_remote_code=True)
        dropout_keys = [x for x in auto_config.__dict__.keys() if "dropout" in x or "pdrop" in x]
        for key in dropout_keys:
            auto_config.__dict__[key] = 0.0
            logger.info("After setting dropout to 0.0, ", [auto_config.__dict__[key] for key in dropout_keys])

def test_get_model_and_tokenizer():
    for model_name in MODEL_REGISTRY:
        
        # full precision
        print()
        logger.info(f"Testing {model_name} with full precision, without LoRA.")
        model, tokenizer = get_model_and_tokenizer(model_name)

        # full precision with lora
        print()
        logger.info(f"Testing {model_name} with full precision, with LoRA.")
        model, tokenizer = get_model_and_tokenizer(model_name, lora=True)

        # if cuda, try 4-bit and 8-bit
        if torch.cuda.is_available():
            logger.info("CUDA is available, testing 4-bit and 8-bit precision.")

            # 4-bit
            print()
            logger.info(f"Testing {model_name} with 8-bit precision, with LoRA.")
            model, tokenizer = get_model_and_tokenizer(model_name, load_in_4bit=True, lora=True)

            # 8-bit
            print()
            logger.info(f"Testing {model_name} with 4-bit precision, with LoRA.")
            model, tokenizer = get_model_and_tokenizer(model_name, load_in_8bit=True, lora=True)


if __name__ == "__main__":
    fire.Fire()