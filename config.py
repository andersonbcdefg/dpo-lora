import jinja2
from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass
class TrainingConfig:
    """
    A dataclass to configure the training process of a model.

    Args:
        model_name (str): Name of the model to be trained.
        run_name (str): Name of the training run.
        train_file (str): Path to the training data file.
        save_dir (str, optional): Directory where checkpoints should be saved. Defaults to "checkpoints".
        lora (bool, optional): Whether to use LoRA (Learning Rate Adaptation). Defaults to True.
        quantization (str, optional): Quantization level of the model (e.g., "4bit"). Defaults to "4bit".
        val_file (str, optional): Path to the validation data file. Defaults to None.
        num_epochs (int, optional): Number of training epochs. Defaults to 4.
        prompt_column (str, optional): Name of the column in the training data that contains the prompts. Defaults to "prompt".
        completion_column (str, optional): Name of the column in the training data that contains the completions. Defaults to "completion".
        is_classification (bool, optional): Whether the task is a classification task. Defaults to False.
        classification_label_to_id (dict, optional): A dictionary mapping labels to IDs for classification tasks. Defaults to None.
        max_lr (float, optional): Maximum learning rate. Defaults to None.
        wandb_api_key (str, optional): API key for Weights & Biases logging. Defaults to None.
        wandb_project_name (str, optional): Project name for Weights & Biases logging. Defaults to None.
        supabase_url (str, optional): URL for Supabase database. Defaults to None.
        supabase_admin_key (str, optional): Admin key for Supabase database. Defaults to None.
        supabase_user_id (str, optional): User ID for Supabase database. Defaults to None.
        supabase_job_id (str, optional): Job ID for Supabase database. Defaults to None.
        supabase_log_every (int, optional): Frequency of logging to Supabase database. Defaults to 25.
    """
    model_name: str
    run_name: str
    num_epochs: int
    train_dataset: str
    train_dataset_split: Optional[str] = None
    save_dir: str = "checkpoints"
    lora: bool = True
    lora_layers_last_k: Optional[int] = None
    max_lr: Optional[float] = None
    quantization: Optional[str] = "4bit"
    val_dataset: Optional[str] = None
    val_dataset_split: Optional[str] = None
    # distinction between instruction/input/output (dataframe) 
    # vs. prompt/completion for the model. if instruction provided,
    # it will be concatenated with the input column.
    chat_model: bool = False
    instruction_column: Optional[str] = None
    input_column: Optional[str] = None
    output_column: Union[str, list[str], None] = None # will be list[str] for multilabel classification
    messages_column: Optional[str] = None
    # prompt and completion suffixes for generative models
    prompt_suffix: Optional[str] = "\n[END INPUT]\n"
    completion_suffix: Optional[str] = "@@@"
    # chat settings for chat models
    user_prefix: Optional[str] = "User: "
    assistant_prefix: Optional[str] = "Assistant: "
    system_prefix: Optional[str] = "System: "
    message_separator: Optional[str] = "\n\n"
    # classification settings
    is_classification: bool = False
    multilabel_classification: bool = False
    classification_label_to_id: Optional[Dict[str, int]] = None
    # dpo settings
    is_preference_training: bool = False
    
    # packing, padding, truncation settings -- CURRENTLY UNUSED
    pack_sequences: Union[bool, str, None] = None
    pad_and_truncate: Optional[bool] = None
    max_input_len: Optional[int] = None
    # logging
    wandb_api_key: Optional[str] = None
    wandb_project_name: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_admin_key: Optional[str] = None
    supabase_user_id: Optional[str] = None
    supabase_job_id: Union[int, str, None] = None
    supabase_log_every: int = 25
    user_email: Optional[str] = None
    # exporting
    upload_adapter: bool = True
    upload_full_model: bool = True
    upload_ctranslate: bool = False
    upload_ggml: bool = False

    

@dataclass
class DPOConfig:
    sft_policy: str # should be path to huggingface repo for now
    datasets: list[str]
    n_epochs: Optional[int]
    n_examples: Optional[int]
    max_prompt_length: int
    max_length: int
    lr: float
    warmup_steps: int
    eval_every: int
    do_first_eval: bool
    sample_during_eval: bool
    n_eval_model_samples: int 
    n_eval_examples: int
    batch_size: int
    eval_batch_size: int
    loss_beta: float
    reference_free: bool
    max_grad_norm: float
    gradient_accumulation_steps: int
    debug: bool = False
    # wandb settings & logging
    wandb_api_key: Optional[str] = None
    wandb_project_name: Optional[str] = None
    minimum_log_interval_secs: int = 60
    local_dirs: tuple[str] = ("cache",)

    def __post_init__(self):
        pass

    @property
    def use_wandb(self):
        return self.wandb_api_key is not None and self.wandb_project_name is not None