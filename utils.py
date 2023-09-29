import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Union, List
import random
import numpy as np


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    lora_modules = 0
    for n, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if "lora" in n.lower():
            lora_modules += 1
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    print("Number of LoRA modules: ", lora_modules / 2)

def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def concatenated_forward(
    model: nn.Module, 
    batch: Dict[str, Union[List, torch.LongTensor]],
    device: torch.device
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = {
            "input_ids": torch.cat([batch['chosen_input_ids'], batch['rejected_input_ids']], dim=0).to(device),
            "attention_mask": torch.cat([batch['chosen_attention_mask'], batch['rejected_attention_mask']], dim=0).to(device),
            "labels": torch.cat([batch['chosen_labels'], batch['rejected_labels']], dim=0).to(device),
        }
        all_logits = model(concatenated_batch['input_ids'], attention_mask=concatenated_batch['attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['labels'], average_log_prob=False)
        chosen_logps, rejected_logps = all_logps.chunk(2, dim=0)
        return chosen_logps, rejected_logps

def forward_batch(model, batch, device, train=True):
    metrics = {}
    train_test = 'train' if train else 'eval'

    # turn on LoRA to get the reference model activations
    model.enable_adapters()
    policy_chosen_logps, policy_rejected_logps = concatenated_forward(model, batch, device)

    # turn off LoRA to get the reference model activations. no gradients here.
    model.disable_adapters()
    with torch.no_grad():
        reference_chosen_logps, reference_rejected_logps = concatenated_forward(model, batch, device)
    
    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1, reference_free=False)
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    # i removed all_gather_if_needed from all of these. will have to add back if doing FSDP/etc.
    # metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.detach().cpu().numpy().tolist()
    # metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.detach().cpu().numpy().tolist()
    # metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.detach().cpu().numpy().tolist()
    # metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).detach().cpu().numpy().tolist()
    # metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.detach().cpu().numpy().tolist()    
    # metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.detach().cpu().numpy().tolist()
    # metrics[f'loss/{train_test}'] = losses.detach().cpu().numpy().tolist()

    return losses.mean(), metrics

class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)

## TODO: Write a special 'forward' that keeps both the LoRA'd and unLoRA'd activations.
## Then we can sample from the policy model and reference model at the same time.