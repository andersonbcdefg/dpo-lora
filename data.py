import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
# from utils import get_local_dir, TemporarilySeededRandom
# from torch.nn.utils.rnn import pad_sequence

import tqdm
import random
import numpy as np
from functools import partial
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple, Literal
from transformers import DefaultDataCollator

def get_dataset(name: str, split: str):
    """
    Load the given dataset by name. Supports any HuggingFace dataset. MUST have columns
    'prompt', 'chosen', and 'rejected'.
    """
    dataset = datasets.load_dataset(name, split=split)
    keys = dataset.features.keys()
    assert 'prompt' in keys, f"Dataset {name} does not have 'prompt' column"
    assert 'chosen' in keys, f"Dataset {name} does not have 'chosen' column"
    assert 'rejected' in keys, f"Dataset {name} does not have 'rejected' column"

    return dataset.select_columns(['prompt', 'chosen', 'rejected'])

def tokenize_batch(batch: dict, tokenizer: AutoTokenizer, add_bos: bool = False, add_eos: bool = True):
    """
    Just tokenizes a batch. Don't worry about truncation/padding yet.
    """
    bos_token = [] if not add_bos else [tokenizer.bos_token_id]
    bos_attn_mask = [] if not add_bos else [1]
    eos_token = [] if not add_eos else [tokenizer.eos_token_id]
    eos_attn_mask = [] if not add_eos else [1]

    prompt_tokenized = tokenizer(batch['prompt'], add_special_tokens=False)
    chosen_tokenized = tokenizer(batch['chosen'], add_special_tokens=False)
    rejected_tokenized = tokenizer(batch['rejected'], add_special_tokens=False)

    return {
        'prompt_input_ids': [bos_token + seq for seq in prompt_tokenized['input_ids']],
        'prompt_attention_mask': [bos_attn_mask + seq for seq in prompt_tokenized['attention_mask']],
        'chosen_input_ids': [seq + eos_token for seq in chosen_tokenized['input_ids']],
        'chosen_attention_mask': [seq + eos_attn_mask for seq in chosen_tokenized['attention_mask']],
        'rejected_input_ids': [seq + eos_token for seq in rejected_tokenized['input_ids']],
        'rejected_attention_mask': [seq + eos_attn_mask for seq in rejected_tokenized['attention_mask']],
    }

def trim_sequence_and_get_labels(
    sequence_dict: dict, 
    tokenizer: AutoTokenizer,
    max_length: int, 
    min_prompt_length: int, 
    truncation_mode: Literal['keep_start', 'keep_end']  = 'keep_end'
):
    longer_response_length = max(
        len(sequence_dict['chosen_input_ids']), 
        len(sequence_dict['rejected_input_ids'])
    )

    # if combined sequence is too long, truncate the prompt
    if len(sequence_dict['prompt_input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            sequence_dict = {k: (v[:min_prompt_length] if k.startswith("prompt") else v) for k, v in sequence_dict.items()}
        elif truncation_mode == 'keep_end':
            sequence_dict = {k: (v[-min_prompt_length:] if k.startswith("prompt") else v) for k, v in sequence_dict.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(sequence_dict['prompt_input_ids']) + longer_response_length > max_length:
        sequence_dict = {k: (v[:max_length - min_prompt_length] if k.startswith("chosen") or k.startswith("rejected") else v) for k, v in sequence_dict.items()}

    # Create full sequences & labels
    result = {}
    prompt_length, chosen_length, rejected_length = len(sequence_dict['prompt_input_ids']), len(sequence_dict['chosen_input_ids']), len(sequence_dict['rejected_input_ids'])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    result['chosen_input_ids'] = sequence_dict['prompt_input_ids'] + sequence_dict['chosen_input_ids'] + [pad_token_id] * (max_length - prompt_length - chosen_length)
    result['chosen_attention_mask'] = sequence_dict['prompt_attention_mask'] + sequence_dict['chosen_attention_mask'] + [0] * (max_length - prompt_length - chosen_length)
    result['chosen_labels'] = [-100] * prompt_length + sequence_dict['chosen_input_ids'] + [-100] * (max_length - prompt_length - chosen_length)

    result['rejected_input_ids'] = sequence_dict['prompt_input_ids'] + sequence_dict['rejected_input_ids'] + [pad_token_id] * (max_length - prompt_length - rejected_length)
    result['rejected_attention_mask'] = sequence_dict['prompt_attention_mask'] + sequence_dict['rejected_attention_mask'] + [0] * (max_length - prompt_length - rejected_length)
    result['rejected_labels'] = [-100] * prompt_length + sequence_dict['rejected_input_ids'] + [-100] * (max_length - prompt_length - rejected_length)

    # make sure the sequence is equal to max length
    assert len(result['chosen_input_ids']) == max_length
    assert len(result['chosen_attention_mask']) == max_length
    assert len(result['chosen_labels']) == max_length
    assert len(result['rejected_input_ids']) == max_length
    assert len(result['rejected_attention_mask']) == max_length
    assert len(result['rejected_labels']) == max_length

    return result

def tokenize_dataset(
    dataset: datasets.Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    min_prompt_length: int,
):
    # first tokenize with no padding/truncation
    tokenize_partial = partial(tokenize_batch, tokenizer=tokenizer)
    dataset = dataset.map(tokenize_partial, batched=True, batch_size=1000, remove_columns=['prompt', 'chosen', 'rejected']) # , load_from_cache_file=False)

    # now trim sequences and get labels
    dataset = dataset.map(
        lambda x: trim_sequence_and_get_labels(x, tokenizer, max_length, min_prompt_length),
        remove_columns=[x for x in dataset.column_names if x.startswith("prompt")],
        # load_from_cache_file=False,
    )

    return dataset

def get_dataloader(
    dataset_names: List[str],
    tokenizer,
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    max_length: int = 512,
    min_prompt_length: int = 128,
    silent: bool = False,
    seed:int = 42,
):
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        dataset_names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        min_prompt_length: Prompt will never be truncated to less than this length.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
    """
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()
    if len(dataset_names) == 1:
        dataset = get_dataset(dataset_names[0], split)
    elif len(dataset_names) > 1:
        dataset = datasets.concatenate_datasets([get_dataset(name, split) for name in dataset_names])
    else:
        raise ValueError("Must specify at least one dataset name")
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    dataset = tokenize_dataset(dataset, tokenizer, max_length, min_prompt_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=DefaultDataCollator(),
        pin_memory=True,
    )
    
    return dataloader