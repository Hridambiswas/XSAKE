"""
OpenWebText data loader for XSAKE.

Downloads a subset of OpenWebText via HuggingFace datasets, tokenises it
with the GPT-2 BPE tokenizer, packs sequences into fixed-length chunks of
seq_len tokens, and exposes a tf.data pipeline with prefetching.

On Kaggle/Colab: set cache_dir to a persistent path to avoid re-downloading.
Locally: uses ~/.cache/huggingface by default.
"""

from __future__ import annotations
import os
from typing import Iterator, Optional

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from data.tokenizer import load_gpt2_tokenizer
from data.preprocessing import pack_sequences, make_batch_iterator


def load_openwebtext(
    split: str = "train",
    subset_fraction: float = 0.01,
    seq_len: int = 1024,
    batch_size: int = 8,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    tokenizer: Optional[Tokenizer] = None,
) -> Iterator[dict]:
    """
    Load a fraction of OpenWebText and return a batched token iterator.

    Args:
        split:             "train" or "test"
        subset_fraction:   fraction of the full dataset to use (0.01 = 1%)
        seq_len:           tokens per sequence
        batch_size:        sequences per batch
        cache_dir:         HuggingFace cache directory
        seed:              random seed for shuffling
        tokenizer:         if None, uses GPT-2 BPE tokenizer

    Yields:
        dict with keys "input_ids" [batch, seq_len] and "labels" [batch, seq_len]
    """
    if tokenizer is None:
        tokenizer = load_gpt2_tokenizer()

    print(f"[data] Loading OpenWebText ({split}, {subset_fraction*100:.1f}%)...")
    dataset = load_dataset(
        "Skylion007/openwebtext",
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # Sample subset
    n = int(len(dataset) * subset_fraction)
    dataset = dataset.shuffle(seed=seed).select(range(n))

    print(f"[data] Tokenising {n} documents...")
    def tokenise(example):
        ids = tokenizer.encode(example["text"]).ids
        return {"input_ids": ids}

    dataset = dataset.map(tokenise, remove_columns=["text"], num_proc=4)

    # Concatenate all token ids into one long stream
    all_ids = []
    for example in dataset:
        all_ids.extend(example["input_ids"])
        all_ids.append(tokenizer.token_to_id("<|endoftext|>") or 50256)

    print(f"[data] Total tokens: {len(all_ids):,}")

    packed = pack_sequences(np.array(all_ids, dtype=np.int32), seq_len=seq_len)
    return make_batch_iterator(packed, batch_size=batch_size, seed=seed)


def estimate_dataset_size(subset_fraction: float = 0.01) -> dict:
    """Return rough estimates without downloading the dataset."""
    full_docs    = 8_013_769
    full_tokens  = 9_035_582_198
    return {
        "documents": int(full_docs * subset_fraction),
        "tokens":    int(full_tokens * subset_fraction),
        "subset_fraction": subset_fraction,
    }
