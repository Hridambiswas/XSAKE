"""
Sequence packing, padding, and batch construction for XSAKE.

Sequence packing (also called "sample packing") concatenates tokenised documents
into a single stream and slices it into fixed-length chunks. This avoids wasting
compute on padding tokens and is standard for large-scale LM pre-training.

Labels are the input_ids shifted left by one position (next-token prediction).
Positions that cross document boundaries are masked with -1 in labels so the
cross-entropy loss does not penalise predicting the first token of a new document.
"""

from __future__ import annotations
from typing import Iterator

import numpy as np
import jax.numpy as jnp


def pack_sequences(
    token_ids: np.ndarray,
    seq_len: int,
    eos_id: int = 50256,
) -> np.ndarray:
    """
    Slice a flat token stream into fixed-length sequences.

    Args:
        token_ids: 1D int32 array of all token ids (concatenated documents)
        seq_len:   tokens per sequence
        eos_id:    end-of-text token id (used to detect document boundaries)

    Returns:
        [n_sequences, seq_len] int32 array
    """
    n = len(token_ids) // seq_len
    packed = token_ids[: n * seq_len].reshape(n, seq_len)
    return packed.astype(np.int32)


def make_labels(input_ids: np.ndarray, eos_id: int = 50256) -> np.ndarray:
    """
    Shift input_ids left by one to create next-token prediction labels.

    Positions that ARE the eos token in the input are set to -1 in labels
    (the loss function skips -1 labels).

    Args:
        input_ids: [batch, seq] int32

    Returns:
        [batch, seq] int32 — next-token labels, -1 at eos positions
    """
    labels = np.roll(input_ids, -1, axis=-1)
    labels[:, -1] = -1   # last position has no target

    # Mask positions where input is eos (document boundary)
    eos_positions = input_ids == eos_id
    labels[eos_positions] = -1

    return labels


def make_batch_iterator(
    sequences: np.ndarray,
    batch_size: int,
    seed: int = 42,
    infinite: bool = True,
) -> Iterator[dict]:
    """
    Yields batches of {"input_ids": ..., "labels": ...} from packed sequences.

    Shuffles the sequence order at the start of each epoch.

    Args:
        sequences:  [n_sequences, seq_len] int32
        batch_size: sequences per batch
        seed:       RNG seed for shuffling
        infinite:   if True, loops forever (for training); False = one epoch

    Yields:
        dict with JAX arrays "input_ids" [batch, seq] and "labels" [batch, seq]
    """
    rng = np.random.default_rng(seed)
    n   = len(sequences)

    while True:
        idx = rng.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx  = idx[start : start + batch_size]
            input_ids  = sequences[batch_idx]
            labels     = make_labels(input_ids)
            yield {
                "input_ids": jnp.array(input_ids),
                "labels":    jnp.array(labels),
            }
        if not infinite:
            break
