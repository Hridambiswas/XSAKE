"""
BPE tokenizer loading for XSAKE — uses GPT-2's vocabulary via HuggingFace tokenizers.

GPT-2's BPE tokenizer has:
  vocab_size = 50,257  (50,000 BPE merges + 256 byte tokens + <|endoftext|>)
  No explicit [PAD] token — padding uses <|endoftext|> (id=50256)

We use the HuggingFace `tokenizers` library (Rust-backed) for fast batch
tokenisation during data preprocessing.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import GPT2TokenizerFast


_CACHED_TOKENIZER: Optional[Tokenizer] = None


def load_gpt2_tokenizer(cache_dir: Optional[str] = None) -> Tokenizer:
    """
    Load the GPT-2 BPE tokenizer from HuggingFace hub.

    Caches the tokenizer in memory for repeated calls.

    Returns:
        tokenizers.Tokenizer (Rust-backed, thread-safe)
    """
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is not None:
        return _CACHED_TOKENIZER

    hf_tok = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=cache_dir)
    _CACHED_TOKENIZER = hf_tok.backend_tokenizer
    return _CACHED_TOKENIZER


def encode_batch(texts: List[str], tokenizer: Optional[Tokenizer] = None) -> List[List[int]]:
    """
    Tokenise a list of strings.

    Returns:
        list of token id lists
    """
    tok = tokenizer or load_gpt2_tokenizer()
    encodings = tok.encode_batch(texts)
    return [e.ids for e in encodings]


def decode(ids: List[int], tokenizer: Optional[Tokenizer] = None) -> str:
    tok = tokenizer or load_gpt2_tokenizer()
    return tok.decode(ids)


def vocab_size() -> int:
    return 50_257


def pad_token_id() -> int:
    return 50_256   # <|endoftext|> used as padding
