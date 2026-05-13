"""Prefetching data pipeline with background thread buffering."""
from __future__ import annotations

import queue
import threading
from typing import Iterator, Optional

import jax.numpy as jnp
import numpy as np


class PrefetchIterator:
    """Wraps any iterator and prefetches `buffer_size` batches on a background thread."""

    def __init__(self, source: Iterator, buffer_size: int = 4):
        self._source = source
        self._buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._sentinel = object()
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _fill(self) -> None:
        try:
            for item in self._source:
                self._buffer.put(item)
        finally:
            self._buffer.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._buffer.get()
        if item is self._sentinel:
            raise StopIteration
        return item


class DataPipeline:
    """Thin wrapper that adds prefetch + device transfer to any batch iterator."""

    def __init__(
        self,
        source: Iterator,
        prefetch_size: int = 4,
        dtype: jnp.dtype = jnp.int32,
    ):
        self._prefetch = PrefetchIterator(source, buffer_size=prefetch_size)
        self._dtype = dtype

    def __iter__(self):
        for batch in self._prefetch:
            if isinstance(batch, dict):
                yield {k: jnp.array(v, dtype=self._dtype) for k, v in batch.items()}
            else:
                yield jnp.array(batch, dtype=self._dtype)


def make_pipeline(
    source: Iterator,
    prefetch_size: int = 4,
    dtype: jnp.dtype = jnp.int32,
) -> DataPipeline:
    return DataPipeline(source, prefetch_size=prefetch_size, dtype=dtype)


def infinite_cycle(source_fn, *args, **kwargs) -> Iterator:
    """Call `source_fn(*args, **kwargs)` repeatedly to produce an infinite stream."""
    while True:
        yield from source_fn(*args, **kwargs)


class ShardedPipeline:
    """Splits each batch into `n_shards` equal pieces for pmap."""

    def __init__(self, source: Iterator, n_shards: int, prefetch_size: int = 4):
        self._inner = DataPipeline(source, prefetch_size=prefetch_size)
        self._n = n_shards

    def __iter__(self):
        for batch in self._inner:
            if isinstance(batch, dict):
                yield {
                    k: v.reshape((self._n, v.shape[0] // self._n, *v.shape[1:]))
                    for k, v in batch.items()
                }
            else:
                yield batch.reshape((self._n, batch.shape[0] // self._n, *batch.shape[1:]))
