"""Streaming-dataset plumbing shared across experiments.

The same class of validity bug kept recurring in per-experiment streaming
implementations:

- DataLoader workers duplicating data: every worker iterates its own copy of
  an IterableDataset, so without a worker-id offset they all yield the same
  stream (fixed independently in selective_pe f77c8c4 and delta_graft).
- Stale data across epochs: re-iterating with a seed that ignores the epoch
  counter repeats the previous epoch's data (lego d19bbc8).
- Train/val leakage when sharding a skipped stream: ``ds.shard()`` after
  ``ds.skip(val_docs)`` splits the skip across worker shards, so the workers
  don't collectively skip the global first ``val_docs`` documents
  (delta_graft 6d49dc3).

Use these wrappers instead of writing a new IterableDataset from scratch.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from torch.utils.data import IterableDataset, get_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterator


T = TypeVar("T")


def stream_rng(seed: int, *, epoch: int = 0, worker_id: int = 0) -> random.Random:
    """RNG seeded distinctly per (seed, epoch, worker).

    Seeding with a string avoids the additive collisions of
    ``seed + epoch + worker_id`` (seed=42/epoch=1 colliding with
    seed=43/epoch=0). ``random.Random`` hashes str seeds with a
    deterministic, salt-independent scheme, so this is reproducible
    across processes and runs.
    """
    return random.Random(f"{seed}:{epoch}:{worker_id}")


def worker_share(n: int, num_workers: int, worker_id: int) -> int:
    """How many of ``n`` total examples this worker should yield.

    The shares sum to exactly ``n`` across workers.
    """
    return n // num_workers + (1 if worker_id < n % num_workers else 0)


def buffered_shuffle[S](
    items: Iterator[S], rng: random.Random, buffer_size: int
) -> Iterator[S]:
    """Stream ``items`` in shuffled order using a fixed-size buffer.

    Same idea as HF's streaming ``.shuffle()``, but as a plain Python
    transform so it composes with skip/take splits without re-seeding the
    upstream pipeline.
    """
    buffer: list[S] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        idx = rng.randrange(buffer_size)
        yield buffer[idx]
        buffer[idx] = item
    rng.shuffle(buffer)
    yield from buffer


class SyntheticStream(IterableDataset[T], ABC):
    """Base class for streaming datasets that generate examples from an RNG.

    Handles the validity-critical plumbing once:

    - mixes the DataLoader worker id into the seed so workers don't yield
      duplicate examples
    - shards ``n_examples`` across workers so one pass over the DataLoader
      yields exactly ``n_examples`` examples in total
    - mixes the epoch counter into the seed so re-iterating yields fresh data

    Subclasses implement :meth:`generate`.

    The automatic epoch counter only advances with ``num_workers=0`` or
    ``persistent_workers=True`` (otherwise each epoch re-forks workers from
    the parent copy, whose counter never advanced). Prefer single-epoch
    streaming (``--n-epochs 1`` with a larger ``n_examples``) per CLAUDE.md;
    for genuine multi-epoch use with non-persistent workers, call
    :meth:`set_epoch` before each epoch like a DistributedSampler.
    """

    def __init__(self, n_examples: int, seed: int = 42) -> None:
        self.n_examples = n_examples
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @abstractmethod
    def generate(self, rng: random.Random) -> T:
        """Generate a single example using ``rng`` as the only randomness."""

    def __len__(self) -> int:
        return self.n_examples

    def __iter__(self) -> Iterator[T]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = stream_rng(self.seed, epoch=self._epoch, worker_id=worker_id)
        self._epoch += 1
        for _ in range(worker_share(self.n_examples, num_workers, worker_id)):
            yield self.generate(rng)
