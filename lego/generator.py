"""S3 group composition chain enumeration.

The LEGO composition task uses S3, the symmetric group on 3 elements
(6 elements, non-abelian, solvable).

Each example is a chain: a starting element followed by k group operations.
The model must output the result of composing all operations in sequence.

The task distribution is small enough to enumerate exhaustively (335,922
chains for S3 with k in [0, 6]), so the dataset is the *full* enumeration
with an explicit, disjoint train/test split — no sampling, no possibility
of train/test overlap.

Composition convention: **left-multiplication**.
    "Apply operation g to state x" means computing g · x.
    Given chain start=x, ops=[g₁, g₂, …, gₖ]:
        state₀ = x
        state₁ = g₁ · x
        state₂ = g₂ · (g₁ · x)
        …
        stateₖ = gₖ · … · g₂ · g₁ · x

Example (k=3, S3):
    <start> r <op> s <op> r2 <predict> [answer]
"""

import itertools
import random
from typing import NamedTuple

# --- Group definition ---


class Group(NamedTuple):
    """A finite group defined by its Cayley table.

    Attributes:
        name: Human-readable name (e.g., "S3").
        elements: Element labels, indexed 0..n-1.
        cayley: cayley[a][b] = a · b (row = left, col = right).
    """

    name: str
    elements: list[str]
    cayley: list[list[int]]

    @property
    def order(self) -> int:
        return len(self.elements)


def _perm_to_index(perm: tuple[int, ...], all_perms: list[tuple[int, ...]]) -> int:
    """Map a permutation tuple to its index in the sorted list."""
    return all_perms.index(perm)


def _compose_perm(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Compose two permutations: (a · b)(i) = a(b(i))."""
    return tuple(a[b[i]] for i in range(len(a)))


# S3 with the original element ordering (e, r, r2, s, rs, r2s) for
# backward compatibility with existing checkpoints and tests.
_S3_PERMS = [
    (0, 1, 2),  # e   — identity
    (1, 2, 0),  # r   — rotation 120°
    (2, 0, 1),  # r2  — rotation 240°
    (0, 2, 1),  # s   — reflection
    (1, 0, 2),  # rs  — rotation then reflection
    (2, 1, 0),  # r2s — two rotations then reflection
]
_S3_LABELS = ["e", "r", "r2", "s", "rs", "r2s"]
S3 = Group(
    name="S3",
    elements=_S3_LABELS,
    cayley=[
        [_perm_to_index(_compose_perm(a, b), _S3_PERMS) for b in _S3_PERMS]
        for a in _S3_PERMS
    ],
)


# --- Example enumeration ---


class ChainExample(NamedTuple):
    """A single group composition chain.

    Attributes:
        start: Starting element index.
        ops: Operation element indices, length k.
        trajectory: Intermediate states, length k + 1.
            trajectory[0] = start
            trajectory[i] = ops[i-1] · trajectory[i-1]  (left-mult)
    """

    start: int
    ops: tuple[int, ...]
    trajectory: tuple[int, ...]


def compose(left: int, right: int) -> int:
    """Compute left · right in S3."""
    return S3.cayley[left][right]


def make_example(start: int, ops: tuple[int, ...]) -> ChainExample:
    """Build a ChainExample from a start element and op sequence.

    Computes the full trajectory via left-multiplication.
    """
    trajectory: list[int] = [start]
    state = start
    for op in ops:
        state = compose(op, state)
        trajectory.append(state)
    return ChainExample(start=start, ops=ops, trajectory=tuple(trajectory))


def enumerate_chains(k_min: int, k_max: int) -> list[ChainExample]:
    """Deterministically enumerate ALL chains with k in [k_min, k_max].

    Every (start element, op sequence) pair is generated exactly once, in a
    fixed order (increasing k, then start, then ops lexicographically), with
    trajectories computed. For a group of order n there are n * n^k chains of
    length k, so e.g. S3 with k in [0, 6] yields
    6 * (6^0 + 6^1 + ... + 6^6) = 335,922 chains — small enough to hold the
    entire task distribution in memory and split it exactly.

    Args:
        k_min: Minimum number of operations (>= 0). k=0 is the identity
            case: answer = start element.
        k_max: Maximum number of operations (inclusive).
    """
    if k_min < 0:
        msg = f"k_min must be >= 0, got {k_min}"
        raise ValueError(msg)
    if k_max < k_min:
        msg = f"k_max ({k_max}) must be >= k_min ({k_min})"
        raise ValueError(msg)

    n = S3.order
    examples: list[ChainExample] = []
    for k in range(k_min, k_max + 1):
        for start in range(n):
            for ops in itertools.product(range(n), repeat=k):
                examples.append(make_example(start, ops))
    return examples


def train_test_split(
    examples: list[ChainExample],
    test_frac: float,
    seed: int,
) -> tuple[list[ChainExample], list[ChainExample]]:
    """Split examples into disjoint train/test sets via a seeded shuffle.

    The split is **stratified by chain length k**: within each k, examples
    are shuffled with a seeded RNG and `round(test_frac * n_k)` (at least 1,
    when the stratum has more than one example) are held out for test. This
    guarantees every chain length is represented in the test set — a plain
    global shuffle can leave small strata (e.g. the 6 k=0 chains) with no
    test examples at all.

    Train and test are disjoint by construction (each input example is
    assigned to exactly one side); together they cover the full input list.

    Args:
        examples: Examples to split (typically from enumerate_chains).
        test_frac: Fraction of each stratum held out for test, in (0, 1).
        seed: RNG seed; the same (examples, test_frac, seed) always yields
            the same split.

    Returns:
        (train, test) lists, each ordered by increasing k.
    """
    if not 0.0 < test_frac < 1.0:
        msg = f"test_frac must be in (0, 1), got {test_frac}"
        raise ValueError(msg)

    rng = random.Random(seed)
    by_k = group_by_k(examples)
    train: list[ChainExample] = []
    test: list[ChainExample] = []
    for k in sorted(by_k):
        stratum = list(by_k[k])
        rng.shuffle(stratum)
        n_test = round(test_frac * len(stratum))
        if len(stratum) > 1:
            n_test = min(max(1, n_test), len(stratum) - 1)
        test.extend(stratum[:n_test])
        train.extend(stratum[n_test:])
    return train, test


def enumerate_split(
    k_min: int,
    k_max: int,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[ChainExample], list[ChainExample]]:
    """Enumerate all chains for k in [k_min, k_max] and split train/test.

    Convenience wrapper combining enumerate_chains + train_test_split so
    that training and analysis scripts reconstruct the *same* held-out test
    split from (k_min, k_max, test_frac, seed).
    """
    return train_test_split(enumerate_chains(k_min, k_max), test_frac, seed)


def subsample_k_uniform(
    examples: list[ChainExample],
    n: int,
    seed: int,
) -> list[ChainExample]:
    """Draw a fixed subset of ``n`` chains, spread as evenly as possible
    across chain-length strata (the grokking-regime memorization set).

    Allocation is a waterfill: strata are visited smallest-first, each takes
    ``min(remaining // strata_left, len(stratum))``, and any shortfall from
    small strata (k <= 1 has only a handful of chains) is redistributed to
    the larger ones. Within a stratum the choice is a seeded shuffle, so the
    same (examples, n, seed) always yields the same subset.

    Returns the subset ordered by increasing k. Raises if ``n`` exceeds the
    number of available examples.
    """
    if n > len(examples):
        msg = f"cannot subsample {n} from {len(examples)} examples"
        raise ValueError(msg)
    by_k = group_by_k(examples)
    rng = random.Random(seed)
    alloc: dict[int, int] = {}
    remaining = n
    smallest_first = sorted(by_k, key=lambda k: (len(by_k[k]), k))
    for i, k in enumerate(smallest_first):
        quota = remaining // (len(smallest_first) - i)
        alloc[k] = min(quota, len(by_k[k]))
        remaining -= alloc[k]
    # Waterfill rounding can leave a few unassigned; top up strata with room.
    for k in sorted(by_k, key=lambda k: len(by_k[k]), reverse=True):
        if remaining == 0:
            break
        extra = min(remaining, len(by_k[k]) - alloc[k])
        alloc[k] += extra
        remaining -= extra
    subset: list[ChainExample] = []
    for k in sorted(by_k):
        stratum = list(by_k[k])
        rng.shuffle(stratum)
        subset.extend(stratum[: alloc[k]])
    return subset


def group_by_k(
    examples: list[ChainExample],
) -> dict[int, list[ChainExample]]:
    """Group examples by chain length k (= len(ops)), preserving order."""
    by_k: dict[int, list[ChainExample]] = {}
    for ex in examples:
        by_k.setdefault(len(ex.ops), []).append(ex)
    return by_k


def verify_trajectory(example: ChainExample) -> bool:
    """Verify that trajectory is consistent with start and ops."""
    if len(example.trajectory) != len(example.ops) + 1:
        return False
    if example.trajectory[0] != example.start:
        return False
    state = example.start
    for i, op in enumerate(example.ops):
        state = compose(op, state)
        if example.trajectory[i + 1] != state:
            return False
    return True
