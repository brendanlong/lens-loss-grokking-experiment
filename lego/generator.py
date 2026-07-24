"""Group composition chain generator.

Supports multiple finite groups for the LEGO composition task:
  - S3: Symmetric group on 3 elements (6 elements, non-abelian, solvable)
  - S4: Symmetric group on 4 elements (24 elements, non-abelian, solvable)
  - A5: Alternating group on 5 elements (60 elements, simple, non-solvable)
  - S5: Symmetric group on 5 elements (120 elements, non-abelian, non-solvable)

Each example is a chain: a starting element followed by k group operations.
The model must output the result of composing all operations in sequence.

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
import math
import random
from collections.abc import Iterator
from typing import Literal, NamedTuple

# --- Group definitions ---

GroupName = Literal["S3", "S4", "A5", "S5"]


class Group(NamedTuple):
    """A finite group defined by its Cayley table.

    Attributes:
        name: Human-readable name (e.g., "S3", "A5").
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


def _perm_sign(perm: tuple[int, ...]) -> int:
    """Compute the sign of a permutation (+1 for even, -1 for odd)."""
    n = len(perm)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if not visited[i]:
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len % 2 == 0:
                sign *= -1
    return sign


def _perm_label(perm: tuple[int, ...]) -> str:
    """Generate a compact label for a permutation.

    Uses cycle notation, e.g. (012) for the 3-cycle 0→1→2→0.
    Identity is labeled 'e'.
    """
    n = len(perm)
    if all(perm[i] == i for i in range(n)):
        return "e"
    visited = [False] * n
    cycles: list[str] = []
    for i in range(n):
        if visited[i] or perm[i] == i:
            visited[i] = True
            continue
        cycle = []
        j = i
        while not visited[j]:
            visited[j] = True
            cycle.append(str(j))
            j = perm[j]
        cycles.append("(" + "".join(cycle) + ")")
    return "".join(cycles)


def _build_symmetric_group(n: int) -> Group:
    """Build the symmetric group Sₙ on {0, 1, ..., n-1}."""
    all_perms = sorted(itertools.permutations(range(n)))
    assert len(all_perms) == math.factorial(n)
    elements = [_perm_label(p) for p in all_perms]
    cayley = [
        [_perm_to_index(_compose_perm(a, b), all_perms) for b in all_perms]
        for a in all_perms
    ]
    return Group(name=f"S{n}", elements=elements, cayley=cayley)


def _build_alternating_group(n: int) -> Group:
    """Build the alternating group Aₙ (even permutations of {0, ..., n-1})."""
    all_perms = sorted(
        p for p in itertools.permutations(range(n)) if _perm_sign(p) == 1
    )
    assert len(all_perms) == math.factorial(n) // 2
    elements = [_perm_label(p) for p in all_perms]
    cayley = [
        [_perm_to_index(_compose_perm(a, b), all_perms) for b in all_perms]
        for a in all_perms
    ]
    return Group(name=f"A{n}", elements=elements, cayley=cayley)


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

# Pre-built groups
S4 = _build_symmetric_group(4)
A5 = _build_alternating_group(5)
S5 = _build_symmetric_group(5)

GROUPS: dict[GroupName, Group] = {
    "S3": S3,
    "S4": S4,
    "A5": A5,
    "S5": S5,
}

# --- Legacy aliases for S3 (used by existing code) ---
ELEMENTS: list[str] = S3.elements
N_ELEMENTS = S3.order
CAYLEY: list[list[int]] = S3.cayley


def get_group(name: GroupName) -> Group:
    """Get a group by name."""
    return GROUPS[name]


# --- Example generation ---


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


# Keep S3Example as an alias for backward compatibility
S3Example = ChainExample


def compose(left: int, right: int, group: Group = S3) -> int:
    """Compute left · right in the given group."""
    return group.cayley[left][right]


def generate_example(k: int, rng: random.Random, group: Group = S3) -> ChainExample:
    """Generate one chain with exactly k operations.

    Args:
        k: Number of operations (chain length). Must be >= 0.
            k=0 is the identity case: answer = start element.
        rng: Random number generator.
        group: The group to use for composition.
    """
    if k < 0:
        msg = f"k must be >= 0, got {k}"
        raise ValueError(msg)

    n = group.order
    start = rng.randint(0, n - 1)
    ops = tuple(rng.randint(0, n - 1) for _ in range(k))

    trajectory: list[int] = [start]
    state = start
    for op in ops:
        state = compose(op, state, group)
        trajectory.append(state)

    return ChainExample(start=start, ops=ops, trajectory=tuple(trajectory))


def verify_trajectory(example: ChainExample, group: Group = S3) -> bool:
    """Verify that trajectory is consistent with start and ops."""
    if len(example.trajectory) != len(example.ops) + 1:
        return False
    if example.trajectory[0] != example.start:
        return False
    state = example.start
    for i, op in enumerate(example.ops):
        state = compose(op, state, group)
        if example.trajectory[i + 1] != state:
            return False
    return True


def generate_stream(
    k_min: int,
    k_max: int,
    n_examples: int,
    seed: int = 42,
    group: Group = S3,
) -> Iterator[ChainExample]:
    """Stream chain examples with random chain lengths."""
    rng = random.Random(seed)
    for _ in range(n_examples):
        k = rng.randint(k_min, k_max)
        yield generate_example(k, rng, group)


def generate_fixed_dataset(
    k: int,
    n_examples: int,
    seed: int = 42,
    group: Group = S3,
) -> list[ChainExample]:
    """Generate a fixed dataset where all chains have length k."""
    rng = random.Random(seed)
    return [generate_example(k, rng, group) for _ in range(n_examples)]


def generate_mixed_dataset(
    k_min: int,
    k_max: int,
    n_examples: int,
    seed: int = 42,
    group: Group = S3,
) -> list[ChainExample]:
    """Generate a fixed dataset with random chain lengths in [k_min, k_max]."""
    return list(generate_stream(k_min, k_max, n_examples, seed, group))
