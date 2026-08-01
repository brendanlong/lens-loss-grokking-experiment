"""Tokenizer for S3 group composition chains.

Sequence format:
    <start> elem <op> elem <op> elem … <predict> answer

For a chain of k operations, sequence length = 2k + 4:
    1 (<start>) + 1 (start elem) + 2k (<op> elem pairs) + 1 (<predict>) + 1 (answer)

Vocabulary layout (N + 4 tokens, N = 6 elements for S3):
    0:            <pad>
    1..N:         group elements
    N+1:          <start>  (marks initial element)
    N+2:          <op>     (marks each operation)
    N+3:          <predict> (marks answer position)
"""

from lego.generator import S3, ChainExample

PAD_ID = 0
ELEMENT_OFFSET = 1
START_TOKEN = S3.order + 1
OP_TOKEN = S3.order + 2
PREDICT_TOKEN = S3.order + 3
VOCAB_SIZE = S3.order + 4


def element_token(idx: int) -> int:
    """Convert element index (0-5) to token ID (1-6)."""
    if not 0 <= idx < S3.order:
        msg = f"Element index must be 0-{S3.order - 1}, got {idx}"
        raise ValueError(msg)
    return ELEMENT_OFFSET + idx


def token_to_str(token_id: int) -> str:
    """Convert a token ID to a human-readable string."""
    if token_id == PAD_ID:
        return "<pad>"
    if ELEMENT_OFFSET <= token_id < ELEMENT_OFFSET + S3.order:
        return S3.elements[token_id - ELEMENT_OFFSET]
    if token_id == START_TOKEN:
        return "<start>"
    if token_id == OP_TOKEN:
        return "<op>"
    if token_id == PREDICT_TOKEN:
        return "<predict>"
    msg = f"Unknown token ID: {token_id}"
    raise ValueError(msg)


def encode(example: ChainExample) -> list[int]:
    """Encode a chain example as a token sequence (no padding)."""
    tokens = [START_TOKEN, element_token(example.start)]
    for op in example.ops:
        tokens.extend([OP_TOKEN, element_token(op)])
    tokens.append(PREDICT_TOKEN)
    tokens.append(element_token(example.trajectory[-1]))
    return tokens


def encode_padded(example: ChainExample, k_max: int) -> list[int]:
    """Encode and pad to the max sequence length for k_max operations."""
    tokens = encode(example)
    tokens.extend([PAD_ID] * (seq_len(k_max) - len(tokens)))
    return tokens


def decode(token_ids: list[int]) -> str:
    """Decode token IDs to a human-readable string (skipping padding)."""
    return " ".join(token_to_str(t) for t in token_ids if t != PAD_ID)


def seq_len(k: int) -> int:
    """Total sequence length for a chain with k operations."""
    return 2 * k + 4


def answer_position(k: int) -> int:
    """0-indexed position of the answer token in the sequence."""
    return 2 * k + 3
