"""Tokenizer for group composition chains.

Sequence format:
    <start> elem <op> elem <op> elem … <predict> answer

For a chain of k operations, sequence length = 2k + 4:
    1 (<start>) + 1 (start elem) + 2k (<op> elem pairs) + 1 (<predict>) + 1 (answer)

Vocabulary layout (N_ELEMENTS + 4 tokens):
    0:            <pad>
    1..N:         group elements (N = group order)
    N+1:          <start>  (marks initial element)
    N+2:          <op>     (marks each operation)
    N+3:          <predict> (marks answer position)
"""

from lego.generator import S3, Group, S3Example


class Tokenizer:
    """Group-aware tokenizer for composition chains.

    Token layout:
        0:         <pad>
        1..n:      group elements (n = group.order)
        n+1:       <start>
        n+2:       <op>
        n+3:       <predict>
    """

    def __init__(self, group: Group) -> None:
        self.group = group
        self.n_elements = group.order
        self.pad_id = 0
        self.element_offset = 1
        self.start_token = group.order + 1
        self.op_token = group.order + 2
        self.predict_token = group.order + 3
        self.vocab_size = group.order + 4

    def element_token(self, idx: int) -> int:
        """Convert element index to token ID."""
        if not 0 <= idx < self.n_elements:
            msg = f"Element index must be 0-{self.n_elements - 1}, got {idx}"
            raise ValueError(msg)
        return self.element_offset + idx

    def element_index(self, token_id: int) -> int:
        """Convert element token ID to element index."""
        idx = token_id - self.element_offset
        if not 0 <= idx < self.n_elements:
            msg = f"Not an element token: {token_id}"
            raise ValueError(msg)
        return idx

    def token_to_str(self, token_id: int) -> str:
        """Convert a token ID to a human-readable string."""
        if token_id == self.pad_id:
            return "<pad>"
        if self.element_offset <= token_id < self.element_offset + self.n_elements:
            return self.group.elements[token_id - self.element_offset]
        if token_id == self.start_token:
            return "<start>"
        if token_id == self.op_token:
            return "<op>"
        if token_id == self.predict_token:
            return "<predict>"
        msg = f"Unknown token ID: {token_id}"
        raise ValueError(msg)

    def is_element_token(self, token_id: int) -> bool:
        """Check if a token ID is a group element token."""
        return self.element_offset <= token_id < self.element_offset + self.n_elements

    def encode(self, example: S3Example) -> list[int]:
        """Encode a chain example as a token sequence (no padding)."""
        tokens = [self.start_token, self.element_token(example.start)]
        for op in example.ops:
            tokens.extend([self.op_token, self.element_token(op)])
        tokens.append(self.predict_token)
        tokens.append(self.element_token(example.trajectory[-1]))
        return tokens

    def encode_padded(self, example: S3Example, k_max: int) -> list[int]:
        """Encode and pad to the max sequence length for k_max operations."""
        tokens = self.encode(example)
        target_len = seq_len(k_max)
        tokens.extend([self.pad_id] * (target_len - len(tokens)))
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to a human-readable string (skipping padding)."""
        return " ".join(self.token_to_str(t) for t in token_ids if t != self.pad_id)


# --- Default S3 tokenizer (backward compatibility) ---

_S3_TOKENIZER = Tokenizer(S3)

# Legacy constants (all S3-specific)
PAD_ID = _S3_TOKENIZER.pad_id
ELEMENT_OFFSET = _S3_TOKENIZER.element_offset
START_TOKEN = _S3_TOKENIZER.start_token
OP_TOKEN = _S3_TOKENIZER.op_token
PREDICT_TOKEN = _S3_TOKENIZER.predict_token
VOCAB_SIZE = _S3_TOKENIZER.vocab_size


# Legacy module-level functions (delegate to S3 tokenizer)
def element_token(idx: int) -> int:
    """Convert element index (0-5) to token ID (1-6). S3 only."""
    return _S3_TOKENIZER.element_token(idx)


def element_index(token_id: int) -> int:
    """Convert element token ID (1-6) to element index (0-5). S3 only."""
    return _S3_TOKENIZER.element_index(token_id)


def token_to_str(token_id: int) -> str:
    """Convert a token ID to a human-readable string. S3 only."""
    return _S3_TOKENIZER.token_to_str(token_id)


def encode(example: S3Example) -> list[int]:
    """Encode an S₃ example as a token sequence (no padding)."""
    return _S3_TOKENIZER.encode(example)


def encode_padded(example: S3Example, k_max: int) -> list[int]:
    """Encode and pad to the max sequence length for k_max operations."""
    return _S3_TOKENIZER.encode_padded(example, k_max)


def decode(token_ids: list[int]) -> str:
    """Decode token IDs to a human-readable string (skipping padding)."""
    return _S3_TOKENIZER.decode(token_ids)


def is_element_token(token_id: int) -> bool:
    """Check if a token ID is a group element token (1-6). S3 only."""
    return _S3_TOKENIZER.is_element_token(token_id)


def seq_len(k: int) -> int:
    """Total sequence length for a chain with k operations."""
    return 2 * k + 4


def answer_position(k: int) -> int:
    """0-indexed position of the answer token in the sequence."""
    return 2 * k + 3
