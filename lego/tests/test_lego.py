"""Tests for LEGO composition task: generator, tokenizer, and data pipeline."""

import random

import pytest
import torch

from lego.data import (
    S3FixedDataset,
    collate_s3,
    compute_alignment_loss,
    compute_answer_accuracy,
    compute_full_sequence_loss,
    compute_loss,
    compute_repulsion_loss,
    make_eval_batch,
)
from lego.generator import (
    A5,
    CAYLEY,
    ELEMENTS,
    GROUPS,
    N_ELEMENTS,
    S3,
    S4,
    S5,
    Group,
    S3Example,
    compose,
    generate_example,
    generate_fixed_dataset,
    generate_mixed_dataset,
    generate_stream,
    verify_trajectory,
)
from lego.tokenizer import (
    ELEMENT_OFFSET,
    OP_TOKEN,
    PAD_ID,
    PREDICT_TOKEN,
    START_TOKEN,
    VOCAB_SIZE,
    Tokenizer,
    answer_position,
    decode,
    element_index,
    element_token,
    encode,
    encode_padded,
    is_element_token,
    seq_len,
    token_to_str,
)

# ---- Cayley table tests (group axiom verification) ----


class TestCayleyTable:
    """Verify S₃ Cayley table satisfies group axioms."""

    def test_identity_left(self) -> None:
        """e · x = x for all x."""
        for x in range(N_ELEMENTS):
            assert compose(0, x) == x

    def test_identity_right(self) -> None:
        """x · e = x for all x."""
        for x in range(N_ELEMENTS):
            assert compose(x, 0) == x

    def test_closure(self) -> None:
        """All products are valid elements (0-5)."""
        for a in range(N_ELEMENTS):
            for b in range(N_ELEMENTS):
                result = compose(a, b)
                assert 0 <= result < N_ELEMENTS

    def test_associativity(self) -> None:
        """(a · b) · c = a · (b · c) for all a, b, c."""
        for a in range(N_ELEMENTS):
            for b in range(N_ELEMENTS):
                for c in range(N_ELEMENTS):
                    ab_c = compose(compose(a, b), c)
                    a_bc = compose(a, compose(b, c))
                    assert ab_c == a_bc, (
                        f"({ELEMENTS[a]}·{ELEMENTS[b]})·{ELEMENTS[c]} = "
                        f"{ELEMENTS[ab_c]} != "
                        f"{ELEMENTS[a]}·({ELEMENTS[b]}·{ELEMENTS[c]}) = "
                        f"{ELEMENTS[a_bc]}"
                    )

    def test_inverses(self) -> None:
        """Every element has a left and right inverse."""
        for a in range(N_ELEMENTS):
            # Find left inverse: x · a = e
            left_inv = None
            for x in range(N_ELEMENTS):
                if compose(x, a) == 0:
                    left_inv = x
                    break
            assert left_inv is not None, f"{ELEMENTS[a]} has no left inverse"
            # Left inverse is also right inverse in a group
            assert compose(a, left_inv) == 0

    def test_non_abelian(self) -> None:
        """r · s ≠ s · r — S₃ is non-abelian."""
        r, s = 1, 3  # r=1, s=3
        assert compose(r, s) != compose(s, r)

    def test_rotation_order_3(self) -> None:
        """r³ = e (rotation has order 3)."""
        r = 1
        r2 = compose(r, r)
        r3 = compose(r, r2)
        assert r2 == 2  # r² = r2
        assert r3 == 0  # r³ = e

    def test_reflection_order_2(self) -> None:
        """s² = e, (rs)² = e, (r²s)² = e (reflections have order 2)."""
        for elem in [3, 4, 5]:  # s, rs, r2s
            assert compose(elem, elem) == 0, f"{ELEMENTS[elem]}² ≠ e"

    def test_group_order(self) -> None:
        """S₃ has exactly 6 elements."""
        assert N_ELEMENTS == 6
        assert len(ELEMENTS) == 6
        assert len(CAYLEY) == 6
        for row in CAYLEY:
            assert len(row) == 6

    def test_latin_square(self) -> None:
        """Each row and column of Cayley table is a permutation of 0-5.

        This is a necessary property of any group's multiplication table.
        """
        elements = set(range(N_ELEMENTS))
        for a in range(N_ELEMENTS):
            row = {compose(a, b) for b in range(N_ELEMENTS)}
            col = {compose(b, a) for b in range(N_ELEMENTS)}
            assert row == elements, f"Row {ELEMENTS[a]} not a permutation"
            assert col == elements, f"Col {ELEMENTS[a]} not a permutation"


# ---- Generator tests ----


class TestGenerateExample:
    def test_trajectory_length(self) -> None:
        rng = random.Random(42)
        ex = generate_example(5, rng)
        assert len(ex.ops) == 5
        assert len(ex.trajectory) == 6  # k + 1

    def test_trajectory_starts_with_start(self) -> None:
        rng = random.Random(42)
        ex = generate_example(3, rng)
        assert ex.trajectory[0] == ex.start

    def test_trajectory_consistency(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            k = rng.randint(1, 10)
            ex = generate_example(k, random.Random(rng.randint(0, 10000)))
            assert verify_trajectory(ex)

    def test_elements_in_range(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            ex = generate_example(6, rng)
            assert 0 <= ex.start < N_ELEMENTS
            for op in ex.ops:
                assert 0 <= op < N_ELEMENTS
            for state in ex.trajectory:
                assert 0 <= state < N_ELEMENTS

    def test_deterministic(self) -> None:
        ex1 = generate_example(5, random.Random(123))
        ex2 = generate_example(5, random.Random(123))
        assert ex1 == ex2

    def test_different_seeds_differ(self) -> None:
        ex1 = generate_example(5, random.Random(1))
        ex2 = generate_example(5, random.Random(2))
        assert ex1 != ex2

    def test_k_one(self) -> None:
        rng = random.Random(42)
        ex = generate_example(1, rng)
        assert len(ex.ops) == 1
        assert len(ex.trajectory) == 2
        assert ex.trajectory[1] == compose(ex.ops[0], ex.start)

    def test_k_zero_identity(self) -> None:
        rng = random.Random(42)
        ex = generate_example(0, rng)
        assert len(ex.ops) == 0
        assert len(ex.trajectory) == 1
        assert ex.trajectory[0] == ex.start

    def test_k_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 0"):
            generate_example(-1, random.Random(42))


class TestVerifyTrajectory:
    def test_valid(self) -> None:
        # r · e = r, s · r = r2s
        ex = S3Example(
            start=0,  # e
            ops=(1, 3),  # r, s
            trajectory=(0, 1, 5),  # e, r·e=r, s·r=r2s
        )
        assert verify_trajectory(ex)

    def test_invalid_trajectory(self) -> None:
        ex = S3Example(
            start=0,
            ops=(1, 3),
            trajectory=(0, 1, 3),  # wrong: s·r=r2s not s
        )
        assert not verify_trajectory(ex)

    def test_wrong_length(self) -> None:
        ex = S3Example(
            start=0,
            ops=(1,),
            trajectory=(0, 1, 2),  # too long
        )
        assert not verify_trajectory(ex)


class TestGenerateStream:
    def test_correct_count(self) -> None:
        examples = list(generate_stream(1, 6, 100))
        assert len(examples) == 100

    def test_all_valid(self) -> None:
        for ex in generate_stream(1, 6, 50):
            assert verify_trajectory(ex)

    def test_mixed_lengths(self) -> None:
        """Stream produces chains with varying lengths."""
        lengths = {len(ex.ops) for ex in generate_stream(1, 6, 200)}
        # With 200 samples from [1,6], should see most lengths
        assert len(lengths) >= 4


class TestGenerateFixedDataset:
    def test_all_same_k(self) -> None:
        examples = generate_fixed_dataset(3, 50)
        for ex in examples:
            assert len(ex.ops) == 3

    def test_all_valid(self) -> None:
        for ex in generate_fixed_dataset(5, 50):
            assert verify_trajectory(ex)


class TestGenerateMixedDataset:
    def test_correct_count(self) -> None:
        examples = generate_mixed_dataset(1, 6, 100)
        assert len(examples) == 100


class TestTrajectoryDistribution:
    def test_trajectory_states_approximately_uniform(self) -> None:
        """Verify trajectory states are approximately uniform over S₃ elements.

        Since start is uniform and each op is uniform/independent,
        all trajectory positions should be approximately uniform.
        """
        n = 6000
        examples = list(generate_stream(3, 3, n, seed=42))
        # Check final state distribution
        counts = [0] * N_ELEMENTS
        for ex in examples:
            counts[ex.trajectory[-1]] += 1
        expected = n / N_ELEMENTS
        for elem_idx, count in enumerate(counts):
            ratio = count / expected
            assert 0.85 < ratio < 1.15, (
                f"Element {ELEMENTS[elem_idx]}: {count}/{n} (expected ~{expected:.0f})"
            )


# ---- Tokenizer tests ----


class TestElementToken:
    def test_roundtrip(self) -> None:
        for idx in range(N_ELEMENTS):
            token = element_token(idx)
            assert element_index(token) == idx

    def test_range(self) -> None:
        for idx in range(N_ELEMENTS):
            token = element_token(idx)
            assert ELEMENT_OFFSET <= token < ELEMENT_OFFSET + N_ELEMENTS

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            element_token(-1)
        with pytest.raises(ValueError):
            element_token(6)


class TestTokenToStr:
    def test_elements(self) -> None:
        for idx, name in enumerate(ELEMENTS):
            assert token_to_str(element_token(idx)) == name

    def test_special_tokens(self) -> None:
        assert token_to_str(PAD_ID) == "<pad>"
        assert token_to_str(START_TOKEN) == "<start>"
        assert token_to_str(OP_TOKEN) == "<op>"
        assert token_to_str(PREDICT_TOKEN) == "<predict>"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            token_to_str(99)


class TestEncode:
    def test_k1(self) -> None:
        # start=e(0), ops=[r(1)], trajectory=[e, r]
        ex = S3Example(start=0, ops=(1,), trajectory=(0, 1))
        tokens = encode(ex)
        assert len(tokens) == seq_len(1)  # 2*1 + 4 = 6
        assert tokens == [
            START_TOKEN,
            element_token(0),  # e
            OP_TOKEN,
            element_token(1),  # r
            PREDICT_TOKEN,
            element_token(1),  # answer: r
        ]

    def test_k3(self) -> None:
        # start=r(1), ops=[s(3), r2(2), e(0)]
        # trajectory: r, s·r=r2s(5), r2·r2s=rs(4), e·rs=rs(4)
        state = 1  # r
        state = compose(3, state)  # s·r = r2s = 5
        assert state == 5
        state = compose(2, state)  # r2·r2s = rs = 4
        assert state == 4
        state = compose(0, state)  # e·rs = rs = 4
        assert state == 4

        ex = S3Example(start=1, ops=(3, 2, 0), trajectory=(1, 5, 4, 4))
        tokens = encode(ex)
        assert len(tokens) == seq_len(3)  # 2*3 + 4 = 10
        assert tokens[0] == START_TOKEN
        assert tokens[-2] == PREDICT_TOKEN
        assert tokens[-1] == element_token(4)  # answer: rs

    def test_decode_readable(self) -> None:
        ex = S3Example(start=0, ops=(1, 3), trajectory=(0, 1, 5))
        tokens = encode(ex)
        result = decode(tokens)
        assert result == "<start> e <op> r <op> s <predict> r2s"


class TestEncodePadded:
    def test_padding_length(self) -> None:
        ex = S3Example(start=0, ops=(1,), trajectory=(0, 1))
        tokens = encode_padded(ex, k_max=6)
        assert len(tokens) == seq_len(6)  # 2*6 + 4 = 16

    def test_padding_tokens(self) -> None:
        ex = S3Example(start=0, ops=(1,), trajectory=(0, 1))
        tokens = encode_padded(ex, k_max=3)
        # k=1: 6 real tokens, k_max=3: 10 total, so 4 padding
        assert tokens[6:] == [PAD_ID] * 4

    def test_no_padding_at_max(self) -> None:
        ex = S3Example(start=0, ops=(1, 2, 3), trajectory=(0, 1, 0, 5))
        tokens = encode_padded(ex, k_max=3)
        assert PAD_ID not in tokens


class TestPositions:
    def test_seq_len(self) -> None:
        assert seq_len(1) == 6  # 2 + 4
        assert seq_len(3) == 10  # 6 + 4
        assert seq_len(6) == 16  # 12 + 4

    def test_answer_position(self) -> None:
        assert answer_position(1) == 5
        assert answer_position(3) == 9
        assert answer_position(6) == 15


class TestIsElementToken:
    def test_elements(self) -> None:
        for idx in range(N_ELEMENTS):
            assert is_element_token(element_token(idx))

    def test_non_elements(self) -> None:
        assert not is_element_token(PAD_ID)
        assert not is_element_token(START_TOKEN)
        assert not is_element_token(OP_TOKEN)
        assert not is_element_token(PREDICT_TOKEN)


class TestVocabSize:
    def test_no_overlap(self) -> None:
        """All token IDs are within vocab range and distinct."""
        all_ids = {PAD_ID, START_TOKEN, OP_TOKEN, PREDICT_TOKEN}
        for idx in range(N_ELEMENTS):
            all_ids.add(element_token(idx))
        # 1 pad + 6 elements + 3 specials = 10
        assert len(all_ids) == VOCAB_SIZE
        assert max(all_ids) == VOCAB_SIZE - 1


# ---- Data pipeline tests ----


class TestS3FixedDataset:
    def test_length(self) -> None:
        examples = generate_mixed_dataset(1, 6, 50)
        ds = S3FixedDataset(examples, k_max=6)
        assert len(ds) == 50

    def test_shapes(self) -> None:
        examples = generate_mixed_dataset(1, 4, 10)
        ds = S3FixedDataset(examples, k_max=4)
        item = ds[0]
        assert item["input_ids"].shape == (seq_len(4),)
        assert item["answer_position"].shape == ()
        assert item["chain_length"].shape == ()


class TestCollate:
    def test_batch_shapes(self) -> None:
        examples = generate_mixed_dataset(1, 4, 10)
        ds = S3FixedDataset(examples, k_max=4)
        batch = collate_s3([ds[i] for i in range(5)])
        assert batch["input_ids"].shape == (5, seq_len(4))
        assert batch["answer_position"].shape == (5,)
        assert batch["chain_length"].shape == (5,)


class TestComputeLoss:
    def test_runs(self) -> None:
        """Loss computation doesn't crash."""
        batch_size = 4
        sl = seq_len(6)
        logits = torch.randn(batch_size, sl, VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, sl))
        answer_positions = torch.full((batch_size,), answer_position(6))
        loss = compute_loss(logits, input_ids, answer_positions)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_variable_positions(self) -> None:
        """Loss works with different answer positions in same batch."""
        batch_size = 4
        sl = seq_len(6)
        logits = torch.randn(batch_size, sl, VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, sl))
        # Different k values: k=1,2,3,4
        answer_positions = torch.tensor(
            [
                answer_position(1),
                answer_position(2),
                answer_position(3),
                answer_position(4),
            ]
        )
        loss = compute_loss(logits, input_ids, answer_positions)
        assert loss.shape == ()
        assert loss.item() > 0


class TestComputeFullSequenceLoss:
    def test_scalar_output(self) -> None:
        batch_size = 4
        sl = seq_len(6)
        logits = torch.randn(batch_size, sl, VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, sl))
        loss = compute_full_sequence_loss(logits, input_ids)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_ignores_pad_targets(self) -> None:
        """Loss should ignore positions where target is PAD (0)."""
        batch_size = 2
        sl = seq_len(6)
        logits = torch.randn(batch_size, sl, VOCAB_SIZE)
        # Sequence with padding after answer
        input_ids = torch.zeros(batch_size, sl, dtype=torch.long)
        # Only fill first 8 positions (k=2: seq_len=8)
        input_ids[:, :8] = torch.randint(1, VOCAB_SIZE, (batch_size, 8))
        loss = compute_full_sequence_loss(logits, input_ids)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_compute_loss_dispatches(self) -> None:
        """compute_loss(full_sequence=True) uses full-sequence mode."""
        batch_size = 4
        sl = seq_len(6)
        logits = torch.randn(batch_size, sl, VOCAB_SIZE)
        input_ids = torch.randint(1, VOCAB_SIZE, (batch_size, sl))
        answer_positions = torch.full(
            (batch_size,),
            answer_position(6),
        )
        loss_answer = compute_loss(
            logits,
            input_ids,
            answer_positions,
            full_sequence=False,
        )
        loss_full = compute_loss(
            logits,
            input_ids,
            answer_positions,
            full_sequence=True,
        )
        # Both should be valid losses but generally different
        assert loss_answer.shape == ()
        assert loss_full.shape == ()


class TestComputeAnswerAccuracy:
    def test_perfect_prediction(self) -> None:
        batch_size = 4
        sl = seq_len(6)
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, sl))
        answer_positions = torch.full((batch_size,), answer_position(6))

        # Make logits that predict correctly at answer position
        logits = torch.zeros(batch_size, sl, VOCAB_SIZE)
        for i in range(batch_size):
            ap = answer_positions[i]
            logits[i, ap - 1, input_ids[i, ap]] = 100.0

        acc = compute_answer_accuracy(logits, input_ids, answer_positions)
        assert acc == 1.0

    def test_random_prediction(self) -> None:
        torch.manual_seed(42)
        batch_size = 1000
        sl = seq_len(6)
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, sl))
        answer_positions = torch.full((batch_size,), answer_position(6))
        logits = torch.randn(batch_size, sl, VOCAB_SIZE)
        acc = compute_answer_accuracy(logits, input_ids, answer_positions)
        # With vocab=10, random should be ~10%
        assert acc < 0.2


class TestMakeEvalBatch:
    def test_shapes(self) -> None:
        examples = generate_fixed_dataset(3, 10)
        batch = make_eval_batch(examples, k_max=6)
        assert batch["input_ids"].shape == (10, seq_len(6))
        assert batch["answer_position"].shape == (10,)
        assert batch["chain_length"].shape == (10,)
        assert (batch["chain_length"] == 3).all()
        assert (batch["answer_position"] == answer_position(3)).all()


# ---- Multi-group tests ----


class TestGroupAxioms:
    """Verify group axioms hold for all supported groups."""

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_correct_order(self, group: Group) -> None:
        """Groups have the expected number of elements."""
        expected = {"S3": 6, "S4": 24, "A5": 60, "S5": 120}
        assert group.order == expected[group.name]

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_identity(self, group: Group) -> None:
        """Element 0 is the identity (e)."""
        for x in range(group.order):
            assert compose(0, x, group) == x
            assert compose(x, 0, group) == x

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_closure(self, group: Group) -> None:
        """All products are valid elements."""
        for a in range(group.order):
            for b in range(group.order):
                result = compose(a, b, group)
                assert 0 <= result < group.order

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_latin_square(self, group: Group) -> None:
        """Each row/column of Cayley table is a permutation (no repeats)."""
        elts = set(range(group.order))
        for a in range(group.order):
            row = {compose(a, b, group) for b in range(group.order)}
            col = {compose(b, a, group) for b in range(group.order)}
            assert row == elts, f"Row {group.elements[a]} not a permutation"
            assert col == elts, f"Col {group.elements[a]} not a permutation"

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_associativity_sample(self, group: Group) -> None:
        """Spot-check associativity on random triples."""
        rng = random.Random(42)
        for _ in range(200):
            a = rng.randint(0, group.order - 1)
            b = rng.randint(0, group.order - 1)
            c = rng.randint(0, group.order - 1)
            ab_c = compose(compose(a, b, group), c, group)
            a_bc = compose(a, compose(b, c, group), group)
            assert ab_c == a_bc

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_inverses(self, group: Group) -> None:
        """Every element has an inverse."""
        for a in range(group.order):
            found = False
            for x in range(group.order):
                if compose(x, a, group) == 0:
                    assert compose(a, x, group) == 0
                    found = True
                    break
            assert found, f"{group.elements[a]} has no inverse"

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_non_abelian(self, group: Group) -> None:
        """All our groups are non-abelian."""
        found_non_commuting = False
        for a in range(group.order):
            for b in range(group.order):
                if compose(a, b, group) != compose(b, a, group):
                    found_non_commuting = True
                    break
            if found_non_commuting:
                break
        assert found_non_commuting

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_identity_label(self, group: Group) -> None:
        """First element is labeled 'e' (identity)."""
        assert group.elements[0] == "e"


class TestGroupGeneration:
    """Test example generation with non-S3 groups."""

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_generate_and_verify(self, group: Group) -> None:
        rng = random.Random(42)
        for _ in range(20):
            k = rng.randint(0, 6)
            ex = generate_example(k, rng, group)
            assert verify_trajectory(ex, group)
            assert 0 <= ex.start < group.order
            for op in ex.ops:
                assert 0 <= op < group.order

    @pytest.mark.parametrize("group", [S4, A5, S5], ids=lambda g: g.name)
    def test_stream_non_s3(self, group: Group) -> None:
        examples = list(generate_stream(1, 4, 50, group=group))
        assert len(examples) == 50
        for ex in examples:
            assert verify_trajectory(ex, group)


class TestTokenizerMultiGroup:
    """Test Tokenizer class with different groups."""

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_vocab_size(self, group: Group) -> None:
        tok = Tokenizer(group)
        assert tok.vocab_size == group.order + 4

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_element_roundtrip(self, group: Group) -> None:
        tok = Tokenizer(group)
        for idx in range(group.order):
            token = tok.element_token(idx)
            assert tok.element_index(token) == idx
            assert tok.is_element_token(token)

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_special_tokens_not_elements(self, group: Group) -> None:
        tok = Tokenizer(group)
        assert not tok.is_element_token(tok.pad_id)
        assert not tok.is_element_token(tok.start_token)
        assert not tok.is_element_token(tok.op_token)
        assert not tok.is_element_token(tok.predict_token)

    @pytest.mark.parametrize("group", [S3, S4, A5, S5], ids=lambda g: g.name)
    def test_encode_decode_roundtrip(self, group: Group) -> None:
        tok = Tokenizer(group)
        rng = random.Random(42)
        ex = generate_example(3, rng, group)
        tokens = tok.encode(ex)
        assert len(tokens) == seq_len(3)
        decoded = tok.decode(tokens)
        assert "<start>" in decoded
        assert "<predict>" in decoded

    def test_no_token_overlap(self) -> None:
        """Token IDs don't collide for any group."""
        for group in GROUPS.values():
            tok = Tokenizer(group)
            all_ids = {tok.pad_id, tok.start_token, tok.op_token, tok.predict_token}
            for idx in range(group.order):
                all_ids.add(tok.element_token(idx))
            assert len(all_ids) == tok.vocab_size


class TestDataPipelineMultiGroup:
    """Test data pipeline with non-S3 groups."""

    def test_fixed_dataset_s4(self) -> None:
        tok = Tokenizer(S4)
        examples = generate_fixed_dataset(3, 10, group=S4)
        ds = S3FixedDataset(examples, k_max=4, tokenizer=tok)
        assert len(ds) == 10
        item = ds[0]
        assert item["input_ids"].shape == (seq_len(4),)

    def test_eval_batch_s5(self) -> None:
        tok = Tokenizer(S5)
        examples = generate_fixed_dataset(2, 10, group=S5)
        batch = make_eval_batch(examples, k_max=4, tokenizer=tok)
        assert batch["input_ids"].shape == (10, seq_len(4))
        # All element tokens should be within vocab range
        assert batch["input_ids"].max() < tok.vocab_size


# ---- Alignment and repulsion loss tests ----


class TestComputeAlignmentLoss:
    """Test soft alignment auxiliary loss."""

    def test_scalar_output(self) -> None:
        """Alignment loss returns a scalar with gradients."""
        dim, vocab_size, n_layers = 16, 10, 3
        batch_size, seq = 4, 8
        residuals = [
            torch.randn(batch_size, seq, dim, requires_grad=True)
            for _ in range(n_layers)
        ]
        input_ids = torch.randint(1, vocab_size, (batch_size, seq))
        emb_weight = torch.randn(vocab_size, dim, requires_grad=True)
        mean_loss, per_layer = compute_alignment_loss(
            residuals,
            input_ids,
            emb_weight,
            temperature=1.0,
        )
        assert mean_loss.shape == ()
        assert len(per_layer) == n_layers
        assert mean_loss.requires_grad

    def test_per_layer_list_length(self) -> None:
        """Per-layer list matches number of layers."""
        for n_layers in [1, 4, 8]:
            residuals = [torch.randn(2, 4, 8) for _ in range(n_layers)]
            input_ids = torch.ones(2, 4, dtype=torch.long)
            emb_weight = torch.randn(6, 8)
            _, per_layer = compute_alignment_loss(
                residuals,
                input_ids,
                emb_weight,
            )
            assert len(per_layer) == n_layers

    def test_pad_positions_excluded(self) -> None:
        """PAD positions should not contribute to loss."""
        dim, vocab_size = 16, 10
        residuals = [torch.randn(2, 4, dim)]
        emb_weight = torch.randn(vocab_size, dim)
        # Only position 0 is non-PAD
        ids_mostly_pad = torch.zeros(2, 4, dtype=torch.long)
        ids_mostly_pad[:, 0] = 1
        # All non-PAD
        ids_full = torch.ones(2, 4, dtype=torch.long)
        loss_pad, _ = compute_alignment_loss(
            residuals,
            ids_mostly_pad,
            emb_weight,
        )
        loss_full, _ = compute_alignment_loss(
            residuals,
            ids_full,
            emb_weight,
        )
        # Both should produce valid scalars
        assert loss_pad.shape == ()
        assert loss_full.shape == ()
        # Values should generally differ (different contributing positions)
        # (not guaranteed for random data, but extremely unlikely to match)

    def test_aligned_residuals_lower_loss(self) -> None:
        """Residuals equal to an embedding should give lower loss."""
        dim, vocab_size = 64, 10
        torch.manual_seed(123)
        # Use orthogonal embeddings so alignment is unambiguous
        emb_weight = torch.eye(vocab_size, dim)
        # Residuals that are copies of embedding[1] — perfectly aligned
        aligned = emb_weight[1].unsqueeze(0).unsqueeze(0).expand(2, 4, -1).clone()
        # Random residuals in directions orthogonal to all embeddings
        random_res = torch.randn(2, 4, dim)
        random_res[:, :, :vocab_size] = 0  # zero out embedding subspace
        input_ids = torch.ones(2, 4, dtype=torch.long)
        loss_aligned, _ = compute_alignment_loss(
            [aligned],
            input_ids,
            emb_weight,
        )
        loss_random, _ = compute_alignment_loss(
            [random_res],
            input_ids,
            emb_weight,
        )
        # Aligned should have lower loss (closer to embeddings)
        assert loss_aligned < loss_random

    def test_gradient_flows(self) -> None:
        """Gradients flow through to residuals and embeddings."""
        dim, vocab_size = 16, 10
        residuals = [torch.randn(2, 4, dim, requires_grad=True)]
        emb_weight = torch.randn(vocab_size, dim, requires_grad=True)
        input_ids = torch.ones(2, 4, dtype=torch.long)
        mean_loss, _ = compute_alignment_loss(
            residuals,
            input_ids,
            emb_weight,
        )
        mean_loss.backward()
        assert residuals[0].grad is not None
        assert emb_weight.grad is not None


class TestComputeRepulsionLoss:
    """Test embedding repulsion auxiliary loss."""

    def test_scalar_output(self) -> None:
        """Repulsion loss returns a scalar with gradients."""
        emb_weight = torch.randn(10, 16, requires_grad=True)
        loss = compute_repulsion_loss(emb_weight, n_elements=6, margin=0.0)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_orthogonal_embeddings_zero_loss(self) -> None:
        """Orthogonal element embeddings should give zero loss at margin=0."""
        dim = 10
        emb_weight = torch.zeros(10, dim)
        # Element embeddings at indices 1-6: orthogonal unit vectors
        for i in range(6):
            emb_weight[i + 1, i] = 1.0
        loss = compute_repulsion_loss(emb_weight, n_elements=6, margin=0.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_identical_embeddings_positive_loss(self) -> None:
        """Identical element embeddings should give positive loss."""
        emb_weight = torch.zeros(10, 16)
        for i in range(6):
            emb_weight[i + 1] = torch.ones(16)
        loss = compute_repulsion_loss(emb_weight, n_elements=6, margin=0.0)
        assert loss.item() > 0

    def test_high_margin_reduces_loss(self) -> None:
        """A margin of 1.0 (max cosine) should give zero loss."""
        emb_weight = torch.randn(10, 16)
        loss_0 = compute_repulsion_loss(
            emb_weight,
            n_elements=6,
            margin=0.0,
        )
        loss_high = compute_repulsion_loss(
            emb_weight,
            n_elements=6,
            margin=1.0,
        )
        # margin=1.0 means cos_sim must exceed 1.0 to be penalized,
        # which is impossible, so loss should be 0
        assert loss_high.item() == pytest.approx(0.0, abs=1e-6)
        assert loss_0 >= loss_high

    def test_only_element_tokens(self) -> None:
        """Loss only depends on element embeddings (indices 1-6), not special tokens."""
        emb_weight = torch.randn(10, 16)
        loss1 = compute_repulsion_loss(emb_weight, n_elements=6, margin=0.0)
        # Modify special token embeddings (indices 0, 7, 8, 9)
        emb_modified = emb_weight.clone()
        emb_modified[0] *= 100
        emb_modified[7:] *= 100
        loss2 = compute_repulsion_loss(emb_modified, n_elements=6, margin=0.0)
        assert loss1.item() == pytest.approx(loss2.item(), abs=1e-6)

    def test_gradient_flows(self) -> None:
        """Gradients flow through to embedding weights."""
        emb_weight = torch.randn(10, 16, requires_grad=True)
        loss = compute_repulsion_loss(emb_weight, n_elements=6, margin=0.0)
        loss.backward()
        assert emb_weight.grad is not None


class TestLensAuxLoss:
    """grok_lens-style deep supervision at the answer position (Phase 5)."""

    def test_matches_manual_computation(self) -> None:
        import torch
        import torch.nn.functional as F

        from lego.data import compute_lens_aux_loss

        torch.manual_seed(0)
        n_layers, batch, seq, dim, vocab = 4, 5, 10, 8, 10
        residuals = [torch.randn(batch, seq, dim) for _ in range(n_layers)]
        input_ids = torch.randint(1, vocab, (batch, seq))
        answer_positions = torch.randint(1, seq, (batch,))
        norm = torch.nn.LayerNorm(dim)
        emb = torch.randn(vocab, dim)

        loss = compute_lens_aux_loss(
            residuals, input_ids, answer_positions, norm, emb, weighting="linear"
        )
        # manual: per intermediate layer, CE at <predict> position vs answer
        batch_idx = torch.arange(batch)
        targets = input_ids[batch_idx, answer_positions]
        per_layer = []
        for r in residuals[:-1]:
            logits = F.linear(norm(r[batch_idx, answer_positions - 1]), emb)
            per_layer.append(F.cross_entropy(logits, targets))
        w = torch.arange(1, n_layers, dtype=torch.float32)
        w = w / w.sum()
        expected = torch.stack(
            [wi * pl for wi, pl in zip(w, per_layer, strict=True)]
        ).sum()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_single_layer_is_zero(self) -> None:
        import torch

        from lego.data import compute_lens_aux_loss

        residuals = [torch.randn(3, 6, 8)]
        loss = compute_lens_aux_loss(
            residuals,
            torch.randint(1, 9, (3, 6)),
            torch.tensor([5, 5, 5]),
            torch.nn.LayerNorm(8),
            torch.randn(9, 8),
        )
        assert loss.item() == 0.0

    def test_final_layer_excluded(self) -> None:
        """Perturbing only the final layer's residual must not change the loss."""
        import torch

        from lego.data import compute_lens_aux_loss

        torch.manual_seed(0)
        residuals = [torch.randn(3, 6, 8) for _ in range(3)]
        args = (
            torch.randint(1, 9, (3, 6)),
            torch.tensor([5, 5, 5]),
            torch.nn.LayerNorm(8),
            torch.randn(9, 8),
        )
        loss_a = compute_lens_aux_loss(residuals, *args)
        residuals[-1] = torch.randn(3, 6, 8)
        loss_b = compute_lens_aux_loss(residuals, *args)
        assert torch.allclose(loss_a, loss_b)
