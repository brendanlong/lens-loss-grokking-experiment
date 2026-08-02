"""Tests for LEGO composition task: enumeration, tokenizer, data, and losses."""

import pytest
import torch
import torch.nn.functional as F

from lego.data import (
    ChainDataset,
    collate_chains,
    encode_trajectory,
    make_eval_batch,
    make_k_uniform_sampler,
)
from lego.generator import (
    S3,
    ChainExample,
    compose,
    enumerate_chains,
    enumerate_split,
    group_by_k,
    make_example,
    subsample_k_uniform,
    train_test_split,
    verify_trajectory,
)
from lego.losses import (
    compute_answer_accuracy,
    compute_answer_only_loss,
    compute_lens_aux_loss,
    compute_lens_aux_loss_all_positions,
)
from lego.tokenizer import (
    ELEMENT_OFFSET,
    OP_TOKEN,
    PAD_ID,
    PREDICT_TOKEN,
    START_TOKEN,
    VOCAB_SIZE,
    answer_position,
    decode,
    element_token,
    encode,
    encode_padded,
    seq_len,
    token_to_str,
)

# ---- Cayley table tests (group axiom verification) ----


class TestCayleyTable:
    """Verify S₃ Cayley table satisfies group axioms."""

    def test_identity_left(self) -> None:
        """e · x = x for all x."""
        for x in range(S3.order):
            assert compose(0, x) == x

    def test_identity_right(self) -> None:
        """x · e = x for all x."""
        for x in range(S3.order):
            assert compose(x, 0) == x

    def test_closure(self) -> None:
        """All products are valid elements (0-5)."""
        for a in range(S3.order):
            for b in range(S3.order):
                result = compose(a, b)
                assert 0 <= result < S3.order

    def test_associativity(self) -> None:
        """(a · b) · c = a · (b · c) for all a, b, c."""
        for a in range(S3.order):
            for b in range(S3.order):
                for c in range(S3.order):
                    ab_c = compose(compose(a, b), c)
                    a_bc = compose(a, compose(b, c))
                    assert ab_c == a_bc, (
                        f"({S3.elements[a]}·{S3.elements[b]})·{S3.elements[c]} = "
                        f"{S3.elements[ab_c]} != "
                        f"{S3.elements[a]}·({S3.elements[b]}·{S3.elements[c]}) = "
                        f"{S3.elements[a_bc]}"
                    )

    def test_inverses(self) -> None:
        """Every element has a left and right inverse."""
        for a in range(S3.order):
            # Find left inverse: x · a = e
            left_inv = None
            for x in range(S3.order):
                if compose(x, a) == 0:
                    left_inv = x
                    break
            assert left_inv is not None, f"{S3.elements[a]} has no left inverse"
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
            assert compose(elem, elem) == 0, f"{S3.elements[elem]}² ≠ e"

    def test_group_order(self) -> None:
        """S₃ has exactly 6 elements."""
        assert S3.order == 6
        assert len(S3.elements) == 6
        assert len(S3.cayley) == 6
        for row in S3.cayley:
            assert len(row) == 6

    def test_latin_square(self) -> None:
        """Each row and column of Cayley table is a permutation of 0-5.

        This is a necessary property of any group's multiplication table.
        """
        elements = set(range(S3.order))
        for a in range(S3.order):
            row = {compose(a, b) for b in range(S3.order)}
            col = {compose(b, a) for b in range(S3.order)}
            assert row == elements, f"Row {S3.elements[a]} not a permutation"
            assert col == elements, f"Col {S3.elements[a]} not a permutation"


# ---- Enumeration tests ----


def expected_count(k_min: int, k_max: int, n: int = S3.order) -> int:
    """n starts x n^k op sequences for each k in [k_min, k_max]."""
    return sum(n * n**k for k in range(k_min, k_max + 1))


class TestEnumerateChains:
    def test_count_formula(self) -> None:
        """Count == sum over k of n · n^k for several small ranges."""
        assert len(enumerate_chains(0, 0)) == 6
        assert len(enumerate_chains(0, 1)) == 6 + 36
        assert len(enumerate_chains(0, 2)) == expected_count(0, 2)  # 258
        assert len(enumerate_chains(1, 3)) == expected_count(1, 3)  # 1548
        assert len(enumerate_chains(0, 3)) == expected_count(0, 3)  # 1554

    def test_full_s3_count(self) -> None:
        """The full k in [0, 6] enumeration has 335,922 chains."""
        assert expected_count(0, 6) == 335_922

    def test_all_unique(self) -> None:
        examples = enumerate_chains(0, 3)
        keys = {(ex.start, ex.ops) for ex in examples}
        assert len(keys) == len(examples)

    def test_all_trajectories_verify(self) -> None:
        for ex in enumerate_chains(0, 3):
            assert verify_trajectory(ex)

    def test_k_range(self) -> None:
        lengths = {len(ex.ops) for ex in enumerate_chains(1, 3)}
        assert lengths == {1, 2, 3}

    def test_deterministic(self) -> None:
        assert enumerate_chains(0, 3) == enumerate_chains(0, 3)

    def test_k_zero_identity(self) -> None:
        for ex in enumerate_chains(0, 0):
            assert len(ex.ops) == 0
            assert ex.trajectory == (ex.start,)

    def test_invalid_ranges_raise(self) -> None:
        with pytest.raises(ValueError, match="k_min must be >= 0"):
            enumerate_chains(-1, 3)
        with pytest.raises(ValueError, match="k_max"):
            enumerate_chains(3, 2)

    def test_final_states_exactly_uniform(self) -> None:
        """Over the full enumeration of length-3 chains, every final state
        appears exactly n^3 times (each op-product is a bijection on starts)."""
        counts = [0] * S3.order
        for ex in enumerate_chains(3, 3):
            counts[ex.trajectory[-1]] += 1
        assert counts == [S3.order**3] * S3.order


class TestMakeExample:
    def test_trajectory(self) -> None:
        # r · e = r, s · r = r2s
        ex = make_example(0, (1, 3))
        assert ex.trajectory == (0, 1, 5)
        assert verify_trajectory(ex)

    def test_k_zero(self) -> None:
        ex = make_example(4, ())
        assert ex.trajectory == (4,)


class TestVerifyTrajectory:
    def test_valid(self) -> None:
        # r · e = r, s · r = r2s
        ex = ChainExample(
            start=0,  # e
            ops=(1, 3),  # r, s
            trajectory=(0, 1, 5),  # e, r·e=r, s·r=r2s
        )
        assert verify_trajectory(ex)

    def test_invalid_trajectory(self) -> None:
        ex = ChainExample(
            start=0,
            ops=(1, 3),
            trajectory=(0, 1, 3),  # wrong: s·r=r2s not s
        )
        assert not verify_trajectory(ex)

    def test_wrong_length(self) -> None:
        ex = ChainExample(
            start=0,
            ops=(1,),
            trajectory=(0, 1, 2),  # too long
        )
        assert not verify_trajectory(ex)


# ---- Train/test split tests ----


class TestTrainTestSplit:
    def test_disjoint_and_covers_enumeration(self) -> None:
        examples = enumerate_chains(0, 3)
        train, test = train_test_split(examples, test_frac=0.2, seed=42)
        train_keys = {(ex.start, ex.ops) for ex in train}
        test_keys = {(ex.start, ex.ops) for ex in test}
        assert not train_keys & test_keys
        assert len(train) + len(test) == len(examples)
        assert train_keys | test_keys == {(ex.start, ex.ops) for ex in examples}

    def test_deterministic_given_seed(self) -> None:
        examples = enumerate_chains(0, 3)
        split_a = train_test_split(examples, test_frac=0.2, seed=42)
        split_b = train_test_split(examples, test_frac=0.2, seed=42)
        assert split_a == split_b

    def test_different_seeds_differ(self) -> None:
        examples = enumerate_chains(0, 3)
        _, test_a = train_test_split(examples, test_frac=0.2, seed=42)
        _, test_b = train_test_split(examples, test_frac=0.2, seed=43)
        assert {(ex.start, ex.ops) for ex in test_a} != {
            (ex.start, ex.ops) for ex in test_b
        }

    def test_every_k_represented_in_test(self) -> None:
        """Stratification: even the 6-example k=0 stratum gets a test chain."""
        _, test = train_test_split(enumerate_chains(0, 4), test_frac=0.2, seed=42)
        assert set(group_by_k(test)) == {0, 1, 2, 3, 4}

    def test_stratum_sizes(self) -> None:
        examples = enumerate_chains(0, 3)
        train, test = train_test_split(examples, test_frac=0.2, seed=42)
        test_by_k = group_by_k(test)
        train_by_k = group_by_k(train)
        for k in range(4):
            n_k = 6 * 6**k
            assert len(test_by_k[k]) == max(1, round(0.2 * n_k))
            assert len(train_by_k[k]) == n_k - len(test_by_k[k])

    def test_invalid_test_frac_raises(self) -> None:
        examples = enumerate_chains(0, 1)
        for frac in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="test_frac"):
                train_test_split(examples, test_frac=frac, seed=42)

    def test_enumerate_split_matches_manual(self) -> None:
        manual = train_test_split(enumerate_chains(0, 3), test_frac=0.2, seed=7)
        assert enumerate_split(0, 3, test_frac=0.2, seed=7) == manual


class TestGroupByK:
    def test_groups_and_preserves_order(self) -> None:
        examples = enumerate_chains(0, 2)
        by_k = group_by_k(examples)
        assert set(by_k) == {0, 1, 2}
        for k, exs in by_k.items():
            assert all(len(ex.ops) == k for ex in exs)
        assert sum(len(exs) for exs in by_k.values()) == len(examples)


# ---- Tokenizer tests ----


class TestElementToken:
    def test_range(self) -> None:
        for idx in range(S3.order):
            token = element_token(idx)
            assert ELEMENT_OFFSET <= token < ELEMENT_OFFSET + S3.order

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            element_token(-1)
        with pytest.raises(ValueError):
            element_token(6)


class TestTokenToStr:
    def test_elements(self) -> None:
        for idx, name in enumerate(S3.elements):
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
        ex = ChainExample(start=0, ops=(1,), trajectory=(0, 1))
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
        ex = make_example(1, (3, 2, 0))
        assert ex.trajectory == (1, 5, 4, 4)
        tokens = encode(ex)
        assert len(tokens) == seq_len(3)  # 2*3 + 4 = 10
        assert tokens[0] == START_TOKEN
        assert tokens[-2] == PREDICT_TOKEN
        assert tokens[-1] == element_token(4)  # answer: rs

    def test_decode_readable(self) -> None:
        ex = ChainExample(start=0, ops=(1, 3), trajectory=(0, 1, 5))
        tokens = encode(ex)
        assert decode(tokens) == "<start> e <op> r <op> s <predict> r2s"


class TestEncodePadded:
    def test_padding_length(self) -> None:
        ex = ChainExample(start=0, ops=(1,), trajectory=(0, 1))
        tokens = encode_padded(ex, k_max=6)
        assert len(tokens) == seq_len(6)  # 2*6 + 4 = 16

    def test_padding_tokens(self) -> None:
        ex = ChainExample(start=0, ops=(1,), trajectory=(0, 1))
        tokens = encode_padded(ex, k_max=3)
        # k=1: 6 real tokens, k_max=3: 10 total, so 4 padding
        assert tokens[6:] == [PAD_ID] * 4

    def test_no_padding_at_max(self) -> None:
        ex = ChainExample(start=0, ops=(1, 2, 3), trajectory=(0, 1, 0, 5))
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


class TestVocabSize:
    def test_no_overlap(self) -> None:
        """All token IDs are within vocab range and distinct."""
        all_ids = {PAD_ID, START_TOKEN, OP_TOKEN, PREDICT_TOKEN}
        for idx in range(S3.order):
            all_ids.add(element_token(idx))
        # 1 pad + 6 elements + 3 specials = 10
        assert len(all_ids) == VOCAB_SIZE
        assert max(all_ids) == VOCAB_SIZE - 1

    def test_vocab_size(self) -> None:
        assert S3.order + 4 == VOCAB_SIZE

    def test_encode_decode_roundtrip(self) -> None:
        ex = make_example(2, (1, 3, 0))
        tokens = encode(ex)
        assert len(tokens) == seq_len(3)
        decoded = decode(tokens)
        assert "<start>" in decoded
        assert "<predict>" in decoded


# ---- Data pipeline tests ----


class TestChainDataset:
    def test_length(self) -> None:
        examples = enumerate_chains(0, 2)
        ds = ChainDataset(examples, k_max=2)
        assert len(ds) == len(examples)

    def test_shapes(self) -> None:
        examples = enumerate_chains(1, 4)[:10]
        ds = ChainDataset(examples, k_max=4)
        item = ds[0]
        assert item["input_ids"].shape == (seq_len(4),)
        assert item["answer_position"].shape == ()
        assert item["chain_length"].shape == ()
        assert item["trajectory"].shape == (5,)  # k_max + 1

    def test_trajectory_encoding(self) -> None:
        ex = make_example(1, (3,))
        ds = ChainDataset([ex], k_max=3)
        expected = encode_trajectory(ex, k_max=3)
        assert ds[0]["trajectory"].tolist() == expected
        # trajectory (1, 5) as element tokens (offset 1), padded to k_max+1
        assert expected == [2, 6, PAD_ID, PAD_ID]


class TestCollate:
    def test_batch_shapes(self) -> None:
        examples = enumerate_chains(1, 4)[:10]
        ds = ChainDataset(examples, k_max=4)
        batch = collate_chains([ds[i] for i in range(5)])
        assert batch["input_ids"].shape == (5, seq_len(4))
        assert batch["answer_position"].shape == (5,)
        assert batch["chain_length"].shape == (5,)
        assert batch["trajectory"].shape == (5, 5)


class TestMakeEvalBatch:
    def test_shapes(self) -> None:
        examples = enumerate_chains(3, 3)[:10]
        batch = make_eval_batch(examples, k_max=6)
        assert batch["input_ids"].shape == (10, seq_len(6))
        assert batch["answer_position"].shape == (10,)
        assert batch["chain_length"].shape == (10,)
        assert (batch["chain_length"] == 3).all()
        assert (batch["answer_position"] == answer_position(3)).all()


# ---- Loss tests ----


class TestComputeLoss:
    def test_runs(self) -> None:
        """Loss computation doesn't crash."""
        batch_size = 4
        sl = seq_len(6)
        logits = torch.randn(batch_size, sl, VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, sl))
        answer_positions = torch.full((batch_size,), answer_position(6))
        loss = compute_answer_only_loss(logits, input_ids, answer_positions)
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
        loss = compute_answer_only_loss(logits, input_ids, answer_positions)
        assert loss.shape == ()
        assert loss.item() > 0


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


class TestLensAuxLoss:
    """grok_lens-style deep supervision at the answer position (Phase 5)."""

    def test_matches_manual_computation(self) -> None:
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


class TestLensAuxLossAllPositions:
    """Full-sequence lens deep supervision: next-token CE, pad-masked."""

    @staticmethod
    def _padded_input_ids(batch: int, seq: int, vocab: int) -> torch.Tensor:
        """Random non-pad ids with a pad tail of varying length per row."""
        input_ids = torch.randint(1, vocab, (batch, seq))
        for i in range(batch):
            pad_from = seq - 1 - i % 3  # rows with 1, 2, 3 trailing pads
            input_ids[i, pad_from:] = 0
        return input_ids

    def test_matches_manual_computation(self) -> None:
        torch.manual_seed(0)
        n_layers, batch, seq, dim, vocab = 4, 5, 10, 8, 10
        residuals = [torch.randn(batch, seq, dim) for _ in range(n_layers)]
        input_ids = self._padded_input_ids(batch, seq, vocab)
        norm = torch.nn.LayerNorm(dim)
        emb = torch.randn(vocab, dim)

        loss = compute_lens_aux_loss_all_positions(
            residuals, input_ids, norm, emb, weighting="linear"
        )
        # manual: per intermediate layer, CE of logits at t vs input_ids[t+1],
        # pad targets ignored
        targets = input_ids[:, 1:].reshape(-1)
        per_layer = []
        for r in residuals[:-1]:
            logits = F.linear(norm(r[:, :-1]), emb)
            per_layer.append(
                F.cross_entropy(
                    logits.reshape(-1, vocab),
                    targets,
                    ignore_index=0,
                )
            )
        w = torch.arange(1, n_layers, dtype=torch.float32)
        w = w / w.sum()
        expected = torch.stack(
            [wi * pl for wi, pl in zip(w, per_layer, strict=True)]
        ).sum()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_ignores_pad_targets(self) -> None:
        """Perturbing residuals at positions whose next-token target is pad
        (and at the last position, which has no target) must not change
        the loss."""
        torch.manual_seed(0)
        n_layers, batch, seq, dim, vocab = 3, 4, 8, 8, 10
        residuals = [torch.randn(batch, seq, dim) for _ in range(n_layers)]
        input_ids = self._padded_input_ids(batch, seq, vocab)
        norm = torch.nn.LayerNorm(dim)
        emb = torch.randn(vocab, dim)

        loss_a = compute_lens_aux_loss_all_positions(residuals, input_ids, norm, emb)
        pad_target = torch.zeros(batch, seq, dtype=torch.bool)
        pad_target[:, :-1] = input_ids[:, 1:] == 0
        pad_target[:, -1] = True  # last position predicts nothing
        for r in residuals:
            r[pad_target] = torch.randn_like(r[pad_target])
        loss_b = compute_lens_aux_loss_all_positions(residuals, input_ids, norm, emb)
        assert torch.allclose(loss_a, loss_b)

    def test_uniform_weighting_is_layer_mean(self) -> None:
        torch.manual_seed(0)
        n_layers, batch, seq, dim, vocab = 4, 3, 6, 8, 10
        residuals = [torch.randn(batch, seq, dim) for _ in range(n_layers)]
        input_ids = self._padded_input_ids(batch, seq, vocab)
        norm = torch.nn.LayerNorm(dim)
        emb = torch.randn(vocab, dim)

        loss = compute_lens_aux_loss_all_positions(residuals, input_ids, norm, emb)
        targets = input_ids[:, 1:].reshape(-1)
        per_layer = [
            F.cross_entropy(
                F.linear(norm(r[:, :-1]), emb).reshape(-1, vocab),
                targets,
                ignore_index=0,
            )
            for r in residuals[:-1]
        ]
        expected = torch.stack(per_layer).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_single_layer_is_zero(self) -> None:
        residuals = [torch.randn(3, 6, 8)]
        loss = compute_lens_aux_loss_all_positions(
            residuals,
            torch.randint(1, 9, (3, 6)),
            torch.nn.LayerNorm(8),
            torch.randn(9, 8),
        )
        assert loss.item() == 0.0

    def test_final_layer_excluded(self) -> None:
        """Perturbing only the final layer's residual must not change the loss."""
        torch.manual_seed(0)
        residuals = [torch.randn(3, 6, 8) for _ in range(3)]
        input_ids = torch.randint(1, 9, (3, 6))
        norm = torch.nn.LayerNorm(8)
        emb = torch.randn(9, 8)
        loss_a = compute_lens_aux_loss_all_positions(residuals, input_ids, norm, emb)
        residuals[-1] = torch.randn(3, 6, 8)
        loss_b = compute_lens_aux_loss_all_positions(residuals, input_ids, norm, emb)
        assert torch.allclose(loss_a, loss_b)


class TestKUniformSampler:
    def test_k_distribution_uniform(self) -> None:
        """Sampled k values are ~uniform even though the enumeration is
        dominated by the largest k."""
        train, _test = enumerate_split(0, 4, test_frac=0.2, seed=42)
        dataset = ChainDataset(train, k_max=4)
        sampler = make_k_uniform_sampler(dataset, seed=0)
        ks = dataset.chain_lengths[torch.tensor(list(sampler))]
        counts = torch.bincount(ks, minlength=5).float()
        frac = counts / counts.sum()
        # each of the 5 strata should get ~1/5 of draws
        assert torch.all((frac > 0.15) & (frac < 0.25)), frac

    def test_deterministic_given_seed(self) -> None:
        train, _test = enumerate_split(0, 3, test_frac=0.2, seed=42)
        dataset = ChainDataset(train, k_max=3)
        a = list(make_k_uniform_sampler(dataset, seed=7))
        b = list(make_k_uniform_sampler(dataset, seed=7))
        c = list(make_k_uniform_sampler(dataset, seed=8))
        assert a == b
        assert a != c

    def test_only_dataset_indices(self) -> None:
        """Sampler indices stay within the train dataset (held-out examples
        can never be drawn)."""
        train, _test = enumerate_split(0, 3, test_frac=0.2, seed=42)
        dataset = ChainDataset(train, k_max=3)
        idx = list(make_k_uniform_sampler(dataset, seed=0))
        assert len(idx) == len(dataset)
        assert min(idx) >= 0 and max(idx) < len(dataset)


class TestSubsampleKUniform:
    def test_size_and_membership(self) -> None:
        train, _test = enumerate_split(0, 4, test_frac=0.2, seed=42)
        subset = subsample_k_uniform(train, 500, seed=42)
        assert len(subset) == 500
        train_set = set(train)
        assert all(ex in train_set for ex in subset)
        assert len(set(subset)) == 500  # no duplicates

    def test_small_strata_fully_included_and_waterfill(self) -> None:
        """k<=1 strata are smaller than an even share; they are taken whole
        and the shortfall goes to the large strata."""
        train, _test = enumerate_split(0, 4, test_frac=0.2, seed=42)
        by_k_train = {k: len(v) for k, v in group_by_k(train).items()}
        subset = subsample_k_uniform(train, 500, seed=42)
        by_k = {k: len(v) for k, v in group_by_k(subset).items()}
        # small strata (fewer than an even 100-per-k share) come in whole
        assert by_k[0] == by_k_train[0]
        assert by_k[1] == by_k_train[1]
        # large strata absorb the remainder about evenly
        assert sum(by_k.values()) == 500
        large = [by_k[k] for k in by_k if by_k[k] < by_k_train[k]]
        assert max(large) - min(large) <= 1

    def test_deterministic_given_seed(self) -> None:
        train, _test = enumerate_split(0, 3, test_frac=0.2, seed=42)
        a = subsample_k_uniform(train, 200, seed=1)
        b = subsample_k_uniform(train, 200, seed=1)
        c = subsample_k_uniform(train, 200, seed=2)
        assert a == b
        assert a != c

    def test_n_too_large_raises(self) -> None:
        train, _test = enumerate_split(0, 1, test_frac=0.2, seed=42)
        with pytest.raises(ValueError, match="cannot subsample"):
            subsample_k_uniform(train, len(train) + 1, seed=0)

    def test_disjoint_from_test_split(self) -> None:
        """The subset is drawn from the train split only, so it can never
        contain held-out chains."""
        train, test = enumerate_split(0, 3, test_frac=0.2, seed=42)
        subset = subsample_k_uniform(train, 300, seed=42)
        test_set = set(test)
        assert not any(ex in test_set for ex in subset)


class TestFitProbes:
    def test_recovers_linear_signal_and_shapes(self) -> None:
        """Probes decode a class that is linearly present in the features and
        stay at chance for one that is absent."""
        from lego.analyze_probes import fit_probes, probe_accuracy

        gen = torch.Generator().manual_seed(0)
        n, dim = 600, 16
        classes = torch.randint(0, 6, (n,), generator=gen)
        noise_classes = torch.randint(0, 6, (n,), generator=gen)
        directions = torch.randn(6, dim, generator=gen)
        feats = directions[classes] + 0.1 * torch.randn(n, dim, generator=gen)
        # layer 0 carries the signal; layer 1 is pure noise
        layer_feats = torch.stack([feats, torch.randn(n, dim, generator=gen)])
        targets = torch.stack([classes, noise_classes])

        weights = fit_probes(layer_feats, targets, steps=300, lr=5e-2)
        assert weights.shape == (2, 2, 6, dim)
        acc = probe_accuracy(layer_feats, targets, weights)
        assert acc.shape == (2, 2)
        assert acc[0, 0] > 0.95  # signal layer, signal target
        assert acc[1, 1] < 0.5  # noise layer can only overfit noise targets

    def test_deterministic(self) -> None:
        from lego.analyze_probes import fit_probes

        gen = torch.Generator().manual_seed(1)
        feats = torch.randn(2, 50, 8, generator=gen)
        targets = torch.randint(0, 6, (3, 50), generator=gen)
        w1 = fit_probes(feats, targets, steps=50)
        w2 = fit_probes(feats, targets, steps=50)
        assert torch.equal(w1, w2)


class TestTotalStepsBudget:
    def test_loop_stops_at_total_steps(self) -> None:
        """train_lego_model stops mid-epoch when total_steps is reached."""
        from torch.utils.data import DataLoader

        from lego.config import lego_model_config
        from lego.data import collate_chains
        from lego.model import StandardTransformer
        from lego.training import train_lego_model

        train, test = enumerate_split(0, 2, test_frac=0.2, seed=42)
        dataset = ChainDataset(train, k_max=2)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_chains)
        config = lego_model_config(dim=16, n_heads=2, n_layers=1)
        model = StandardTransformer(config)
        result = train_lego_model(
            model,
            loader,
            group_by_k(test),
            config,
            torch.device("cpu"),
            total_steps=3,
            k_min=0,
            k_max=2,
            n_epochs=5,
            use_compile=False,
            early_stop_patience=None,
            log_every_steps=1000,
            eval_every_steps=1000,
            checkpoint_dir=None,
            save_every_steps=None,
            use_wandb=False,
        )
        assert result.total_steps == 3


class TestFullSequenceLoss:
    def test_shift_and_mask(self) -> None:
        """Loss scores next tokens only at non-pad targets, with the
        standard autoregressive shift."""
        from lego.losses import compute_full_sequence_loss

        ex = make_example(2, (1, 3))
        ids = torch.tensor([encode_padded(ex, k_max=3)])  # padded to len 10
        n_real = len(encode(ex))  # 8 tokens; targets = positions 1..7
        vocab = VOCAB_SIZE
        # Perfect logits on the real next tokens -> loss ~ 0
        logits = torch.full((1, ids.shape[1], vocab), -10.0)
        for t in range(ids.shape[1] - 1):
            if ids[0, t + 1] != PAD_ID:
                logits[0, t, ids[0, t + 1]] = 10.0
        loss = compute_full_sequence_loss(logits, ids)
        assert loss.item() < 1e-4
        # Corrupting a PAD-target position must not change the loss
        logits2 = logits.clone()
        logits2[0, n_real, :] = torch.randn(vocab)
        assert torch.allclose(loss, compute_full_sequence_loss(logits2, ids))
        # Corrupting the answer target (predict position) must change it
        logits3 = logits.clone()
        logits3[0, n_real - 2, ids[0, n_real - 1]] = -10.0
        assert compute_full_sequence_loss(logits3, ids) > loss + 0.1

    def test_answer_position_included(self) -> None:
        """The answer token is a next-token target (at <predict>)."""
        ex = make_example(0, (1,))
        ids = torch.tensor([encode(ex)])
        pos = answer_position(1)
        assert ids[0, pos] == element_token(ex.trajectory[-1])
        # <predict> is at pos-1, so logits there score the answer
        assert ids[0, pos - 1] == PREDICT_TOKEN
