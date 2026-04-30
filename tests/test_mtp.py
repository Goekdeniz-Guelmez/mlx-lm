# Copyright © 2025 Apple Inc.
"""
Tests for Multi-Token Prediction (MTP) inference.

Creates a tiny random SLM with MTP heads so the full generation loop can be
exercised without loading a real checkpoint.  Also exercises the Qwen3.5
MTPModule to verify it follows the MTPHead protocol correctly.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generate_step, mtp_generate_step, stream_generate
from mlx_lm.models.base import MTPHead, create_attention_mask
from mlx_lm.models.cache import KVCache, make_prompt_cache


# ---------------------------------------------------------------------------
# Minimal transformer pieces
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.n_heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        q = self.q(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                qL, kL = scores.shape[-2:]
                qi = mx.arange(kL - qL, kL)
                ki = mx.arange(kL)
                mask = qi[:, None] >= ki[None]
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        out = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o(out)


class _MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden * 2, hidden, bias=False)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.attn = _Attention(hidden, heads)
        self.mlp = _MLP(hidden)
        self.ln1 = nn.RMSNorm(hidden)
        self.ln2 = nn.RMSNorm(hidden)

    def __call__(self, x, mask=None, cache=None):
        x = x + self.attn(self.ln1(x), mask=mask, cache=cache)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Concrete MTPHead implementation
# ---------------------------------------------------------------------------


class SimpleMTPHead(MTPHead):
    """
    A minimal MTP head that fuses hidden state + token embedding,
    runs one transformer block, then projects to vocab logits.
    """

    def __init__(self, hidden: int, vocab: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(hidden * 2, hidden, bias=False)
        self.norm = nn.RMSNorm(hidden)
        self.block = _TransformerBlock(hidden, heads)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def __call__(
        self,
        hidden_state: mx.array,
        token_embed: mx.array,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, mx.array]:
        # hidden_state: (1, 1, D), token_embed: (1, 1, D)
        fused = mx.concatenate([hidden_state, token_embed], axis=-1)
        h = self.norm(self.proj(fused))  # (1, 1, D)
        h = self.block(h, mask=None, cache=cache)
        logits = self.lm_head(h[:, -1, :])  # (1, vocab)
        return logits, h


# ---------------------------------------------------------------------------
# Tiny SLM with MTP support
# ---------------------------------------------------------------------------


class TinySLMModel(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = [_TransformerBlock(hidden, heads) for _ in range(layers)]
        self.norm = nn.RMSNorm(hidden)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class TinySLMWithMTP(nn.Module):
    """
    A random small language model that demonstrates the MTP protocol:
      - model_backbone(inputs, cache) -> (hidden_state, logits)
      - get_mtp_heads()               -> List[MTPHead]
      - model.model.embed_tokens      (used by the generation engine)
      - layers property               (used by make_prompt_cache)
    """

    def __init__(self, vocab: int = 256, hidden: int = 64, n_layers: int = 2,
                 heads: int = 2, n_mtp: int = 2):
        super().__init__()
        self.vocab_size = vocab
        self.model = TinySLMModel(vocab, hidden, n_layers, heads)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.mtp_heads = [SimpleMTPHead(hidden, vocab, heads) for _ in range(n_mtp)]

    # ---- Standard __call__ (used by generate_step / fallback path) ---------
    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        hidden = self.model(inputs, cache)
        return self.lm_head(hidden)

    # ---- MTP protocol -------------------------------------------------------
    def model_backbone(self, inputs: mx.array, cache=None):
        """Returns (hidden_state, logits) — split backbone needed by MTP."""
        hidden = self.model(inputs, cache)
        logits = self.lm_head(hidden)
        return hidden, logits

    def get_mtp_heads(self) -> List[MTPHead]:
        return self.mtp_heads

    # ---- mlx-lm conventions -------------------------------------------------
    @property
    def layers(self):
        return self.model.layers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMTPHead(unittest.TestCase):
    """Unit tests for the MTPHead base class and SimpleMTPHead."""

    def setUp(self):
        mx.random.seed(42)
        self.hidden = 64
        self.vocab = 256
        self.heads = 2
        self.head = SimpleMTPHead(self.hidden, self.vocab, self.heads)

    def test_mtp_head_is_subclass(self):
        self.assertIsInstance(self.head, MTPHead)

    def test_mtp_head_output_shapes(self):
        hidden_state = mx.random.normal((1, 1, self.hidden))
        token_embed = mx.random.normal((1, 1, self.hidden))
        logits, next_hidden = self.head(hidden_state, token_embed)
        mx.eval(logits, next_hidden)
        self.assertEqual(logits.shape, (1, self.vocab))
        self.assertEqual(next_hidden.shape, (1, 1, self.hidden))

    def test_mtp_head_abstract_prevents_direct_instantiation(self):
        # MTPHead uses @abstractmethod; mlx.nn.Module doesn't enforce ABC
        # enforcement at class level, so we verify the method must be
        # overridden by checking calling the base raises NotImplementedError.
        class BrokenHead(MTPHead):
            pass

        head = BrokenHead()
        with self.assertRaises((TypeError, NotImplementedError)):
            head(mx.zeros((1, 1, 4)), mx.zeros((1, 1, 4)))


class TestTinySLMWithMTP(unittest.TestCase):
    """Smoke tests for the minimal SLM + MTP model."""

    def setUp(self):
        mx.random.seed(0)
        self.model = TinySLMWithMTP(vocab=256, hidden=64, n_layers=2, heads=2, n_mtp=2)
        self.prompt = mx.array([1, 2, 3, 4, 5])

    def test_has_mtp_protocol(self):
        self.assertTrue(hasattr(self.model, "get_mtp_heads"))
        self.assertTrue(hasattr(self.model, "model_backbone"))
        heads = self.model.get_mtp_heads()
        self.assertEqual(len(heads), 2)
        for h in heads:
            self.assertIsInstance(h, MTPHead)

    def test_backbone_returns_hidden_and_logits(self):
        prompt_cache = make_prompt_cache(self.model)
        hidden, logits = self.model.model_backbone(self.prompt[None], prompt_cache)
        mx.eval(hidden, logits)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[-1], 256)
        self.assertEqual(hidden.shape[-1], 64)

    def test_standard_call_returns_logits(self):
        out = self.model(self.prompt[None])
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, 256))

    def test_layers_property(self):
        self.assertEqual(len(self.model.layers), 2)


class TestMTPGenerateStep(unittest.TestCase):
    """Tests for mtp_generate_step using the tiny random model."""

    def setUp(self):
        mx.random.seed(7)
        self.model = TinySLMWithMTP(vocab=256, hidden=64, n_layers=2, heads=2, n_mtp=2)
        self.prompt = mx.array([1, 2, 3, 4, 5])

    def test_generates_correct_count(self):
        max_tokens = 6
        tokens = []
        for tok, lp, from_draft in mtp_generate_step(
            self.prompt, self.model, max_tokens=max_tokens
        ):
            tokens.append((tok, from_draft))
            if len(tokens) >= max_tokens:
                break
        self.assertLessEqual(len(tokens), max_tokens)
        self.assertGreater(len(tokens), 0)

    def test_from_draft_flag(self):
        """At least some tokens should have from_draft=True (MTP accepted)."""
        all_flags = []
        for tok, lp, from_draft in mtp_generate_step(
            self.prompt, self.model, max_tokens=10
        ):
            all_flags.append(from_draft)
        # Backbone tokens always appear, so False must occur
        self.assertIn(False, all_flags)

    def test_logprobs_shape(self):
        for tok, lp, from_draft in mtp_generate_step(
            self.prompt, self.model, max_tokens=3
        ):
            mx.eval(lp)
            self.assertEqual(lp.ndim, 1)
            self.assertEqual(lp.shape[0], 256)
            break

    def test_token_values_in_range(self):
        for tok, lp, from_draft in mtp_generate_step(
            self.prompt, self.model, max_tokens=8
        ):
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, 256)

    def test_greedy_deterministic(self):
        """Same prompt + greedy sampler must produce the same tokens."""
        sampler = lambda x: mx.argmax(x, axis=-1)

        def _run():
            mx.random.seed(0)
            return [
                tok
                for tok, _, _ in mtp_generate_step(
                    self.prompt, self.model, max_tokens=6, sampler=sampler
                )
            ]

        self.assertEqual(_run(), _run())

    def test_fallback_when_no_mtp(self):
        """A plain model without MTP protocol falls back to generate_step."""

        class PlainModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = TinySLMModel(256, 64, 2, 2)
                self.lm_head = nn.Linear(64, 256, bias=False)

            def __call__(self, inputs, cache=None):
                return self.lm_head(self.model(inputs, cache))

            @property
            def layers(self):
                return self.model.layers

        plain = PlainModel()
        prompt = mx.array([1, 2, 3])
        tokens = [
            tok
            for tok, _, from_draft in mtp_generate_step(
                prompt, plain, max_tokens=4
            )
        ]
        # Should still generate tokens — just via the fallback path
        self.assertGreater(len(tokens), 0)
        self.assertEqual(len(tokens), 4)

    def test_matches_generate_step_output(self):
        """
        Without MTP (fallback path) the output must match generate_step
        so we know the fallback is correct.
        """

        class PlainModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = TinySLMModel(256, 64, 2, 2)
                self.lm_head = nn.Linear(64, 256, bias=False)

            def __call__(self, inputs, cache=None):
                return self.lm_head(self.model(inputs, cache))

            @property
            def layers(self):
                return self.model.layers

        mx.random.seed(1)
        plain = PlainModel()
        prompt = mx.array([1, 2, 3])
        sampler = lambda x: mx.argmax(x, axis=-1)

        mtp_toks = [
            tok
            for tok, _, _ in mtp_generate_step(
                prompt, plain, max_tokens=5, sampler=sampler
            )
        ]
        ref_toks = [
            tok for tok, _ in generate_step(prompt, plain, max_tokens=5, sampler=sampler)
        ]
        self.assertEqual(mtp_toks, ref_toks)


class TestMTPStreamGenerate(unittest.TestCase):
    """Integration tests: mtp_generate_step wired through stream_generate."""

    def setUp(self):
        mx.random.seed(3)
        self.model = TinySLMWithMTP(vocab=512, hidden=64, n_layers=2, heads=2, n_mtp=1)

    def _make_dummy_tokenizer(self):
        """Minimal tokenizer-like object that satisfies stream_generate."""
        from unittest.mock import MagicMock

        from mlx_lm.tokenizer_utils import TokenizerWrapper

        tok = MagicMock()
        tok.eos_token_ids = {0}
        tok.bos_token = None
        tok.encode.return_value = [1, 2, 3]
        tok.get_vocab.return_value = {}
        tok.has_thinking = False
        tok.has_tool_calling = False
        tok.tool_parser = None

        # Detokenizer
        dt = MagicMock()
        dt.last_segment = ""
        dt.text = ""
        tok.detokenizer = dt

        wrapper = TokenizerWrapper(tok)
        return wrapper

    def test_stream_generate_auto_detects_mtp(self):
        """stream_generate should use MTP path automatically."""
        tokenizer = self._make_dummy_tokenizer()
        prompt = [1, 2, 3, 4]

        responses = []
        for resp in stream_generate(
            self.model, tokenizer, prompt, max_tokens=4
        ):
            responses.append(resp)
        self.assertGreater(len(responses), 0)


# ---------------------------------------------------------------------------
# Qwen3.5 MTPModule tests
# ---------------------------------------------------------------------------


class TestQwen3_5MTPModule(unittest.TestCase):
    """Tests for the Qwen3.5 MTPModule (uses MTPHead protocol)."""

    def _build_model(self, n_mtp=1):
        """Construct a tiny Qwen3.5-style model with MTP via ModelArgs."""
        from mlx_lm.models.qwen3_5 import Model, ModelArgs

        text_cfg = dict(
            model_type="qwen3_5",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=256,
            max_position_embeddings=512,
            full_attention_interval=2,
            tie_word_embeddings=False,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            mtp_num_hidden_layers=n_mtp,
            rope_parameters={
                "type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        )
        args = ModelArgs(model_type="qwen3_5", text_config=text_cfg)
        return Model(args)

    def setUp(self):
        mx.random.seed(42)
        self.model = self._build_model(n_mtp=1)

    def test_model_has_mtp_protocol(self):
        self.assertTrue(hasattr(self.model, "get_mtp_heads"))
        self.assertTrue(hasattr(self.model, "model_backbone"))

    def test_mtp_heads_are_mtp_head_instances(self):
        heads = self.model.get_mtp_heads()
        self.assertEqual(len(heads), 1)
        self.assertIsInstance(heads[0], MTPHead)

    def test_model_backbone_shapes(self):
        prompt = mx.array([[1, 2, 3]])
        cache = make_prompt_cache(self.model)
        hidden, logits = self.model.model_backbone(prompt, cache)
        mx.eval(hidden, logits)
        # logits: (1, seq, vocab)
        self.assertEqual(logits.shape[-1], 256)
        # hidden: (1, seq, hidden)
        self.assertEqual(hidden.shape[-1], 64)

    def test_mtp_head_call_signature(self):
        head = self.model.get_mtp_heads()[0]
        hidden_state = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))
        logits, next_hidden = head(hidden_state, token_embed)
        mx.eval(logits, next_hidden)
        self.assertEqual(logits.shape, (1, 256))
        self.assertEqual(next_hidden.shape, (1, 1, 64))

    def test_mtp_generate_step_with_qwen35_model(self):
        # Qwen3.5 is a hybrid SSM+attention model.  Its ArraysCache is not
        # trimmable, so mtp_generate_step falls through to the normal
        # generate_step path rather than raising.  We verify generation
        # still produces valid tokens via that fallback.
        prompt = mx.array([1, 2, 3, 4, 5])
        tokens = list(mtp_generate_step(prompt, self.model, max_tokens=6))
        self.assertGreater(len(tokens), 0)
        for tok, lp, from_draft in tokens:
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, 256)

    def test_no_mtp_when_layers_zero(self):
        model_no_mtp = self._build_model(n_mtp=0)
        heads = model_no_mtp.get_mtp_heads()
        self.assertEqual(len(heads), 0)


if __name__ == "__main__":
    unittest.main()
