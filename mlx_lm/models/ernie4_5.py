# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention

# python -m mlx_lm.generate --model baidu/ERNIE-4.5-0.3B-PT --prompt "The capital of France is" -m 20


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int = 1024
    intermediate_size: int = 3072
    max_position_embeddings: int = 131072
    model_type: str = "ernie4_5"
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    num_hidden_layers: int = 18
    rms_norm_eps: float = 1e-05
    vocab_size: int = 103424
    rope_theta: float = 500000
    use_bias: bool = False
    tie_word_embeddings: bool = True
    compression_ratio: float = 1.0
    hidden_dropout_prob: float = 0.0


class Ernie4_5_MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.use_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.use_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.use_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
    

class Ernie4_5_Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = getattr(
            args, "head_dim", args.hidden_size // args.num_attention_heads
        )
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.is_gqa = (
            self.n_kv_heads is not None
            and self.n_kv_heads != self.n_heads
        )

        if self.is_gqa:
            kv_hidden_size = self.head_dim * self.n_kv_heads
            q_hidden_size = self.head_dim * self.n_heads
        else:
            q_hidden_size = kv_hidden_size = self.head_dim * self.n_heads

        self.q_proj = nn.Linear(args.hidden_size, q_hidden_size, bias=args.use_bias)
        self.k_proj = nn.Linear(args.hidden_size, kv_hidden_size, bias=args.use_bias)
        self.v_proj = nn.Linear(args.hidden_size, kv_hidden_size, bias=args.use_bias)
        self.o_proj = nn.Linear(q_hidden_size, args.hidden_size, bias=args.use_bias)

        # Use custom RoPE with compression ratio
        self.rope = nn.RoPE(
            dims=self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )
        self.compression_ratio = args.compression_ratio

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            effective_offset = int(cache.offset / self.compression_ratio)
            queries = self.rope(queries, offset=effective_offset)
            keys = self.rope(keys, offset=effective_offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Ernie4_5_FusedDropoutImpl(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        self.dropout = nn.Dropout(p=prob)

    def __call__(self, x, y):
        if self.prob > 0:
            x = self.dropout(x)
        output = x + y
        return output


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Ernie4_5_Attention(args)
        self.mlp = Ernie4_5_MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.residual_add1 = Ernie4_5_FusedDropoutImpl(args.hidden_dropout_prob)
        self.residual_add2 = Ernie4_5_FusedDropoutImpl(args.hidden_dropout_prob)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        x = self.residual_add1(x, self.self_attn(self.input_layernorm(x), mask=mask, cache=cache))
        x = self.residual_add2(x, self.mlp(self.post_attention_layernorm(x)))
        return x

class Ernie4_5_Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Ernie4_5_DecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Ernie4_5_Model(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers