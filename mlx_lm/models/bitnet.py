# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention

@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    model_type: str
    rms_norm_eps: float
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rope_theta: float
    tie_word_embeddings: bool
    vocab_size: int
    quantization_config: Dict[str, Any]


class BitNetLinear(nn.Module):
    """Custom linear layer that supports weight scaling for BitNet quantization"""
    def __init__(self, input_dims: int, output_dims: int, bias: bool = False):
        super().__init__()
        self.weight = mx.random.normal((output_dims, input_dims))
        self.weight_scale = mx.ones((output_dims,))  # Add weight scale parameter
        if bias:
            self.bias = mx.zeros((output_dims,))
        else:
            self.bias = None
    
    def __call__(self, x: mx.array) -> mx.array:
        # Apply scaling to weights during forward pass
        scaled_weight = self.weight * self.weight_scale[:, None]
        output = x @ scaled_weight.T
        if self.bias is not None:
            output = output + self.bias
        return output


class BitNetMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = BitNetLinear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = BitNetLinear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = BitNetLinear(args.hidden_size, args.intermediate_size, bias=False)
        
        self.ffn_sub_norm = nn.RMSNorm(args.intermediate_size, eps=args.rms_norm_eps)

    def forward(self, x: mx.array)  -> mx.array:
        return self.down_proj(self.ffn_sub_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
    

class BitNetAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = BitNetLinear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = BitNetLinear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = BitNetLinear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = BitNetLinear(self.n_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_sub_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            dims=self.head_dim,
            base=args.rope_theta,
            scale=self.scale
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.attn_sub_norm(output)
        return self.o_proj(output)
    

class BitNetDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.self_attn = BitNetAttention(args)

        self.mlp = BitNetMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
    

class BitNetModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BitNetDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ) -> mx.array:
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
        self.model = BitNetModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = BitNetLinear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ) -> mx.array:
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers
