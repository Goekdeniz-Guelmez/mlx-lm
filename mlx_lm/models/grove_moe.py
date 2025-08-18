# Copyright © 2025 Apple Inc.

from typing import Optional, Dict, List, Union, Any
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .qwen3_moe import Attention, MLP


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    max_window_layers: int
    decoder_sparse_step: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    norm_topk_prob: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False


class GroveMoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts_per_group = 2
        self.parallel_expert_intermediate_size = 128

        # gating
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros(self.num_experts)

        self.experts = [
            MLP(args.hidden_size, hidden_dim=args.moe_intermediate_size) 
            for _ in range(self.num_experts)
        ]
        self.chunk_experts = [
            MLP(args.hidden_size, hidden_dim=self.parallel_expert_intermediate_size) 
            for _ in range(self.num_experts // self.num_experts_per_group)
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = mx.softmax(router_logits, axis=-1)
        bias_routing_weights = mx.sigmoid(router_logits)

        _, selected_experts = mx.topk(bias_routing_weights, self.top_k, axis=-1)
        group_selected_experts = selected_experts // self.num_experts_per_group

        routing_weights = mx.take_along_axis(routing_weights, selected_experts, axis=-1)
        routing_weights = routing_weights / mx.sum(routing_weights, axis=-1, keepdims=True)

        # forward large
        large_experts_hidden_states = mx.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = mx.transpose(
            mx.one_hot(selected_experts, num_classes=self.num_experts), 
            axes=(2, 1, 0)
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            
            # Find indices where mask is True
            indices = mx.where(mask)
            if len(indices) == 2:
                idx, top_x = indices
            else:
                continue
                
            if len(top_x) == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state
            current_state = hidden_states[top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx:idx+1]

            # Add to the final hidden states
            large_experts_hidden_states = large_experts_hidden_states.at[top_x].add(current_hidden_states)

        # forward small
        small_experts_hidden_states = mx.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = mx.transpose(
            mx.one_hot(group_selected_experts, num_classes=self.num_experts // self.num_experts_per_group),
            axes=(2, 1, 0)
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts // self.num_experts_per_group):
            expert_layer = self.chunk_experts[expert_idx]
            mask = expert_mask[expert_idx]
            
            # Find indices where mask is True
            indices = mx.where(mask)
            if len(indices) == 2:
                idx, top_x = indices
            else:
                continue
                
            if len(top_x) == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state
            current_state = hidden_states[top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx:idx+1]

            # Add to the final hidden states
            small_experts_hidden_states = small_experts_hidden_states.at[top_x].add(current_hidden_states)

        final_hidden_states = 0.05 * small_experts_hidden_states + large_experts_hidden_states
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states
    

class GroveMoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, hidden_dim=args.moe_intermediate_size)

        self.self_attn = Attention(args, layer_idx)

        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = GroveMoeSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, intermediate_size=args.intermediate_size)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
    

class GroveMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            GroveMoeDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
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
            h = layer(h, mask, c)

        return self.norm(h)
    

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GroveMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.mlp.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers
