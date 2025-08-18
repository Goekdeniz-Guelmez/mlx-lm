from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

class Qwen3MoeAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        return hidden_states


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
    def forward(self, x):
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states


class GroveMoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.num_experts_per_group = 2
        self.parallel_expert_intermediate_size = 128

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.register_buffer('expert_bias', torch.zeros(self.num_experts))

        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.chunk_experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=self.parallel_expert_intermediate_size) for _ in range(self.num_experts // self.num_experts_per_group)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        bias_routing_weights = torch.sigmoid(router_logits).to(torch.float)

        _, selected_experts = torch.topk(bias_routing_weights, self.top_k, dim=-1)
        group_selected_experts = selected_experts // self.num_experts_per_group

        routing_weights = routing_weights.gather(-1, selected_experts)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # forward large
        large_experts_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            large_experts_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # forward small
        small_experts_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(group_selected_experts, num_classes=self.num_experts // self.num_experts_per_group).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts // self.num_experts_per_group):
            expert_layer = self.chunk_experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            small_experts_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = 0.05 * small_experts_hidden_states + large_experts_hidden_states
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states


class GroveMoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        self.mlp = Qwen3MoeMLP(config)

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = GroveMoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        outputs = residual + hidden_states
        return outputs


class GroveMoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GroveMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                inputs_embeds
            )

        return self.norm(hidden_states)


class GroveMoeForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.model = GroveMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids
        )

        return self.lm_head(hidden_states)