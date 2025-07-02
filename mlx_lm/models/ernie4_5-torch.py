from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

class Ernie4_5_RopeEmbedding(nn.Module):
    def __init__(self, head_dim, compression_ratio=1.0, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.base = base

    def forward(self, seq_length, position_ids=None):
        indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        indices = 1 / self.base ** (indices / self.head_dim)
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, 1, dtype=torch.float32
            ).unsqueeze(1)
            position_ids = position_ids / self.compression_ratio
            sinusoid_inp = position_ids * indices.unsqueeze(0)
        else:
            position_ids = position_ids / self.compression_ratio
            seq_length = position_ids.shape[-1]
            sinusoid_inp = position_ids.unsqueeze(-1).to(
                torch.float32
            ) * indices.unsqueeze(0)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(-1, 1, seq_length, self.head_dim)
        pos_emb = pos_emb.detach()
        return pos_emb

    def apply_rotary(self, rp, q, k):
        sin, cos = torch.chunk(rp.to(q.device), 2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape(rp.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape(rp.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = torch.stack(
            [-q[:, :, :, 1::2], q[:, :, :, 0::2]], dim=-1
        ).reshape(q.shape)
        query = (q.to(torch.float32) * cos_pos) + (
            rotate_half_q.to(torch.float32) * sin_pos
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = torch.stack(
            [-k[:, :, :, 1::2], k[:, :, :, 0::2]], dim=-1
        ).reshape(k.shape)
        key = (k.to(torch.float32) * cos_pos) + (
            rotate_half_k.to(torch.float32) * sin_pos
        )
        return query, key


class Ernie4_5_FusedDropoutImpl(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x, y):
        if self.prob > 0:
            x = self.dropout(x)
        output = x + y

        return output


class Ernie4_5_MLP(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.use_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.use_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.use_bias
        )

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Ernie4_5_Attention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        if config.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = config.head_dim

        self.is_gqa = (
            self.num_key_value_heads is not None
            and self.num_key_value_heads != self.num_heads
        )

        if self.is_gqa:
            kv_hidden_size = self.head_dim * self.num_key_value_heads
            q_hidden_size = self.head_dim * self.num_heads
        else:
            q_hidden_size = kv_hidden_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, q_hidden_size, bias=config.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=config.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=config.use_bias)
        self.o_proj = nn.Linear(q_hidden_size, self.hidden_size, bias=config.use_bias)

        self.rotary_emb = Ernie4_5_RopeEmbedding(
            self.head_dim,
            compression_ratio=config.compression_ratio,
            base=config.rope_theta,
        )
        self.config = config

    def repeat_kv(self, hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def rope_attn(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        past_key_value=None,
        use_cache=False,
        attn_mask_start_row_indices=None,
    ):
        query_states_dtype = query_states.dtype

        kv_seq_len = key_states.shape[-3]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-3]
            kv_seq_len += offset

        cos_sin = self.rotary_emb(kv_seq_len).permute(
            [0, 2, 1, 3]
        )  # [b,h,s,d]->[b,s,h,d]
        if offset > 0:
            cos_sin = cos_sin[:, offset:]
        query_states, key_states = self.rotary_emb.apply_rotary(
            cos_sin, query_states, key_states
        )

        query_states = query_states.to(query_states_dtype)
        key_states = key_states.to(query_states_dtype)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

        # shape: [2, b, s, kvh, d]
        past_key_value = [key_states, value_states] if use_cache else None
        seq_length = query_states.shape[1]
        attn_output, attn_weights = self.attn_func(
            query_states,
            key_states,
            value_states,
            attention_mask,
            attn_mask_start_row_indices,
            seq_length,
        )
        return attn_output


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config

        self.self_attn = Ernie4_5_Attention(config, layer_idx)
        self.mlp = Ernie4_5_MLP(config)

        self.input_layernorm = nn.RMSNorm(config)
        self.post_attention_layernorm = nn.RMSNorm(config)

        self.residual_add1 = Ernie4_5_FusedDropoutImpl(config.hidden_dropout_prob)
        self.residual_add2 = Ernie4_5_FusedDropoutImpl(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_start_row_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = self.residual_add1(self.self_attn(hidden_states=self.input_layernorm(hidden_states)), residual)
        hidden_states = self.residual_add2(self.mlp(self.post_attention_layernorm(hidden_states)), residual)
        return outputs

class Ernie4_5_Model(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList(
            [Ernie4_5_DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        self.norm = nn.RMSNorm(config)

    def forward(
        self,
        input_ids=None,
    ):
        for idx, (decoder_layer) in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                attn_mask_start_row_indices,
                position_ids,
                token_type_ids,
                output_attentions,
                past_key_value,
                use_cache,
            )
        return self.norm(layer_outputs)
    

class Ernie4_5_LMHead(nn.Module):
    def __init__(self, config):
        super(Ernie4_5_LMHead, self).__init__()
        self.config = config
        vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            # Weight of shape [vocab_size, hidden_size]
            self.weight = nn.Parameter(
                torch.empty(
                    vocab_size, config.hidden_size, dtype=torch.get_default_dtype()
                )
            )
        else:
            # Weight of shape [hidden_size, vocab_size]
            self.weight = nn.Parameter(
                torch.empty(
                    config.hidden_size, vocab_size, dtype=torch.get_default_dtype()
                )
            )
        nn.init.xavier_uniform_(self.weight)

        if config.weight_share_add_bias and config.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(vocab_size, dtype=torch.get_default_dtype())
            )
        else:
            self.bias = None

    def forward(self, hidden_states):
        return self.calc_lm_head_logits(
            self.config, hidden_states, self.weight, self.bias
        )

    def calc_lm_head_logits(self, config, hidden_states, weight, bias):
        if config.tie_word_embeddings:
            logits = torch.matmul(hidden_states, weight.T)
        else:
            logits = torch.matmul(hidden_states, weight)

        if bias is not None:
            logits = logits + bias

        return logits


class Ernie4_5_ForCausalLM(Ernie4_5_PretrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.model = Ernie4_5_Model(config)
        self.lm_head = Ernie4_5_LMHead(config)

    def forward(
        self,
        input_ids
    ):
        outputs = self.model(
            input_ids,
        )
        return self.lm_head(outputs)