import math
from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn import functional as F

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


class BloomGelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)


class BloomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _reshape(self, fused_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        query_layer = fused_qkv[..., 0, :].transpose(1, 2)
        key_layer = fused_qkv[..., 1, :].transpose(1, 2)
        value_layer = fused_qkv[..., 2, :].transpose(1, 2)
        return query_layer, key_layer, value_layer

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, num_heads, seq_length, head_dim]
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

        # reshape qkv for further computations
        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        # [batch_size * num_heads, q_length, kv_length]
        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, -1)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, layer_past)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        return dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)


class BloomBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        return output  # hidden_states, past_kv, attentions


class BloomModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        # Cache, Attention Mask

        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states,
            )

        # Add last hidden state
        return self.ln_f(hidden_states)


class BloomForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = self.transformer(
            input_ids
        )

        return self.lm_head(hidden_states)
