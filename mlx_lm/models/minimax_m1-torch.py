import math
import warnings
from typing import List, Optional, Tuple, Union
import copy
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from einops import rearrange
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

BLOCK = 256

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniMaxM1Config"


class GLU(nn.Module):
    def __init__(self, d1, d2, bias=False):
        super().__init__()
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)

    def forward(self, x):
        return self.l3(self.l1(x) * self.l2(x))


class MiniMaxM1LightningAttention(nn.Module):
    def __init__(self, config: MiniMaxM1Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)

        self.out_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(self.head_dim * self.num_heads)

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.head_dim * self.num_heads, bias=False)
        self.output_gate = nn.Linear(self.hidden_size, self.head_dim * self.num_heads, bias=False)

        # for inference only
        self.offset = 0

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,  # (b, n)
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        # x: b n d
        b, n, d = x.shape
        # linear map
        qkv = self.act(self.qkv_proj(x))
        new_shape = qkv.size()[:-1] + (self.num_heads, -1)
        qkv = qkv.view(*new_shape)
        q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=3)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1

        # for align with metaseq
        ratio = torch.exp(-slope_rate)

        # only use for the first time
        if past_key_value is None:
            slope_rate = slope_rate.to(torch.float32)
            if attn_mask is not None:
                v = v.masked_fill((1 - attn_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool), 0)
            NUM_BLOCK = (n + BLOCK - 1) // BLOCK
            b, h, n, d = q.shape
            e = v.shape[-1]
            # other
            array = torch.arange(BLOCK).to(q) + 1
            q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
            k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
            index = array[:, None] - array[None, :]
            s_index = slope_rate * index[
                None,
                None,
            ]
            s_index = torch.where(index >= 0, -s_index, float("-inf"))
            diag_decay = torch.exp(s_index)

            kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
            output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si
                qi = q[:, :, si:ei].contiguous()
                ki = k[:, :, si:ei].contiguous()
                vi = v[:, :, si:ei].contiguous()
                qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)

                # diag
                qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m]
                qkv_diag = torch.matmul(qk, vi.to(torch.float32))
                block_decay = torch.exp(-slope_rate * m)
                output[:, :, si:ei] = qkv_none_diag + qkv_diag
                kv = block_decay * kv + torch.matmul((ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)

        else:
            kv = past_key_value
            output = []
            for i in range(n):
                kv = ratio * kv + torch.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i:i + 1],
                    v[:, :, i:i + 1],
                )
                qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :, i:i + 1], kv.to(q.dtype))
                output.append(qkv)
            output = torch.concat(output, dim=-2)
        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # normalize
        output = self.norm(output)
        # gate
        output = F.sigmoid(self.output_gate(x)) * output
        # outproj
        output = self.out_proj(output)

        attn_weights = None

        return output, attn_weights, kv


# Copied from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->MiniMaxM1
class MiniMaxM1Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MiniMaxM1Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_dim = getattr(config, 'rotary_dim', self.head_dim)

        self.rotary_emb = MiniMaxM1RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Union[Cache, Tuple[torch.Tensor]]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
                _flash_supports_window_size
                and getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reshape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            # reuse k, v, for evaluation only
            key_states = torch.cat([past_key_value[0], key_states], dim=-3)
            value_states = torch.cat([past_key_value[1], value_states], dim=-3)
        
        past_key_value = (key_states, value_states) if use_cache else None

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniMaxM1MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MiniMaxM1BlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MiniMaxM1Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MiniMaxM1SparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MiniMaxM1BlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
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
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MiniMaxM1DecoderLayer(nn.Module):
    def __init__(self, config: MiniMaxM1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = self.build_attn(config, layer_idx)

        self.layer_idx = layer_idx

        self.block_sparse_moe = MiniMaxM1SparseMoeBlock(config)
        self.input_layernorm = MiniMaxM1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.postnorm = getattr(config, 'postnorm', False)
        self.layernorm_attention_alpha = getattr(config, 'layernorm_linear_attention_alpha', 1) \
            if config.attention_type == 0 else getattr(config, 'layernorm_full_attention_alpha', 1)
        self.layernorm_attention_beta = getattr(config, 'layernorm_linear_attention_beta', 1) \
            if config.attention_type == 0 else getattr(config, 'layernorm_full_attention_beta', 1)
        self.layernorm_mlp_alpha = getattr(config, 'layernorm_mlp_alpha', 1)
        self.layernorm_mlp_beta = getattr(config, 'layernorm_mlp_beta', 1)

        shared_intermediate = getattr(config, 'shared_intermediate_size', 0)
        self.shared_moe = False
        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = MiniMaxM1MLP(config)
            self.coefficient = torch.nn.Linear(self.hidden_size, 1, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            slope_rate: Optional[float] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            slope_rate=slope_rate,
        )

        hidden_states = residual * self.layernorm_attention_alpha \
                        + hidden_states * self.layernorm_attention_beta

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        if self.shared_moe:
            output_mlp = self.shared_mlp(hidden_states)
            weight_fp32 = self.coefficient.weight.float()
            coef = hidden_states.to(torch.float32) @ weight_fp32.T
            coef = torch.nn.functional.sigmoid(coef).to(hidden_states.dtype)
            hidden_states = moe_hidden_states * (1 - coef) + output_mlp * coef
        else:
            hidden_states = moe_hidden_states

        hidden_states = residual * self.layernorm_mlp_alpha + hidden_states * self.layernorm_mlp_beta
        return hidden_states


class MiniMaxM1Model():
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.attn_type_list = config.attn_type_list

        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            if self.attn_type_list[i] == 0:
                _config._attn_implementation = 'linear_attention'
                _config.attention_type = 0
            else:
                _config._attn_implementation = config_copy._attn_implementation
                _config.attention_type = 1
            self.layers.append(MiniMaxM1DecoderLayer(_config, i))

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.slopes = self._build_slope_tensor(config.num_attention_heads)
        
        # mask
        self._linear_attn_mask = torch.empty(0)

    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n)  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2)
                        + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        # h, 1, 1
        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float32).reshape(n_attention_heads, 1, 1)

        return slopes

    def forward(
            self,
            input_ids: torch.LongTensor = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        inputs_embeds = self.embed_tokens(input_ids)
        slope_rates = [self.slopes.to("cpu") for _ in range(len(self.layers))]
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            slope_rate = slope_rates[idx]
            slope_rate = slope_rate * (1 - idx / (len(self.layers) - 1) + 1e-5)
            hidden_states = decoder_layer(
                hidden_states,
                slope_rate=slope_rate
            )

        return self.norm(hidden_states)


class MiniMaxM1ForCausalLM():
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxM1Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
        )
        return self.lm_head(hidden_states)