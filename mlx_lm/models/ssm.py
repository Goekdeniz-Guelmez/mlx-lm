from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_dt(dt, dt_bias, time_step_limit):
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def make_ssm_kernel():
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        // C and B have shape [batch, group, state_dim]
        // C and B need to be offset by group size
        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<T>(state);
            acc += state * C_[s_idx];
        }}
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {{
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }}
    """
    return mx.fast.metal_kernel(
        name="ssm_kernel",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


def make_mamba1_kernel():
    if not mx.metal.is_available():
        return None
    source = R"""
        // Per-step Mamba v1 SSM forward (seq_len == 1).
        // Template symbols:
        //   D  : channel dim (intermediate_size)
        //   Ds : state dim
        //   T  : threads per channel (= 32)
        // Grid: (T, D, B)
        auto b = thread_position_in_grid.z;
        auto d_idx = thread_position_in_grid.y;    // channel in [0, D)
        auto lane = thread_position_in_threadgroup.x; // 0..T-1

        constexpr int T = 32;
        constexpr int n_per_lane = (Ds + T - 1) / T; // ceil

        // Pointers are laid out as:
        //   X:         (B, 1, D)          -> X[b, 0, d]
        //   A_log:     (D, Ds)            -> A_log[d, s]
        //   B, C:      (B, D, Ds)         -> B[b, d, s], C[b, d, s]
        //   delta:     (B, D)             -> delta[b, d]
        //   state_in:  (B, D, Ds)         -> state_in[b, d, s]

        // Base offsets
        auto X_bd = X + (b * D + d_idx);           // scalar at seq_len==1
        auto A_d  = A_log + (d_idx * Ds);          // row pointer
        auto B_bd = B + ((b * D + d_idx) * Ds);
        auto C_bd = C + ((b * D + d_idx) * Ds);
        auto S_bd = state_in + ((b * D + d_idx) * Ds);
        auto O_bd = state_out + ((b * D + d_idx) * Ds);
        auto Y_bd = out + (b * D + d_idx);         // scalar output

        float x = static_cast<float>(*X_bd);
        float dlt = static_cast<float>(delta[b * D + d_idx]);
        float acc = 0.0f;

        for (int i = 0; i < n_per_lane; ++i) {
            int s = lane + i * T;
            if (s >= Ds) break;
            float A = -fast::exp(static_cast<float>(A_d[s]));
            float dA = fast::exp(dlt * A);
            float new_state = dA * static_cast<float>(S_bd[s])
                            + (dlt * x) * static_cast<float>(B_bd[s]);
            O_bd[s] = static_cast<TYPE>(new_state);
            acc += new_state * static_cast<float>(C_bd[s]);
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            // y = acc + D[d] * x
            float Dd = static_cast<float>(Dvec[d_idx]);
            *Y_bd = static_cast<TYPE>(acc + Dd * x);
        }
    """
    return mx.fast.metal_kernel(
        name="mamba1_ssm_step_kernel",
        input_names=["X", "A_log", "B", "C", "Dvec", "delta", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


_ssm_kernel = make_ssm_kernel()
_mamba1_kernel = make_mamba1_kernel()


def ssm_update_kernel(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: Tuple[float, float],
):
    n, _, h, d = hidden_states.shape
    input_type = hidden_states.dtype
    hb, ds = B.shape[-2:]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    return _ssm_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[("T", input_type), ("Dh", d), ("Ds", ds), ("H", h), ("G", h // hb)],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, 1, h, d), state.shape],
        output_dtypes=[input_type, input_type],
    )


def segsum(x):
    l = x.shape[-1]
    x = mx.repeat(x[..., None], l, axis=-1)
    x = mx.tril(x, -1)
    x_segsum = mx.cumsum(x, axis=-2)
    return x_segsum


def ssm_attn(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
) -> Tuple[mx.array, mx.array]:
    """SSD-SSM forward pass.

    Args:
        x: Input of shape (batch_size, seq_len, num_heads, head_dim).
        dt: Time deltas of shape (seq_len, num_heads,).
        A_log: State transition of shape (num_heads,).
        B: Input mixing of shape (batch_size, seq_len, num_groups, n).
        C: Output mixing of shape (batch_size, seq_len, num_groups, n).
        D: Residual connection.
        dt_bias: Bias for time deltas of shape (num_heads,).
        time_step_limit: Minimum and maximum value for time deltas.

    Code modified from
    https://github.com/cartesia-ai/edge/blob/main/cartesia-mlx/cartesia_mlx/layers/ssd/ops.py

    """
    b, l, h, dh = x.shape
    _, _, g, d = B.shape

    dt = compute_dt(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -mx.exp(A_log)
    B = mx.transpose(B, (0, 2, 3, 1))

    # A * s + B * C
    CB = mx.swapaxes(C, 1, 2) @ B
    CB = mx.repeat(CB, repeats, axis=1)

    dtA = dt * A.reshape(1, 1, -1)

    decay = mx.exp(segsum(dtA.swapaxes(1, 2)))

    surrogate_attention_matrix = mx.tril(CB * decay, 0)

    dtx = dt.reshape(b, l, h, 1) * x
    y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
    y = mx.swapaxes(y, 1, 2)

    decay = decay[:, :, -1:, :].transpose(0, 3, 1, 2)
    B = mx.repeat(B, h // g, axis=1).swapaxes(2, 3)
    dtxdecay = dtx * decay
    dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)

    next_state = dtxdecay @ B

    if state is not None:
        exp_dtA_cumsum = mx.exp(mx.cumsum(dtA, axis=-2))
        next_state += exp_dtA_cumsum[:, -1, :, None, None] * state
        state = state.reshape((b, 1, g, repeats, dh, d))
        C = C.reshape(b, l, g, 1, d, 1)
        y_prev = (state @ C).squeeze(-1).flatten(2, 3)
        y += exp_dtA_cumsum[..., None] * y_prev

    y += x * D.reshape(1, 1, h, 1)
    return y, next_state


def ssm_update(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
):
    seq_len = hidden_states.shape[1]
    if (
        seq_len > 1
        or state is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        fn = ssm_attn
    else:
        fn = ssm_update_kernel
    return fn(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state,
        time_step_limit,
    )


def ssm_v1_step(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    delta: mx.array,
    state: Optional[mx.array] = None,
):
    b, _, d = x.shape
    ds = A_log.shape[-1]

    x_ = x.reshape(b, d)
    A = -mx.exp(A_log)
    dA = mx.exp(delta[..., None] * A)

    inp = (delta * x_)[..., None]
    if state is None:
        prev = mx.zeros((b, d, ds), dtype=x.dtype)
    else:
        prev = state
    new_state = prev * dA + inp * B

    y = (new_state * C).sum(axis=-1) + D.reshape(1, d) * x_
    y = y.reshape(b, 1, d)
    return y, new_state


def ssm_v1_update(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    delta: mx.array,
    state: Optional[mx.array] = None,
):
    b, l, d = hidden_states.shape
    assert l == 1, "ssm_v1_update expects seq_len == 1"

    if (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and _mamba1_kernel is not None
    ):
        input_type = hidden_states.dtype
        ds = A_log.shape[-1]
        if state is None:
            state = mx.zeros((b, d, ds), dtype=input_type)
        out, new_state = _mamba1_kernel(
            inputs=[hidden_states.reshape(b, d), A_log, B, C, D, delta, state],
            template=[("TYPE", input_type), ("D", d), ("Ds", ds)],
            grid=(32, d, b),
            threadgroup=(32, 1, 1),
            output_shapes=[(b, d), (b, d, ds)],
            output_dtypes=[input_type, input_type],
        )
        return out.reshape(b, 1, d), new_state
    return ssm_v1_step(hidden_states, A_log, B, C, D, delta, state)
