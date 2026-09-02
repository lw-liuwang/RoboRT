// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pi05_graph.h"

#include "ggml.h"
#include "model.h"
#include "pi05_weights.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace pi05 {

// ────────────────────────────────────────────────────────────────────────────
// Internal helper: self-attention sub-graph shared by both layer builders.
// ────────────────────────────────────────────────────────────────────────────

static ggml_tensor * build_attention(ggml_context *      ctx,
                                     ggml_tensor *       q,
                                     ggml_tensor *       k,
                                     ggml_tensor *       v,
                                     ggml_tensor *       positions,
                                     ggml_tensor *       cached_K,
                                     ggml_tensor *       cached_V,
                                     ggml_tensor *       mask,
                                     const vla::Config & cfg,
                                     int64_t             seq,
                                     float               rope_base,
                                     bool                use_flash,
                                     ggml_tensor **      k_out,
                                     ggml_tensor **      v_out) {
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t qf  = nq * hd;

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq, seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    auto rope = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, nullptr, static_cast<int>(hd), GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.f,
                             0.f, 1.f, 32.f, 1.f);
    };
    ggml_tensor * q_rope = rope(q_h);
    ggml_tensor * k_rope = rope(k_h);

    if (k_out) {
        *k_out = k_rope;
    }
    if (v_out) {
        *v_out = v_h;
    }

    ggml_tensor * K_full = cached_K ? ggml_concat(ctx, cached_K, k_rope, 2) : k_rope;
    ggml_tensor * V_full = cached_V ? ggml_concat(ctx, cached_V, v_h, 2) : v_h;

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));

    const float scale = 1.f / std::sqrt(static_cast<float>(hd));

    ggml_tensor * att_pre = nullptr;
    if (use_flash) {
        // Flash attention: Q stays F32; K/V cast to F16 for tensor cores.
        // The all-zeros pi0.5 suffix mask is irrelevant here (null).
        ggml_tensor * Vf  = ggml_cont(ctx, ggml_permute(ctx, V_full, 0, 2, 1, 3));
        ggml_tensor * Kh  = ggml_cast(ctx, K, GGML_TYPE_F16);
        ggml_tensor * Vh  = ggml_cast(ctx, Vf, GGML_TYPE_F16);
        ggml_tensor * kqv = ggml_flash_attn_ext(ctx, Q, Kh, Vh, nullptr, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
        att_pre = ggml_reshape_2d(ctx, ggml_cont(ctx, kqv), qf, seq);
    } else {
        ggml_tensor * V  = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale, 0.f);
        ggml_tensor * kqv  = ggml_mul_mat(ctx, V, attn);
        att_pre            = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq);
    }
    return att_pre;
}

// ────────────────────────────────────────────────────────────────────────────
// Public functions
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * build_gemma_layer(ggml_context *      ctx,
                                const GemmaLayerW & w,
                                ggml_tensor *       x_in,
                                ggml_tensor *       positions,
                                const vla::Config & cfg,
                                int64_t             seq,
                                float               rope_base,
                                ggml_tensor *       cached_K,
                                ggml_tensor *       cached_V,
                                ggml_tensor *       mask,
                                bool                use_flash,
                                ggml_tensor **      k_out,
                                ggml_tensor **      v_out) {
    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.ln_in);

    // Share a single F32→BF16 cast across Q/K/V projections (bit-exact).
    ggml_tensor * x_bf16 = ggml_cast(ctx, x_norm, GGML_TYPE_BF16);
    ggml_tensor * q      = ggml_mul_mat(ctx, w.Wq, x_bf16);
    ggml_tensor * k      = ggml_mul_mat(ctx, w.Wk, x_bf16);
    ggml_tensor * v      = ggml_mul_mat(ctx, w.Wv, x_bf16);

    ggml_tensor * att_pre = build_attention(ctx, q, k, v, positions, cached_K, cached_V, mask, cfg, seq, rope_base,
                                            use_flash, k_out, v_out);
    ggml_tensor * o_out   = ggml_mul_mat(ctx, w.Wo, att_pre);
    ggml_tensor * h1      = ggml_add(ctx, x_in, o_out);

    ggml_tensor * norm_mlp  = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
    ggml_tensor * norm_bf16 = ggml_cast(ctx, norm_mlp, GGML_TYPE_BF16);
    ggml_tensor * gate      = ggml_mul_mat(ctx, w.Wgate, norm_bf16);
    ggml_tensor * up        = ggml_mul_mat(ctx, w.Wup, norm_bf16);
    ggml_tensor * mlp_out   = ggml_mul_mat(ctx, w.Wdown, ggml_mul(ctx, ggml_gelu(ctx, gate), up));
    return ggml_add(ctx, h1, mlp_out);
}

ggml_tensor * adarms_norm(ggml_context *      ctx,
                          ggml_tensor *       x,
                          ggml_tensor *       dense_w,
                          ggml_tensor *       dense_b,
                          ggml_tensor *       cond,
                          const vla::Config & cfg,
                          ggml_tensor **      gate_out) {
    // AdaRMSNorm: RMSNorm(x) * (1 + scale) + shift, where [scale, shift, gate]
    // are slices of the dense projection of the time conditioning vector.
    ggml_tensor * mod       = ggml_add(ctx, ggml_mul_mat(ctx, dense_w, cond), dense_b);
    const size_t  row_bytes = static_cast<size_t>(cfg.expert_h) * sizeof(float);
    ggml_tensor * scale     = ggml_view_2d(ctx, mod, cfg.expert_h, 1, row_bytes, 0);
    ggml_tensor * shift     = ggml_view_2d(ctx, mod, cfg.expert_h, 1, row_bytes, row_bytes);
    ggml_tensor * gate      = ggml_view_2d(ctx, mod, cfg.expert_h, 1, row_bytes, 2 * row_bytes);
    ggml_tensor * base      = ggml_rms_norm(ctx, x, cfg.rms_eps);
    ggml_tensor * scaled    = ggml_add(ctx, base, ggml_mul(ctx, base, scale));
    if (gate_out) {
        *gate_out = gate;
    }
    return ggml_add(ctx, scaled, shift);
}

ggml_tensor * gated_residual(ggml_context * ctx, ggml_tensor * x, ggml_tensor * y, ggml_tensor * gate) {
    return gate ? ggml_add(ctx, x, ggml_mul(ctx, y, gate)) : ggml_add(ctx, x, y);
}

ggml_tensor * build_adarms_gemma_layer(ggml_context *         ctx,
                                       const AdaGemmaLayerW & w,
                                       ggml_tensor *          x_in,
                                       ggml_tensor *          positions,
                                       ggml_tensor *          adarms_cond,
                                       const vla::Config &    cfg,
                                       int64_t                seq,
                                       float                  rope_base,
                                       ggml_tensor *          cached_K,
                                       ggml_tensor *          cached_V,
                                       ggml_tensor *          mask,
                                       bool                   use_flash) {
    ggml_tensor * attn_gate = nullptr;
    ggml_tensor * x_norm    = adarms_norm(ctx, x_in, w.ln_in_W, w.ln_in_b, adarms_cond, cfg, &attn_gate);

    // Share one F32→BF16 cast for Q/K/V (bit-exact).
    ggml_tensor * x_bf16 = ggml_cast(ctx, x_norm, GGML_TYPE_BF16);
    ggml_tensor * q      = ggml_mul_mat(ctx, w.Wq, x_bf16);
    ggml_tensor * k      = ggml_mul_mat(ctx, w.Wk, x_bf16);
    ggml_tensor * v      = ggml_mul_mat(ctx, w.Wv, x_bf16);

    ggml_tensor * att_pre = build_attention(ctx, q, k, v, positions, cached_K, cached_V, mask, cfg, seq, rope_base,
                                            use_flash, nullptr, nullptr);
    ggml_tensor * o_out   = ggml_mul_mat(ctx, w.Wo, att_pre);
    ggml_tensor * h1      = gated_residual(ctx, x_in, o_out, attn_gate);

    ggml_tensor * mlp_gate  = nullptr;
    ggml_tensor * norm_mlp  = adarms_norm(ctx, h1, w.ln_post_W, w.ln_post_b, adarms_cond, cfg, &mlp_gate);
    ggml_tensor * norm_bf16 = ggml_cast(ctx, norm_mlp, GGML_TYPE_BF16);
    ggml_tensor * gate      = ggml_mul_mat(ctx, w.Wgate, norm_bf16);
    ggml_tensor * up        = ggml_mul_mat(ctx, w.Wup, norm_bf16);
    ggml_tensor * mlp_out   = ggml_mul_mat(ctx, w.Wdown, ggml_mul(ctx, ggml_gelu(ctx, gate), up));
    return gated_residual(ctx, h1, mlp_out, mlp_gate);
}

ggml_tensor * build_embed_suffix(ggml_context * ctx,
                                 ggml_tensor *  x,
                                 ggml_tensor *  time_vec,
                                 ggml_tensor *  W_ain,
                                 ggml_tensor *  b_ain,
                                 ggml_tensor *  W_tm1,
                                 ggml_tensor *  b_tm1,
                                 ggml_tensor *  W_tm2,
                                 ggml_tensor *  b_tm2,
                                 ggml_tensor ** adarms_cond) {
    // Map action tokens to the expert hidden space.
    ggml_tensor * action_emb = ggml_add(ctx, ggml_mul_mat(ctx, W_ain, x), b_ain);
    // Two-layer MLP on the timestep embedding → AdaRMSNorm conditioning vector.
    ggml_tensor * t1         = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, W_tm1, time_vec), b_tm1));
    ggml_tensor * t2         = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, W_tm2, t1), b_tm2));
    if (adarms_cond) {
        *adarms_cond = t2;
    }
    return action_emb;
}

}  // namespace pi05
