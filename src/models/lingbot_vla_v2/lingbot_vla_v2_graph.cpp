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

#include "lingbot_vla_v2_graph.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace lingbot_vla_v2 {

// ────────────────────────────────────────────────────────────────────────────
// FA K/V dtype.  launch_fattn (fattn-common.cuh) casts F32 K/V to F16 with
// ggml_cuda_cast<half>(float) — the exact same conversion ggml_cast uses, so
// holding the cache in F16 is bit-exact (verified: identical action chunks).
// It moves the cast of the 230-token prefix out of the 10-step denoise loop
// (done once per layer instead of once per step) and replaces the per-step
// full-length concat with a copy into the cache tail.
// VLA_LINGBOT_V2_FA_F16=0 disables it (A/B switch).
// ────────────────────────────────────────────────────────────────────────────

bool fa_f16_kv() {
    static const bool on = []() {
        const char * e = std::getenv("VLA_LINGBOT_V2_FA_F16");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

// ────────────────────────────────────────────────────────────────────────────
// Norms / RoPE primitives
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * qk_norm_3d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    // x: [head_dim, n_heads, seq] — rms_norm normalises along ne[0] (per head),
    // w: [head_dim] broadcasts along ne[0].
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

ggml_tensor * apply_imrope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, float theta) {
    // Qwen3-VL interleaved M-RoPE.  mrope_section = [24, 20, 20, 0]; the last
    // section is unused because IMROPE derives the "extra" pairs from the t
    // plane of the packed 4-plane positions (see the converter notes).
    int sections[GGML_MROPE_SECTIONS] = { 24, 20, 20, 0 };
    return ggml_rope_multi(ctx, x, pos, nullptr,
                           /* n_dims   */ 128,
                           sections,
                           /* mode     */ GGML_ROPE_TYPE_IMROPE,
                           /* n_ctx_orig */ 0,
                           /* freq_base */ theta,
                           /* freq_scale */ 1.f,
                           /* ext_factor */ 0.f,
                           /* attn_factor */ 1.f,
                           /* beta_fast  */ 32.f,
                           /* beta_slow  */ 1.f);
}

// ────────────────────────────────────────────────────────────────────────────
// Joint-attention core (pi05 op pattern; F32 scores + softmax)
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * attention_out(ggml_context *      ctx,
                            ggml_tensor *       q,
                            ggml_tensor *       k,
                            ggml_tensor *       v,
                            ggml_tensor *       mask,
                            const LayerParams & p,
                            int64_t             q_len) {
    const int64_t hd = p.head_dim;
    const float   scale = 1.f / std::sqrt(static_cast<float>(hd));

    if (p.use_flash) {
        // ggml_flash_attn_ext: q [hd, q_len, n_q_heads], k/v [hd, k_len, n_kv_heads]
        // (v not transposed), additive F16 mask [k_len, q_len].  The result is
        // laid out as [hd, n_q_heads, q_len] contiguous, which is already the
        // head-major memory order the output projection expects.
        //
        // The permutes only swap ne[1]/ne[2]; the CUDA FA kernels read k/v/q
        // through nb[1]/nb[2] (see fattn-common.cuh: k_row_stride/k_head_stride),
        // so materialising them with ggml_cont is pure bookkeeping traffic.
        // VLA_LINGBOT_V2_FA_COPY=1 restores the copies for A/B measurement.
        static const bool fa_copy = std::getenv("VLA_LINGBOT_V2_FA_COPY") != nullptr;
        ggml_tensor * Q = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * K = ggml_permute(ctx, k, 0, 2, 1, 3);
        ggml_tensor * V = ggml_permute(ctx, v, 0, 2, 1, 3);
        if (fa_copy) {
            Q = ggml_cont(ctx, Q);
            K = ggml_cont(ctx, K);
            V = ggml_cont(ctx, V);
        }
        ggml_tensor * m16 = mask ? ggml_cast(ctx, mask, GGML_TYPE_F16) : nullptr;
        ggml_tensor * kqv = ggml_flash_attn_ext(ctx, Q, K, V, m16, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, kqv, p.n_q_heads * hd, q_len);
    }

    // q: [hd, n_q_heads, q_len]  → Q: [hd, q_len, n_q_heads]
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    // k: [hd, n_kv_heads, k_len] → K: [hd, k_len, n_kv_heads]
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    // v: [hd, n_kv_heads, k_len] → V: [k_len, hd, n_kv_heads]
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    // kq: [k_len, q_len, n_q_heads] — softmax runs along ne[0] (keys).
    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale, 0.f);

    // kqv: [hd, q_len, n_q_heads]
    ggml_tensor * kqv = ggml_mul_mat(ctx, V, attn);

    // → [n_q_heads * hd, q_len]
    return ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), p.n_q_heads * hd, q_len);
}

// ────────────────────────────────────────────────────────────────────────────
// VLM (Qwen3-VL text tower) prefix layer
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * build_vlm_layer(ggml_context *      ctx,
                              const VlmLayerW &   w,
                              ggml_tensor *       x_in,
                              ggml_tensor *       pos,
                              ggml_tensor *       mask,
                              const LayerParams & p,
                              int64_t             seq,
                              bool                f16_kv,
                              ggml_tensor **      k_out,
                              ggml_tensor **       v_out) {
    const int64_t hd  = p.head_dim;
    const int64_t nq  = p.n_q_heads;
    const int64_t nkv = p.n_kv_heads;
    const float   eps = p.rms_eps;

    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, eps), w.attn_norm);

    // Share a single F32→BF16 cast across Q/K/V (bit-exact), like pi05.
    ggml_tensor * xb = p.use_bf16 ? ggml_cast(ctx, x_norm, GGML_TYPE_BF16) : x_norm;
    ggml_tensor * q  = ggml_mul_mat(ctx, w.Wq, xb);
    ggml_tensor * k  = ggml_mul_mat(ctx, w.Wk, xb);
    ggml_tensor * v  = ggml_mul_mat(ctx, w.Wv, xb);

    ggml_tensor * q3 = qk_norm_3d(ctx, ggml_reshape_3d(ctx, q, hd, nq, seq), w.q_norm, eps);
    ggml_tensor * k3 = qk_norm_3d(ctx, ggml_reshape_3d(ctx, k, hd, nkv, seq), w.k_norm, eps);
    ggml_tensor * v3 = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    ggml_tensor * q3r = apply_imrope(ctx, q3, pos, p.rope_theta);
    ggml_tensor * k3r = apply_imrope(ctx, k3, pos, p.rope_theta);

    if (f16_kv) {
        // Feeding the FA kernel F16 directly skips its internal F32->F16 pass.
        k3r = ggml_cast(ctx, k3r, GGML_TYPE_F16);
        v3  = ggml_cast(ctx, v3,  GGML_TYPE_F16);
    }

    if (k_out) {
        *k_out = k3r;
    }
    if (v_out) {
        *v_out = v3;
    }

    ggml_tensor * att_pre = attention_out(ctx, q3r, k3r, v3, mask, p, seq);
    ggml_tensor * o_out   = ggml_mul_mat(ctx, w.Wo, att_pre);
    ggml_tensor * h1      = ggml_add(ctx, x_in, o_out);

    // SwiGLU FFN (Qwen3-VL: silu(gate) * up).
    ggml_tensor * norm_mlp  = ggml_mul(ctx, ggml_rms_norm(ctx, h1, eps), w.ffn_norm);
    ggml_tensor * nb        = p.use_bf16 ? ggml_cast(ctx, norm_mlp, GGML_TYPE_BF16) : norm_mlp;
    ggml_tensor * gate      = ggml_mul_mat(ctx, w.Wgate, nb);
    ggml_tensor * up        = ggml_mul_mat(ctx, w.Wup, nb);
    ggml_tensor * mlp_out   = ggml_mul_mat(ctx, w.Wdown, ggml_swiglu_split(ctx, gate, up));
    return ggml_add(ctx, h1, mlp_out);
}

// ────────────────────────────────────────────────────────────────────────────
// AdaRMSNorm (time-conditioned)
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * adarms_norm(ggml_context * ctx, ggml_tensor * x, const AdanormW & w, ggml_tensor * cond, float eps) {
    // cond: [768, 1] raw sinusoidal time embedding (fp32).
    ggml_tensor * gamma = ggml_add(ctx, ggml_mul_mat(ctx, w.gamma_w, cond), w.gamma_b);  // [768, 1]
    ggml_tensor * beta  = ggml_add(ctx, ggml_mul_mat(ctx, w.beta_w, cond), w.beta_b);    // [768, 1]

    ggml_tensor * base   = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w.w);  // [768, n]
    ggml_tensor * scaled = ggml_add(ctx, base, ggml_mul(ctx, base, gamma));  // (1 + gamma) * base
    return ggml_add(ctx, scaled, beta);
}

// ────────────────────────────────────────────────────────────────────────────
// Token-choice MoE FFN (llama.cpp build_moe_ffn sequence, sigmoid router)
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * build_moe_ffn(ggml_context *      ctx,
                            ggml_cgraph *       gf,
                            const ExpertLayerW & w,
                            ggml_tensor *       x,
                            int64_t             n_tok,
                            const MoeParams &   moe) {
    const int64_t E = moe.n_experts;
    const int64_t k = moe.top_k;
    const int64_t hidden = x->ne[0];
    GGML_ASSERT(x->ne[1] == n_tok);

    ggml_tensor * moe_out = nullptr;

    // Router: sigmoid logits, bias-corrected top-k selection.
    ggml_tensor * logits = ggml_mul_mat(ctx, w.moe_gate, x);                       // [E, n]
    ggml_tensor * probs  = ggml_sigmoid(ctx, logits);                              // [E, n]
    ggml_tensor * selection = ggml_add(ctx, probs, w.experts_bias);                // [E, n]
    ggml_tensor * sel = ggml_argsort_top_k(ctx, selection, static_cast<int>(k));   // [k, n] i32

    // Gather per-expert weights (raw probs, without the selection bias).
    ggml_tensor * probs3d = ggml_reshape_3d(ctx, probs, 1, E, n_tok);              // [1, E, n]
    ggml_tensor * weights = ggml_get_rows(ctx, probs3d, sel);                      // [1, k, n]

    // norm_topk_prob: w / (sum + 1e-20), then routed_scaling_factor.
    ggml_tensor * w2    = ggml_reshape_2d(ctx, weights, k, n_tok);                 // [k, n]
    ggml_tensor * wsum  = ggml_sum_rows(ctx, w2);                                  // [1, n]
    wsum                = ggml_clamp(ctx, wsum, 1e-20f, INFINITY);
    ggml_tensor * wnorm = ggml_div(ctx, w2, wsum);                                 // [k, n]
    wnorm               = ggml_scale(ctx, wnorm, moe.scale);
    weights             = ggml_reshape_3d(ctx, wnorm, 1, k, n_tok);                // [1, k, n]

    // Register the routing sub-graph early (llama.cpp does the same).
    ggml_build_forward_expand(gf, weights);

    // Routed experts via mul_mat_id.  b must be [hidden, 1, n_tok].
    ggml_tensor * xb   = ggml_reshape_3d(ctx, x, hidden, 1, n_tok);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, w.exp_up, xb, sel);                  // [ff, k, n]
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w.exp_gate, xb, sel);                // [ff, k, n]
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, w.exp_down, act, sel);            // [hidden, k, n]

    // Weight each expert output (broadcast over ne[0]).
    experts = ggml_mul(ctx, experts, weights);                                     // [hidden, k, n]
    ggml_build_forward_expand(gf, experts);

    // Aggregate the k expert slices (per-k views + adds, llama.cpp pattern).
    moe_out = ggml_view_2d(ctx, experts, hidden, n_tok, experts->nb[2], 0);
    ggml_build_forward_expand(gf, moe_out);
    for (int64_t i = 1; i < k; ++i) {
        ggml_tensor * vi = ggml_view_2d(ctx, experts, hidden, n_tok, experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(gf, vi);
        moe_out = ggml_add(ctx, moe_out, vi);
        ggml_build_forward_expand(gf, moe_out);
    }

    // Shared expert: down(silu(gate) * up), optionally gated by
    // sigmoid(router(x)) when the checkpoint provides a shared_expert_gate.
    ggml_tensor * sg     = ggml_mul_mat(ctx, w.sh_gate, x);                        // [sff, n]
    ggml_tensor * su     = ggml_mul_mat(ctx, w.sh_up, x);                          // [sff, n]
    ggml_tensor * sh     = ggml_mul_mat(ctx, w.sh_down, ggml_swiglu_split(ctx, sg, su));  // [hidden, n]
    ggml_tensor * shared = sh;
    if (moe.use_shared_gate) {
        GGML_ASSERT(w.sh_router != nullptr);
        ggml_tensor * router = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w.sh_router, x));  // [1, n]
        shared               = ggml_mul(ctx, sh, router);                            // [hidden, n]
    }

    return ggml_add(ctx, moe_out, shared);
}

// ────────────────────────────────────────────────────────────────────────────
// Expert (Qwen2 + AdaRMS + MoE) suffix layer on cached prefix K/V
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * build_expert_layer(ggml_context *       ctx,
                                 ggml_cgraph *        gf,
                                 const ExpertLayerW & w,
                                 ggml_tensor *        x_in,
                                 ggml_tensor *        pos,
                                 ggml_tensor *        cond,
                                 ggml_tensor *        cK,
                                 ggml_tensor *        cV,
                                 ggml_tensor *        mask,
                                 const LayerParams &  p,
                                 const MoeParams &    moe,
                                 int64_t              seq,
                                 ggml_tensor **       kfull_out,
                                 ggml_tensor **       vfull_out) {
    const int64_t hd  = p.head_dim;
    const int64_t nq  = p.n_q_heads;
    const int64_t nkv = p.n_kv_heads;
    const float   eps = p.rms_eps;

    // Qwen2-style attention with biases (kept in F32: the expert is small).
    ggml_tensor * xn = adarms_norm(ctx, x_in, w.attn, cond, eps);
    ggml_tensor * q  = ggml_add(ctx, ggml_mul_mat(ctx, w.Wq, xn), w.bq);
    ggml_tensor * k  = ggml_add(ctx, ggml_mul_mat(ctx, w.Wk, xn), w.bk);
    ggml_tensor * v  = ggml_add(ctx, ggml_mul_mat(ctx, w.Wv, xn), w.bv);

    ggml_tensor * q3 = ggml_reshape_3d(ctx, q, hd, nq, seq);
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v3 = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    ggml_tensor * q3r = apply_imrope(ctx, q3, pos, p.rope_theta);
    ggml_tensor * k3r = apply_imrope(ctx, k3, pos, p.rope_theta);

    // Joint attention: prefix K/V from the cache + suffix K/V.
    ggml_tensor * K_full = nullptr;
    ggml_tensor * V_full = nullptr;
    if (cK != nullptr && cK->type == GGML_TYPE_F16) {
        // cK/cV are the persistent F16 joint K/V [hd, nkv, P + seq]: the prefix
        // slice is staged once by the caller, so here we only overwrite the
        // suffix tail instead of re-concatenating the whole cache every step.
        const int64_t P = cK->ne[2] - seq;
        GGML_ASSERT(P >= 0);
        K_full = cK;
        V_full = cV;
        ggml_tensor * kdst = ggml_view_3d(ctx, cK, hd, nkv, seq, cK->nb[1], cK->nb[2], P * cK->nb[2]);
        ggml_tensor * vdst = ggml_view_3d(ctx, cV, hd, nkv, seq, cV->nb[1], cV->nb[2], P * cV->nb[2]);
        // The copies write into the cache; nothing consumes their result, so
        // they must be registered with the graph explicitly.
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cast(ctx, k3r, GGML_TYPE_F16), kdst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cast(ctx, v3, GGML_TYPE_F16), vdst));
    } else {
        K_full = cK ? ggml_concat(ctx, cK, k3r, 2) : k3r;  // [hd, nkv, P + seq]
        V_full = cV ? ggml_concat(ctx, cV, v3, 2) : v3;
    }
    if (kfull_out) {
        *kfull_out = K_full;
    }
    if (vfull_out) {
        *vfull_out = V_full;
    }

    ggml_tensor * att_pre = attention_out(ctx, q3r, K_full, V_full, mask, p, seq);
    ggml_tensor * o_out   = ggml_mul_mat(ctx, w.Wo, att_pre);
    ggml_tensor * h1      = ggml_add(ctx, x_in, o_out);

    ggml_tensor * hn = adarms_norm(ctx, h1, w.ffn, cond, eps);
    ggml_tensor * moe_out = build_moe_ffn(ctx, gf, w, hn, seq, moe);
    return ggml_add(ctx, h1, moe_out);
}

// ────────────────────────────────────────────────────────────────────────────
// Suffix input embedding
// ────────────────────────────────────────────────────────────────────────────

ggml_tensor * build_embed_suffix(ggml_context * ctx,
                                 ggml_tensor *  x_t,
                                 ggml_tensor *  state,
                                 ggml_tensor *  t,
                                 ggml_tensor *  W_state,
                                 ggml_tensor *  b_state,
                                 ggml_tensor *  W_ain,
                                 ggml_tensor *  b_ain,
                                 ggml_tensor *  W_mi,
                                 ggml_tensor *  b_mi,
                                 ggml_tensor *  W_mo,
                                 ggml_tensor *  b_mo) {
    // state token: state_proj(state) → [768, 1]
    ggml_tensor * state_emb = ggml_add(ctx, ggml_mul_mat(ctx, W_state, state), b_state);

    // action tokens: action_in_proj(x_t) → [768, 50]
    ggml_tensor * ain = ggml_add(ctx, ggml_mul_mat(ctx, W_ain, x_t), b_ain);

    // time conditioning broadcast over the action tokens.
    ggml_tensor * t50 = ggml_repeat(ctx, t, ain);  // [768, 50]

    // mlp_out(silu(mlp_in(cat(ain, t50))))  → [768, 50]
    ggml_tensor * cat = ggml_concat(ctx, ain, t50, 0);  // [1536, 50]
    ggml_tensor * h   = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, W_mi, cat), b_mi));
    ggml_tensor * at  = ggml_add(ctx, ggml_mul_mat(ctx, W_mo, h), b_mo);

    // [768, 51]: row 0 = state token, rows 1..50 = action tokens.
    return ggml_concat(ctx, state_emb, at, 1);
}

}  // namespace lingbot_vla_v2
