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

/**
 * @file lingbot_vla_v2_graph.h
 * @brief GGML compute-graph builders for the LingBot-VLA-v2 transformer.
 *
 * The model is a joint-attention dual-tower flow-matching VLA:
 *   - VLM tower  = Qwen3-VL text tower (2560 wide, QK-norm, interleaved M-RoPE,
 *                  causal prefix pass fills the KV cache);
 *   - expert tower = Qwen2-style action expert (768 wide, AdaRMSNorm time
 *                  conditioning, per-layer sigmoid-router MoE).  Each suffix
 *                  pass concatenates its K/V onto the cached prefix K/V and
 *                  runs 51 tokens (1 state + 50 action queries).
 *
 * Builders (all tensors stay in the graph context; no side effects):
 *   qk_norm_3d()        — per-head RMSNorm on [hd, heads, seq] (Qwen3-VL).
 *   apply_imrope()      — interleaved M-RoPE, sections [24,20,20,0].
 *   attention_out()     — joint-attention core (pi05 op pattern, F32 softmax).
 *   build_vlm_layer()   — one VLM prefix layer; exports roped K / raw V.
 *   adarms_norm()       — (1+gamma(c)) * (w * rms(x)) + beta(c).
 *   build_moe_ffn()     — token-choice MoE (llama.cpp build_moe_ffn sequence).
 *   build_expert_layer()— one expert layer on cached prefix K/V.
 *   build_embed_suffix()- state/action/time embedding → [768, 51].
 */

#pragma once

#include "ggml.h"
#include "lingbot_vla_v2_weights.h"

#include <cstdint>

namespace lingbot_vla_v2 {

/// Fixed layer geometry shared by the graph builders.
struct LayerParams {
    int64_t hidden     = 2560;  ///< VLM width.
    int64_t n_q_heads  = 32;
    int64_t n_kv_heads = 8;
    int64_t head_dim   = 128;
    float   rms_eps    = 1e-6f;
    float   rope_theta = 5e6f;
    bool    use_bf16   = true;  ///< Cast VLM activations to BF16 before matmuls.
    bool    use_flash  = false; ///< Attention via ggml_flash_attn_ext (CUDA FA).
};

/// MoE router configuration.
struct MoeParams {
    int64_t n_experts = 32;
    int64_t top_k     = 4;
    float   scale     = 4.0f;  ///< routed_scaling_factor.
    /// Sigmoid(Linear(hidden->1)) gating the shared expert output; false adds
    /// the shared expert ungated (robotwin.yaml: use_shared_expert_gate=false).
    bool    use_shared_gate = false;
};

/// Per-head RMSNorm + scale on a [head_dim, n_heads, seq] tensor.
ggml_tensor * qk_norm_3d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps);

/// Interleaved M-RoPE (Qwen3-VL text tower) on a [hd, heads, seq] tensor.
/// @param pos 1-D I32 tensor of 4*seq positions (t, h, w, e planes; the e
///             plane is filled with the t plane to match the HF reference).
ggml_tensor * apply_imrope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, float theta);

/// Joint-attention core.  Two paths, selected by p.use_flash:
///   false — F32 matmul scores + softmax (bit-exact with the HF reference);
///   true  — ggml_flash_attn_ext (CUDA), needs the mask cast to F16.
/// @param q  roped Q [hd, n_q_heads, q_len]
/// @param k  roped K [hd, n_kv_heads, k_len] (prefix-cached + suffix concat)
/// @param v  raw V   [hd, n_kv_heads, k_len]
/// @param mask additive F32 mask [k_len, q_len] (0 = allow, -inf = block)
/// @return attention output [n_q_heads*hd, q_len]
ggml_tensor * attention_out(ggml_context * ctx,
                            ggml_tensor *  q,
                            ggml_tensor *  k,
                            ggml_tensor *  v,
                            ggml_tensor *  mask,
                            const LayerParams & p,
                            int64_t         q_len);

/// Whether the FA joint K/V cache should be held in F16 (default; see the
/// comment in lingbot_vla_v2_graph.cpp).  VLA_LINGBOT_V2_FA_F16=0 turns it off.
bool fa_f16_kv();

/// One VLM (Qwen3-VL text) prefix layer.  Causal mask, fills the KV cache.
/// @param pos  1-D I32 4*seq prefix positions.
/// @param f16_kv  Emit the cached K/V as F16 (only when FA is on).
/// @param k_out / v_out  Output: roped K / raw V ([hd, n_kv, seq]) for caching.
/// @return layer output [hidden, seq]
ggml_tensor * build_vlm_layer(ggml_context *       ctx,
                              const VlmLayerW &    w,
                              ggml_tensor *        x_in,
                              ggml_tensor *        pos,
                              ggml_tensor *        mask,
                              const LayerParams &  p,
                              int64_t              seq,
                              bool                 f16_kv,
                              ggml_tensor **       k_out,
                              ggml_tensor **       v_out);

/// AdaRMSNorm: (1 + gamma(cond)) * (w * rms_norm(x)) + beta(cond).
/// @param cond time conditioning [768, 1] (raw sinusoidal embedding).
ggml_tensor * adarms_norm(ggml_context * ctx, ggml_tensor * x, const AdanormW & w, ggml_tensor * cond, float eps);

/// Token-choice MoE FFN (sigmoid router + bias-corrected top-k + shared expert).
/// Requires @param gf (the compute graph) because intermediate views are
/// registered with ggml_build_forward_expand, following llama.cpp.
/// @param x [768, n_tok] F32 (post-AdaRMSNorm activations).
ggml_tensor * build_moe_ffn(ggml_context *      ctx,
                            ggml_cgraph *       gf,
                            const ExpertLayerW & w,
                            ggml_tensor *       x,
                            int64_t             n_tok,
                            const MoeParams &   moe);

/// One expert (Qwen2 + AdaRMS + MoE) suffix layer on cached prefix K/V.
/// @param cond time conditioning [768, 1] used by both AdaRMSNorms.
/// @param cK / cV  prefix KV cache tensors [hd, n_kv, n_prefix].
/// @param mask additive F32 mask [n_prefix + seq, seq].
/// @param pos  1-D I32 4*seq suffix positions.
/// @param kfull_out / vfull_out  Optional outputs: the roped / raw
///        prefix+suffix concatenated K / V ([hd, n_kv, n_prefix + seq]) —
///        used by the numerical-alignment dumps.
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
                                 ggml_tensor **       kfull_out = nullptr,
                                 ggml_tensor **       vfull_out = nullptr);

/// Suffix input embedding: state_proj(state) ⊕ time-MLP(action_in(x_t), t).
/// @param x_t   noisy actions [55, 50]
/// @param state proprioception [55, 1]
/// @param t     raw sinusoidal time embedding [768, 1] (also the AdaRMS cond)
/// @return [768, 51] (row 0 = state token, rows 1..50 = action tokens)
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
                                 ggml_tensor *  b_mo);

}  // namespace lingbot_vla_v2
