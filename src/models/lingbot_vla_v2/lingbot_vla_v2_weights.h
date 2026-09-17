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
 * @file lingbot_vla_v2_weights.h
 * @brief Weight tensor structs for the LingBot-VLA-v2 transformer layers.
 *
 * VlmLayerW     — Qwen3-VL text-tower layer (RMSNorm + QK-norm + SwiGLU).
 * ExpertLayerW  — Qwen2-style action-expert layer with AdaRMSNorm conditioning
 *                 and a per-layer token MoE (sigmoid router, top-4 of 32,
 *                 routed + shared experts).
 *
 * All norms are plain RMSNorm (no Gemma +1 bias).  Router gate, expert bias
 * and every FiLM/bias tensor are resident in F32; the big matrices can be
 * BF16/quantised depending on the matmul type.
 */

#pragma once

#include "ggml.h"

#include <string>

namespace lingbot_vla_v2 {

/// Weights for one VLM (Qwen3-VL text tower) layer.
struct VlmLayerW {
    ggml_tensor * attn_norm = nullptr;  ///< Pre-attention RMSNorm scale [2560] (F32).
    ggml_tensor * Wq        = nullptr;  ///< Query projection [4096, 2560].
    ggml_tensor * q_norm    = nullptr;  ///< Per-head Q RMSNorm scale [128] (F32).
    ggml_tensor * Wk        = nullptr;  ///< Key projection [1024, 2560].
    ggml_tensor * k_norm    = nullptr;  ///< Per-head K RMSNorm scale [128] (F32).
    ggml_tensor * Wv        = nullptr;  ///< Value projection [1024, 2560].
    ggml_tensor * Wo        = nullptr;  ///< Output projection [2560, 4096].
    ggml_tensor * ffn_norm  = nullptr;  ///< Post-attention RMSNorm scale [2560] (F32).
    ggml_tensor * Wgate     = nullptr;  ///< FFN gate projection [9728, 2560].
    ggml_tensor * Wup       = nullptr;  ///< FFN up projection [9728, 2560].
    ggml_tensor * Wdown     = nullptr;  ///< FFN down projection [2560, 9728].
};

/// FiLM conditioning tensors of one AdaRMSNorm:
///   out = (1 + gamma(c)) * (w * rms(x)) + beta(c),  c = 768-dim time embedding.
struct AdanormW {
    ggml_tensor * w      = nullptr;  ///< RMSNorm scale [768] (F32).
    ggml_tensor * gamma_w = nullptr;  ///< FiLM gamma Linear weight [768, 768].
    ggml_tensor * gamma_b = nullptr;  ///< FiLM gamma Linear bias [768] (F32).
    ggml_tensor * beta_w  = nullptr;  ///< FiLM beta Linear weight [768, 768].
    ggml_tensor * beta_b  = nullptr;  ///< FiLM beta Linear bias [768] (F32).
};

/// Weights for one action-expert (Qwen2-style + AdaRMS + MoE) layer.
struct ExpertLayerW {
    AdanormW attn;   ///< Pre-attention AdaRMSNorm.
    AdanormW ffn;    ///< Post-attention (pre-MoE) AdaRMSNorm.

    ggml_tensor * Wq = nullptr, *bq = nullptr;  ///< [4096, 768] + bias (F32).
    ggml_tensor * Wk = nullptr, *bk = nullptr;  ///< [1024, 768] + bias (F32).
    ggml_tensor * Wv = nullptr, *bv = nullptr;  ///< [1024, 768] + bias (F32).
    ggml_tensor * Wo = nullptr;                 ///< [768, 4096], no bias.

    // MoE (Qwen2 token-choice, sigmoid router, norm_topk_prob, scale 4.0)
    ggml_tensor * moe_gate     = nullptr;  ///< Router logits [n_experts, 768] (F32).
    ggml_tensor * experts_bias = nullptr;  ///< e_score_correction_bias [n_experts] (F32).
    ggml_tensor * exp_gate     = nullptr;  ///< Routed experts gate_proj, 3D [768, 512, E].
    ggml_tensor * exp_up       = nullptr;  ///< Routed experts up_proj,   3D [768, 512, E].
    ggml_tensor * exp_down     = nullptr;  ///< Routed experts down_proj, 3D [512, 768, E].
    ggml_tensor * sh_gate      = nullptr;  ///< Shared expert gate_proj [704, 768].
    ggml_tensor * sh_up        = nullptr;  ///< Shared expert up_proj   [704, 768].
    ggml_tensor * sh_down      = nullptr;  ///< Shared expert down_proj [768, 704].
    /// Optional sigmoid gate [1, 768] (F32).  Only present when the training
    /// config set use_shared_expert_gate=true; otherwise the shared expert is
    /// added ungated (robotwin checkpoint has no such weights).
    ggml_tensor * sh_router    = nullptr;
};

}  // namespace lingbot_vla_v2
