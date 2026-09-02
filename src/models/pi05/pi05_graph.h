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
 * @file pi05_graph.h
 * @brief GGML compute-graph builders for the pi0.5 transformer layers.
 *
 * build_gemma_layer()       — one Gemma prefix-tower layer (cross-attention
 *                             keys/values output for caching).
 * build_adarms_gemma_layer() — one action-expert layer with AdaRMSNorm
 *                             conditioning on the per-step time embedding.
 * adarms_norm()             — shared AdaRMSNorm primitive (scale+shift+gate).
 * gated_residual()          — x + y*gate (or plain x + y when gate is null).
 * build_embed_suffix()      — action-token embedding + time MLP for the expert.
 */

#pragma once

#include "ggml.h"
#include "model.h"
#include "pi05_weights.h"

namespace pi05 {

/// Build the compute graph for one plain Gemma transformer layer.
/// @param k_out / v_out  Optional; when non-null, the new KV tensors are
///                       written here so the caller can cache them for the
///                       cross-attention in the action expert.
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
                                ggml_tensor **      v_out);

/// AdaRMSNorm: RMSNorm(x) * (1 + scale(cond)) + shift(cond).
/// When gate_out is non-null, the third projection slice (gate) is written
/// there for use in the residual path.
ggml_tensor * adarms_norm(ggml_context *      ctx,
                          ggml_tensor *       x,
                          ggml_tensor *       dense_w,
                          ggml_tensor *       dense_b,
                          ggml_tensor *       cond,
                          const vla::Config & cfg,
                          ggml_tensor **      gate_out);

/// Gated residual add: returns x + y*gate when gate is non-null, x + y otherwise.
ggml_tensor * gated_residual(ggml_context * ctx, ggml_tensor * x, ggml_tensor * y, ggml_tensor * gate);

/// Build the compute graph for one AdaRMSNorm-conditioned Gemma layer
/// (action expert).  Uses pre-computed prefix KV caches cK / cV.
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
                                       bool                   use_flash);

/// Build the action-token input embedding and the time MLP for one denoise step.
/// @param adarms_cond  Output: the time conditioning vector (t2) used by AdaRMSNorm.
ggml_tensor * build_embed_suffix(ggml_context * ctx,
                                 ggml_tensor *  x,
                                 ggml_tensor *  time_vec,
                                 ggml_tensor *  W_ain,
                                 ggml_tensor *  b_ain,
                                 ggml_tensor *  W_tm1,
                                 ggml_tensor *  b_tm1,
                                 ggml_tensor *  W_tm2,
                                 ggml_tensor *  b_tm2,
                                 ggml_tensor ** adarms_cond);

}  // namespace pi05
