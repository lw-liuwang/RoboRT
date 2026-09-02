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
 * @file pi05_weights.h
 * @brief Weight tensor structs for the pi0.5 transformer layers and helpers
 *        to identify Gemma-norm tensors (which need a +1 bias on load).
 *
 * GemmaLayerW    — plain Gemma transformer layer (prefix / VLM tower).
 * AdaGemmaLayerW — adaptive Gemma layer with per-step AdaRMSNorm conditioning
 *                  (action expert).
 */

#pragma once

#include "ggml.h"

#include <string>

namespace pi05 {

/// Weights for one Gemma transformer layer (attention + FFN).
struct GemmaLayerW {
    ggml_tensor * ln_in   = nullptr;  ///< Pre-attention RMSNorm scale.
    ggml_tensor * Wq      = nullptr;  ///< Query projection.
    ggml_tensor * Wk      = nullptr;  ///< Key projection.
    ggml_tensor * Wv      = nullptr;  ///< Value projection.
    ggml_tensor * Wo      = nullptr;  ///< Output projection.
    ggml_tensor * ln_post = nullptr;  ///< Post-attention RMSNorm scale.
    ggml_tensor * Wgate   = nullptr;  ///< FFN gate projection.
    ggml_tensor * Wup     = nullptr;  ///< FFN up projection.
    ggml_tensor * Wdown   = nullptr;  ///< FFN down projection.
};

/// Weights for one AdaRMSNorm-conditioned Gemma layer (action expert).
/// The two norm projections (dense.weight + dense.bias) map the time
/// condition to per-token scale, shift, and gate coefficients.
struct AdaGemmaLayerW {
    ggml_tensor * ln_in_W   = nullptr;  ///< Pre-attn AdaRMSNorm projection weight.
    ggml_tensor * ln_in_b   = nullptr;  ///< Pre-attn AdaRMSNorm projection bias.
    ggml_tensor * Wq        = nullptr;
    ggml_tensor * Wk        = nullptr;
    ggml_tensor * Wv        = nullptr;
    ggml_tensor * Wo        = nullptr;
    ggml_tensor * ln_post_W = nullptr;  ///< Post-attn AdaRMSNorm projection weight.
    ggml_tensor * ln_post_b = nullptr;  ///< Post-attn AdaRMSNorm projection bias.
    ggml_tensor * Wgate     = nullptr;
    ggml_tensor * Wup       = nullptr;
    ggml_tensor * Wdown     = nullptr;
};

/// Returns true when `name` is a Gemma RMSNorm weight that requires the +1
/// stored-in-GGUF-as-zero bias to be applied on load.
inline bool is_gemma_norm(const std::string & name) {
    return name.find("norm.weight") != std::string::npos;
}

}  // namespace pi05
