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
 * @file pi05_vispruner.h
 * @brief Two-stage visual token pruning (importance + diversity) for pi0.5.
 *
 * vispruner_prune() reduces a set of image-patch embeddings from N tokens to
 * `target_tokens` using:
 *   Stage 1 — importance: keep the top (target * important_ratio) patches
 *             ranked by mean attention received across all heads / queries.
 *   Stage 2 — diversity: iteratively pair the remaining patches by cosine
 *             similarity and drop the most similar pairs until the target is met.
 *
 * The caller supplies the attention matrix from the last CLIP layer
 * (clip_get_last_attn_data) immediately after encoding each camera view.
 * Multi-view inputs must call this once per view (the CLIP attention buffer
 * is overwritten on every encode).
 */

#pragma once

#include <cstdint>
#include <vector>

namespace pi05 {

/// Prune `img_emb` in-place from `n_img_tokens` to `target_tokens`.
///
/// @param img_emb         [in/out] row-major float embeddings, size = N * hidden_pl.
/// @param n_img_tokens    [in/out] N before call, T after.
/// @param hidden_pl       Embedding dimension (columns per token).
/// @param attn_data       Attention weights [n_heads * N * N], row-major
///                        (attn[h * N*N + q * N + p] = attention from query q to key p in head h).
/// @param n_patches       Must equal N (validated internally).
/// @param n_heads         Number of attention heads in the CLIP last layer.
/// @param target_tokens   T: desired token count after pruning.
/// @param important_ratio Fraction of T selected by importance (remainder by diversity).
/// @param keep_order      If true, retain the original positional order of kept tokens
///                        (FastV-style spatial preservation); if false, reorder by importance score.
void vispruner_prune(std::vector<float> & img_emb,
                     int64_t &            n_img_tokens,
                     int64_t              hidden_pl,
                     const float *        attn_data,
                     int                  n_patches,
                     int                  n_heads,
                     int                  target_tokens,
                     float                important_ratio,
                     bool                 keep_order);

}  // namespace pi05
