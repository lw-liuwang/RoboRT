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

#include "pi05_vispruner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace pi05 {

void vispruner_prune(std::vector<float> & img_emb,
                     int64_t &            n_img_tokens,
                     int64_t              hidden_pl,
                     const float *        attn_data,
                     int                  n_patches,
                     int                  n_heads,
                     int                  target_tokens,
                     float                important_ratio,
                     bool                 keep_order) {
    if (target_tokens <= 0 || target_tokens >= static_cast<int>(n_img_tokens)) {
        return;
    }
    if (!attn_data || n_patches != static_cast<int>(n_img_tokens)) {
        std::fprintf(stderr, "pi05: VisPruner: invalid attention data (patches=%d tokens=%lld)\n", n_patches,
                     static_cast<long long>(n_img_tokens));
        return;
    }

    const int N     = n_patches;
    const int T     = target_tokens;
    const int T_imp = static_cast<int>(T * important_ratio);
    const int T_div = T - T_imp;

    if (T_imp < 1 || T_div < 0) {
        std::fprintf(stderr, "pi05: VisPruner: bad T=%d imp_ratio=%f (imp=%d div=%d)\n", T, important_ratio, T_imp,
                     T_div);
        return;
    }

    // ── Stage 1: importance score = mean attention received over all heads and queries ──
    // attn_data layout: [n_heads, N, N] row-major — attn[h*N*N + q*N + p].
    std::vector<float> importance(N, 0.f);
    for (int h = 0; h < n_heads; ++h) {
        for (int q = 0; q < N; ++q) {
            for (int p = 0; p < N; ++p) {
                importance[p] += attn_data[static_cast<size_t>(h) * N * N + static_cast<size_t>(q) * N + p];
            }
        }
    }
    const float inv = 1.f / static_cast<float>(n_heads * N);
    for (int p = 0; p < N; ++p) {
        importance[p] *= inv;
    }

    // ── Sort tokens by importance (descending) ──
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return importance[a] > importance[b]; });

    // ── Stage 1 result: take top T_imp tokens ──
    std::vector<int> selected;
    selected.reserve(T);
    std::vector<bool> used(N, false);
    for (int i = 0; i < T_imp; ++i) {
        selected.push_back(order[i]);
        used[order[i]] = true;
    }

    // ── Stage 2: diversity selection via iterative pair-matching ──
    // Remaining tokens (by importance rank, low to high):
    std::vector<int> remaining;
    remaining.reserve(N - T_imp);
    for (int i = T_imp; i < N; ++i) {
        remaining.push_back(order[i]);
    }

    // Normalise all embeddings once for cosine similarity.
    std::vector<float> emb_norm(static_cast<size_t>(N) * hidden_pl);
    for (int i = 0; i < N; ++i) {
        float *       dst = emb_norm.data() + static_cast<size_t>(i) * hidden_pl;
        const float * src = img_emb.data() + static_cast<size_t>(i) * hidden_pl;
        float         sq  = 0.f;
        for (int64_t j = 0; j < hidden_pl; ++j) {
            sq += src[j] * src[j];
        }
        const float inv_n = (sq > 1e-10f) ? (1.f / std::sqrt(sq)) : 0.f;
        for (int64_t j = 0; j < hidden_pl; ++j) {
            dst[j] = src[j] * inv_n;
        }
    }

    // Each round: pair adjacent tokens, measure cosine similarity, drop the
    // `r` most similar pairs.  Repeat until the diversity budget is met.
    int n_remaining = static_cast<int>(remaining.size());
    while (n_remaining > T_div) {
        const int R            = n_remaining;
        const int need_to_drop = R - T_div;
        if (need_to_drop <= 0) {
            break;
        }

        const int n_pairs = R / 2;
        // Ceiling division: one pair covers 2 tokens, so we need
        // ceil(need_to_drop / 2) pairs.  Capped at n_pairs.
        const int r       = std::max(1, std::min((need_to_drop + 1) / 2, n_pairs));

        std::vector<float> pair_sim(n_pairs, -2.f);
        for (int i = 0; i < n_pairs; ++i) {
            const float * a   = emb_norm.data() + static_cast<size_t>(remaining[i * 2]) * hidden_pl;
            const float * b   = emb_norm.data() + static_cast<size_t>(remaining[i * 2 + 1]) * hidden_pl;
            float         dot = 0.f;
            for (int64_t j = 0; j < hidden_pl; ++j) {
                dot += a[j] * b[j];
            }
            pair_sim[i] = std::max(-1.f, std::min(1.f, dot));
        }

        // Sort pairs by similarity descending; drop the most similar r pairs.
        std::vector<int> pair_order(n_pairs);
        for (int i = 0; i < n_pairs; ++i) {
            pair_order[i] = i;
        }
        std::sort(pair_order.begin(), pair_order.end(), [&](int a, int b) { return pair_sim[a] > pair_sim[b]; });

        std::vector<int> new_remaining;
        new_remaining.reserve(R - r * 2);
        for (int pi = r; pi < n_pairs; ++pi) {
            const int i = pair_order[pi];
            new_remaining.push_back(remaining[i * 2]);
            new_remaining.push_back(remaining[i * 2 + 1]);
        }
        if (R % 2 != 0) {  // odd tail token always survives
            new_remaining.push_back(remaining[R - 1]);
        }

        remaining   = std::move(new_remaining);
        n_remaining = static_cast<int>(remaining.size());
    }

    for (int i = 0; i < n_remaining; ++i) {
        selected.push_back(remaining[i]);
    }

    // ── Reorder and compact the embedding buffer ──
    const int final_count = static_cast<int>(selected.size());
    if (keep_order) {
        std::sort(selected.begin(), selected.end());
    }

    std::vector<float> pruned(static_cast<size_t>(final_count) * hidden_pl);
    for (int i = 0; i < final_count; ++i) {
        std::memcpy(pruned.data() + static_cast<size_t>(i) * hidden_pl,
                    img_emb.data() + static_cast<size_t>(selected[i]) * hidden_pl,
                    static_cast<size_t>(hidden_pl) * sizeof(float));
    }

    img_emb      = std::move(pruned);
    n_img_tokens = final_count;
    std::printf("pi05: VisPruner: pruned %d -> %d tokens (imp=%d div=%d, ratio=%.2f)\n", N, final_count, T_imp,
                n_remaining, important_ratio);
}

}  // namespace pi05
