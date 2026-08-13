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

#include "clip.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model.h"
#ifdef GGML_USE_CUDA
#    include "ggml-cuda.h"
#endif
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace vla {

namespace {

struct gguf_reader {
    gguf_context * gctx     = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE *         fp       = nullptr;
    size_t         data_off = 0;

    bool open(const std::string & path) {
        gguf_init_params p{};
        p.no_alloc = true;
        p.ctx      = &meta_ctx;
        gctx       = gguf_init_from_file(path.c_str(), p);
        if (!gctx) {
            std::fprintf(stderr, "vla(pi05): gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla(pi05): fopen failed for %s\n", path.c_str());
            return false;
        }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }

    ~gguf_reader() {
        if (fp) {
            std::fclose(fp);
        }
        if (gctx) {
            gguf_free(gctx);
        }
        if (meta_ctx) {
            ggml_free(meta_ctx);
        }
    }

    gguf_reader()                                = default;
    gguf_reader(const gguf_reader &)             = delete;
    gguf_reader & operator=(const gguf_reader &) = delete;

    bool has_key(const char * k) const { return gguf_find_key(gctx, k) >= 0; }

    uint32_t u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }

    float f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }

    double f64(const char * k) const { return gguf_get_val_f64(gctx, gguf_find_key(gctx, k)); }

    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) {
            std::fprintf(stderr, "vla(pi05): missing tensor %s\n", name);
            return false;
        }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) {
            return false;
        }
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target, bool gemma_norm) {
        const ggml_tensor * t = meta(name);
        if (!t) {
            std::fprintf(stderr, "vla(pi05): missing tensor %s\n", name);
            return {};
        }
        const int64_t n = ggml_nelements(t);

        std::vector<float> f32(n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32.data())) {
                return {};
            }
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(n);
            if (!read_raw(name, tmp.data())) {
                return {};
            }
            ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
        } else {
            std::fprintf(stderr, "vla(pi05): tensor %s has unsupported type %d\n", name, (int) t->type);
            return {};
        }
        if (gemma_norm) {
            for (int64_t i = 0; i < n; ++i) {
                f32[i] += 1.0f;
            }
        }

        if (target == GGML_TYPE_F32) {
            std::vector<uint8_t> out(n * sizeof(float));
            std::memcpy(out.data(), f32.data(), out.size());
            return out;
        }
        if (target == GGML_TYPE_BF16) {
            std::vector<uint8_t> out(n * sizeof(ggml_bf16_t));
            ggml_fp32_to_bf16_row(f32.data(), reinterpret_cast<ggml_bf16_t *>(out.data()), n);
            return out;
        }
        if (ggml_is_quantized(target)) {
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = n / n_per_row;
            const int64_t blck      = ggml_blck_size(target);
            if (n_per_row % blck != 0) {
                std::fprintf(stderr, "vla(pi05): cannot quantize %s to %s: ne0=%lld not divisible by block=%lld\n",
                             name, ggml_type_name(target), (long long) n_per_row, (long long) blck);
                return {};
            }
            const size_t         qbytes = (size_t) nrows * (size_t) (n_per_row / blck) * ggml_type_size(target);
            std::vector<uint8_t> out(qbytes);
            const size_t written = ggml_quantize_chunk(target, f32.data(), out.data(), 0, nrows, n_per_row, nullptr);
            if (written != qbytes) {
                std::fprintf(stderr, "vla(pi05): quantized byte mismatch for %s (%zu vs %zu)\n", name, written, qbytes);
                return {};
            }
            return out;
        }
        std::fprintf(stderr, "vla(pi05): unsupported resident type %d for %s\n", (int) target, name);
        return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids, float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t) {
            std::fprintf(stderr, "vla(pi05): missing tensor %s\n", name);
            return false;
        }
        if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "vla(pi05): %s shape unfit for row-fetch\n", name);
            return false;
        }
        const int64_t        rows = t->ne[1];
        const int64_t        id   = gguf_find_tensor(gctx, name);
        const size_t         base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t         elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t         rb   = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) {
                std::fprintf(stderr, "vla(pi05): row %d out of range for %s\n", r, name);
                return false;
            }
            if (std::fseek(fp, (long) (base + (size_t) r * rb), SEEK_SET) != 0) {
                return false;
            }
            if (std::fread(row.data(), 1, rb, fp) != rb) {
                return false;
            }
            if (elsz == 4) {
                std::memcpy(dst + k * cols, row.data(), rb);
            } else {
                ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
            }
        }
        return true;
    }
};

struct GemmaLayerW {
    ggml_tensor * ln_in   = nullptr;
    ggml_tensor * Wq      = nullptr;
    ggml_tensor * Wk      = nullptr;
    ggml_tensor * Wv      = nullptr;
    ggml_tensor * Wo      = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * Wgate   = nullptr;
    ggml_tensor * Wup     = nullptr;
    ggml_tensor * Wdown   = nullptr;
};

struct AdaGemmaLayerW {
    ggml_tensor * ln_in_W   = nullptr;
    ggml_tensor * ln_in_b   = nullptr;
    ggml_tensor * Wq        = nullptr;
    ggml_tensor * Wk        = nullptr;
    ggml_tensor * Wv        = nullptr;
    ggml_tensor * Wo        = nullptr;
    ggml_tensor * ln_post_W = nullptr;
    ggml_tensor * ln_post_b = nullptr;
    ggml_tensor * Wgate     = nullptr;
    ggml_tensor * Wup       = nullptr;
    ggml_tensor * Wdown     = nullptr;
};

bool is_gemma_norm(const std::string & name) {
    return name.find("norm.weight") != std::string::npos;
}

std::vector<float> sinusoidal_time_emb(double t, int64_t dim, double min_p, double max_p) {
    const int64_t      half = dim / 2;
    std::vector<float> out(dim);
    for (int64_t i = 0; i < half; ++i) {
        const double frac   = (half == 1) ? 0.0 : double(i) / double(half - 1);
        const double period = min_p * std::pow(max_p / min_p, frac);
        const double s      = (2.0 * M_PI / period) * t;
        out[i]              = (float) std::sin(s);
        out[half + i]       = (float) std::cos(s);
    }
    return out;
}

// VisPruner: two-stage visual token pruning (importance + diversity)
// Returns the pruned embedding vector and updates n_img_tokens.
static void vispruner_prune(
    std::vector<float> & img_emb,          // [in/out] (N * hidden_pl) row-major
    int64_t &            n_img_tokens,     // [in/out] N
    int64_t              hidden_pl,        // embedding dimension
    const float *        attn_data,        // [N*N*n_heads] attention weights
    int                  n_patches,        // N
    int                  n_heads,          // H
    int                  target_tokens,    // T: target token count after pruning
    float                important_ratio,  // r: ratio of importance-based tokens
    bool                 keep_order)  // true: keep original spatial order (FastV-style), false: reorder by importance
{
    if (target_tokens <= 0 || target_tokens >= (int) n_img_tokens) {
        return;  // disabled or no reduction needed
    }
    if (!attn_data || n_patches != (int) n_img_tokens) {
        std::fprintf(stderr, "vla(pi05): VisPruner: invalid attention data (patches=%d tokens=%lld)\n", n_patches,
                     (long long) n_img_tokens);
        return;
    }

    const int N     = n_patches;
    const int T     = target_tokens;
    const int T_imp = (int) (T * important_ratio);  // importance-selected count
    const int T_div = T - T_imp;                    // diversity-selected count
    if (T_imp < 1 || T_div < 0) {
        std::fprintf(stderr, "vla(pi05): VisPruner: bad T=%d imp_ratio=%f (imp=%d div=%d)\n", T, important_ratio, T_imp,
                     T_div);
        return;
    }

    // ---- Step 1: compute per-token importance ----
    // attn_data layout: (N, N, H) row-major => [h][q][p]
    // importance[p] = mean over q then mean over h
    std::vector<float> importance(N, 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        for (int q = 0; q < N; ++q) {
            for (int p = 0; p < N; ++p) {
                importance[p] += attn_data[(size_t) h * N * N + (size_t) q * N + (size_t) p];
            }
        }
    }
    const float inv = 1.0f / (float) (n_heads * N);
    for (int p = 0; p < N; ++p) {
        importance[p] *= inv;
    }

    // ---- Step 2: sort by importance (descending) ----
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return importance[a] > importance[b]; });

    // ---- Step 3: Stage 1 - importance selection ----
    std::vector<int> selected;
    selected.reserve(T);
    std::vector<bool> used(N, false);
    for (int i = 0; i < T_imp; ++i) {
        int idx = order[i];
        selected.push_back(idx);
        used[idx] = true;
    }

    // ---- Step 4: Stage 2 - diversity via iterative pair-matching ----
    // Collect remaining (unused) indices
    std::vector<int> remaining;
    remaining.reserve(N - T_imp);
    for (int i = T_imp; i < N; ++i) {
        remaining.push_back(order[i]);
    }

    // Normalize embeddings for cosine similarity
    std::vector<float> emb_norm(N * hidden_pl);
    for (int i = 0; i < N; ++i) {
        float *       dst = emb_norm.data() + (size_t) i * hidden_pl;
        const float * src = img_emb.data() + (size_t) i * hidden_pl;
        float         sq  = 0.0f;
        for (int64_t j = 0; j < hidden_pl; ++j) {
            sq += src[j] * src[j];
        }
        float inv_norm = (sq > 1e-10f) ? (1.0f / std::sqrt(sq)) : 0.0f;
        for (int64_t j = 0; j < hidden_pl; ++j) {
            dst[j] = src[j] * inv_norm;
        }
    }

    // Iterative pair-matching: pair even/odd tokens, compute similarity, drop most similar
    int n_remaining = (int) remaining.size();
    while (n_remaining > T_div) {
        const int R            = n_remaining;
        const int need_to_drop = R - T_div;  // tokens that still need to be dropped
        if (need_to_drop <= 0) {
            break;
        }

        // Pair tokens: [0,1], [2,3], ..., [2*(R/2)-1]
        const int          n_pairs = (R / 2);
        // r = number of most-similar PAIRS to drop; each pair carries 2 tokens,
        // so we need ceil(need_to_drop/2) pairs (never more than n_pairs).
        // (Previously r was set to need_to_drop directly, which over-dropped
        //  by 2x whenever need_to_drop >= R/2, wiping out the diversity set.)
        const int          r       = std::max(1, std::min((need_to_drop + 1) / 2, n_pairs));
        std::vector<float> pair_sim(n_pairs, -2.0f);

        for (int i = 0; i < n_pairs; ++i) {
            const int     a_idx = remaining[i * 2];
            const int     b_idx = remaining[i * 2 + 1];
            const float * a     = emb_norm.data() + (size_t) a_idx * hidden_pl;
            const float * b     = emb_norm.data() + (size_t) b_idx * hidden_pl;
            float         dot   = 0.0f;
            for (int64_t j = 0; j < hidden_pl; ++j) {
                dot += a[j] * b[j];
            }
            // Clamp to [-1, 1] for numerical safety
            pair_sim[i] = std::max(-1.0f, std::min(1.0f, dot));
        }

        // Sort pairs by similarity descending, keep the least similar pairs
        std::vector<int> pair_order(n_pairs);
        for (int i = 0; i < n_pairs; ++i) {
            pair_order[i] = i;
        }
        std::sort(pair_order.begin(), pair_order.end(), [&](int a, int b) { return pair_sim[a] > pair_sim[b]; });

        // Drop the most similar r pairs (keep the rest)
        std::vector<int> new_remaining;
        new_remaining.reserve(R - r * 2);
        for (int pi = r; pi < n_pairs; ++pi) {
            int i = pair_order[pi];
            new_remaining.push_back(remaining[i * 2]);
            new_remaining.push_back(remaining[i * 2 + 1]);
        }
        // If odd count, the last unpaired token survives
        if (R % 2 != 0) {
            new_remaining.push_back(remaining[R - 1]);
        }

        remaining   = std::move(new_remaining);
        n_remaining = (int) remaining.size();
    }

    // Append diversity-selected tokens
    for (int i = 0; i < n_remaining; ++i) {
        selected.push_back(remaining[i]);
    }

    // ---- Step 5: reorder embeddings ----
    const int final_count = (int) selected.size();
    // FastV-style: keep the original spatial/token order (drop only), which
    // preserves the positional layout of the image; importance reordering
    // scrambles the grid and harms the policy on manipulation tasks.
    if (keep_order) {
        std::sort(selected.begin(), selected.end());
    }
    std::vector<float> pruned((size_t) final_count * hidden_pl);
    for (int i = 0; i < final_count; ++i) {
        std::memcpy(pruned.data() + (size_t) i * hidden_pl, img_emb.data() + (size_t) selected[i] * hidden_pl,
                    (size_t) hidden_pl * sizeof(float));
    }

    img_emb      = std::move(pruned);
    n_img_tokens = final_count;
    std::printf("vla(pi05): VisPruner: pruned %d -> %d tokens (imp=%d div=%d, ratio=%.2f)\n", N, final_count, T_imp,
                n_remaining, important_ratio);
}

static bool ends_with(const std::string & s, const char * sfx) {
    const size_t n = std::strlen(sfx);
    return s.size() >= n && s.compare(s.size() - n, n, sfx) == 0;
}

}  // namespace

struct Pi05Model final : public Model {
    ~Pi05Model() override;

    std::vector<float> predict(const Inputs & in);
    PreparedInput      prepare_inputs(const Inputs & in);
    std::vector<float> compute_actions(const PreparedInput & p);

    Stats  stats{};  ///< Phase timings of the most recent predict.
    Config cfg{};    ///< Resolved model hyper-parameters.
    std::mutex stats_mu_;  ///< Guards @ref stats (async server: two threads).

    clip_ctx *            cctx        = nullptr;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    bool                  flash_attn_ = false;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    std::string           ckpt_path_;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    std::vector<GemmaLayerW> pl_layers;

    // pi0.5 conditions every expert RMSNorm on the per-step time embedding.
    std::vector<AdaGemmaLayerW> ex_layers;
    ggml_tensor *               ex_final_norm_W = nullptr, *ex_final_norm_b = nullptr;

    ggml_tensor *W_ain = nullptr, *b_ain = nullptr;
    ggml_tensor *W_tm1 = nullptr, *b_tm1 = nullptr;
    ggml_tensor *W_tm2 = nullptr, *b_tm2 = nullptr;
    ggml_tensor *W_aout = nullptr, *b_aout = nullptr;

    // Cached compute graph: rebuilt only when the input token counts change.
    // The server is single-threaded (REQ/REP), so a single cached graph is safe.
    ggml_context *             comp_ctx_     = nullptr;
    ggml_cgraph *              gf_           = nullptr;
    ggml_gallocr_t             galloc_       = nullptr;
    ggml_tensor *              t_image_emb_  = nullptr;
    ggml_tensor *              t_lang_emb_   = nullptr;
    ggml_tensor *              t_prefix_pos_ = nullptr;
    ggml_tensor *              t_x0_         = nullptr;
    ggml_tensor *              t_suffix_pos_ = nullptr;
    ggml_tensor *              t_full_mask_  = nullptr;
    std::vector<ggml_tensor *> t_time_;
    ggml_tensor *              x_final_        = nullptr;
    int64_t                    c_n_img_tokens_ = -1;
    int64_t                    c_n_lang_       = -1;
    int64_t                    c_n_suf_        = -1;

    // RTC (Real-Time Chunking) cached-graph inputs: prev_chunk prefix guidance
    // and the per-step schedule weights. Both are uploaded every request.
    ggml_tensor * t_prev_chunk_  = nullptr;   // [max_ad, chunk] latent leftover
    ggml_tensor * t_rtc_weights_ = nullptr;   // [max_ad, chunk] schedule weights
    int64_t c_rtc_         = 0;               // rtc_enabled_ at graph-build time
    int64_t c_rtc_horizon_ = 0;               // rtc_horizon_ at graph-build time

    // RTC configuration (read from env once; default off = bit-exact baseline).
    bool rtc_initialized_ = false;
    bool rtc_enabled_     = false;
    int  rtc_horizon_     = 10;               // execution horizon (VLA_PI05_RTC_HORIZON)
    int  rtc_delay_       = 0;                // inference delay prefix (VLA_PI05_RTC_DELAY)
    float rtc_max_guidance_ = 10.0f;          // max guidance weight (VLA_PI05_RTC_GUIDANCE)
    std::string rtc_schedule_ = "linear";     // linear / ones / exp (VLA_PI05_RTC_SCHEDULE)
    bool rtc_x0_inpaint_  = false;            // inject leftover into x0 (VLA_PI05_RTC_X0_INPAINT)
    std::vector<float> rtc_step_guidance_;    // per-step guidance weight [num_steps]

    void init_rtc_config() {
        if (rtc_initialized_) return;
        rtc_initialized_ = true;
        if (const char * e = std::getenv("VLA_PI05_RTC"); e && std::atoi(e) > 0) rtc_enabled_ = true;
        if (const char * e = std::getenv("VLA_PI05_RTC_HORIZON");  e && std::atoi(e) >= 1)  rtc_horizon_  = std::atoi(e);
        if (const char * e = std::getenv("VLA_PI05_RTC_DELAY");    e && std::atoi(e) >= 0)  rtc_delay_    = std::atoi(e);
        if (const char * e = std::getenv("VLA_PI05_RTC_GUIDANCE"); e && std::atof(e) > 0.f) rtc_max_guidance_ = (float) std::atof(e);
        if (const char * e = std::getenv("VLA_PI05_RTC_SCHEDULE"); e && e[0])               rtc_schedule_ = e;
        if (const char * e = std::getenv("VLA_PI05_RTC_X0_INPAINT"); e && std::atoi(e) > 0) rtc_x0_inpaint_ = true;
        if (rtc_enabled_) {
            // Per-step guidance weight (LeRobot modeling_rtc.denoise_step):
            //   tau = 1 - time; c = time/tau;
            //   inv_r2 = (time^2 + tau^2) / time^2;
            //   guidance = min(c*inv_r2, max);  time = 1 -> 1/N (tau = s/N).
            const int num_steps = (int) cfg.num_steps;
            const float max_g = rtc_max_guidance_;
            rtc_step_guidance_.assign(num_steps, max_g);
            for (int s = 0; s < num_steps; ++s) {
                const float time = 1.0f - (float) s / (float) num_steps;
                const float tau  = (float) s / (float) num_steps;
                if (tau > 0.f) {
                    const float c      = time / tau;
                    const float inv_r2 = (time * time + tau * tau) / (time * time);
                    float g = c * inv_r2;
                    if (!(g > 0.f) || g > max_g) g = max_g; // NaN/Inf -> clamp to max
                    rtc_step_guidance_[s] = g;
                }
            }
            std::printf("vla(pi05): RTC enabled: horizon=%d delay=%d guidance=%.1f schedule=%s inpaint=%d steps=%d\n",
                        rtc_horizon_, rtc_delay_, rtc_max_guidance_, rtc_schedule_.c_str(),
                        rtc_x0_inpaint_ ? 1 : 0, num_steps);
        }
    }

    // State stats are kept for the Python client prompt path; C++ inference
    // itself receives already-tokenized text and therefore only unnormalizes actions.
    std::string        state_norm_mode  = "QUANTILES";
    std::string        action_norm_mode = "QUANTILES";
    std::vector<float> state_mean, state_std, action_mean, action_std;
    std::vector<float> state_q01, state_q99, action_q01, action_q99;

    // Noise RNG for the action expert. Random by default (entropy-seeded);
    // VLA_PI05_SEED=N makes the noise sequence reproducible so two servers
    // (e.g. baseline vs optimized) fed the identical request stream can be
    // compared action-for-action / episode-for-episode.
    std::mt19937 rng{ []() -> std::mt19937::result_type {
        const char * e = std::getenv("VLA_PI05_SEED");
        return e ? (std::mt19937::result_type) std::strtoul(e, nullptr, 10) : std::random_device{}();
    }() };
    int n_threads = 4;
};

namespace {

ggml_tensor * build_gemma_layer(ggml_context *      ctx,
                                const GemmaLayerW & w,
                                ggml_tensor *       x_in,
                                ggml_tensor *       positions,
                                const Config &      cfg,
                                int64_t             seq,
                                float               rope_base,
                                ggml_tensor *       cached_K,
                                ggml_tensor *       cached_V,
                                ggml_tensor *       mask,
                                bool                use_flash,
                                ggml_tensor **      k_out,
                                ggml_tensor **      v_out) {
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t qf  = nq * hd;

    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.ln_in);

    // q/k/v all read the same x_norm: convert F32->BF16 once and share it.
    // Bit-exact: ggml_cpy and ggml-cuda's internal src1 convert both use the
    // same deterministic __float2bfloat16 RN rounding, so the GEMM inputs are
    // bit-identical and only the redundant converts are removed.
    ggml_tensor * x_norm_bf16 = ggml_cast(ctx, x_norm, GGML_TYPE_BF16);
    ggml_tensor * q           = ggml_mul_mat(ctx, w.Wq, x_norm_bf16);
    ggml_tensor * k           = ggml_mul_mat(ctx, w.Wk, x_norm_bf16);
    ggml_tensor * v           = ggml_mul_mat(ctx, w.Wv, x_norm_bf16);

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq, seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    auto rope_call = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.f, 0.f, 1.f,
                             32.f, 1.f);
    };
    ggml_tensor * q_rope = rope_call(q_h);
    ggml_tensor * k_rope = rope_call(k_h);

    if (k_out) {
        *k_out = k_rope;
    }
    if (v_out) {
        *v_out = v_h;
    }

    ggml_tensor * K_full = k_rope;
    ggml_tensor * V_full = v_h;
    if (cached_K && cached_V) {
        K_full = ggml_concat(ctx, cached_K, k_rope, 2);
        V_full = ggml_concat(ctx, cached_V, v_h, 2);
    }

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));

    const float   scale   = 1.f / std::sqrt((float) hd);
    ggml_tensor * kqv     = nullptr;
    ggml_tensor * att_pre = nullptr;
    if (use_flash) {
        // Flash attention mirrors llama.cpp's own Gemma path: Q stays F32 (the
        // fused kernels require it) while K/V are cast to fp16 tensor cores.
        // The all-zeros pi0.5 suffix mask (= no masking) is passed as null.
        // Its "permuted" result is (head_dim, n_heads, n_tokens), so no extra
        // permute is needed. Note: the manual path's V is transposed
        // (n_kv, kv_tokens, head_dim) for ggml_mul_mat; flash expects
        // (head_dim, kv_tokens, n_kv) like K.
        ggml_tensor * Vf = ggml_cont(ctx, ggml_permute(ctx, V_full, 0, 2, 1, 3));
        ggml_tensor * Kh = ggml_cast(ctx, K, GGML_TYPE_F16);
        ggml_tensor * Vh = ggml_cast(ctx, Vf, GGML_TYPE_F16);
        kqv              = ggml_flash_attn_ext(ctx, Q, Kh, Vh, nullptr, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
        att_pre = ggml_reshape_2d(ctx, ggml_cont(ctx, kqv), qf, seq);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale, 0.f);
        kqv                = ggml_mul_mat(ctx, V, attn);
        att_pre            = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq);
    }
    ggml_tensor * o_out = ggml_mul_mat(ctx, w.Wo, att_pre);
    ggml_tensor * h1    = ggml_add(ctx, x_in, o_out);

    ggml_tensor * x_norm_mlp      = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
    ggml_tensor * x_norm_mlp_bf16 = ggml_cast(ctx, x_norm_mlp, GGML_TYPE_BF16);
    ggml_tensor * gate            = ggml_mul_mat(ctx, w.Wgate, x_norm_mlp_bf16);
    ggml_tensor * up              = ggml_mul_mat(ctx, w.Wup, x_norm_mlp_bf16);
    ggml_tensor * inter_t         = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
    ggml_tensor * mlp_out         = ggml_mul_mat(ctx, w.Wdown, inter_t);
    return ggml_add(ctx, h1, mlp_out);
}

ggml_tensor * adarms_norm(ggml_context * ctx,
                          ggml_tensor *  x,
                          ggml_tensor *  dense_w,
                          ggml_tensor *  dense_b,
                          ggml_tensor *  cond,
                          const Config & cfg,
                          ggml_tensor ** gate_out) {
    // AdaRMSNorm mirrors OpenPI: RMSNorm(x) * (1 + scale(cond)) + shift(cond),
    // and the third projection slice gates the following residual branch.
    ggml_tensor * mod       = ggml_add(ctx, ggml_mul_mat(ctx, dense_w, cond), dense_b);
    const size_t  row_bytes = (size_t) cfg.expert_h * sizeof(float);
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
    // OpenPI uses x + y * gate when AdaRMS produces a gate, and plain x + y otherwise.
    return gate ? ggml_add(ctx, x, ggml_mul(ctx, y, gate)) : ggml_add(ctx, x, y);
}

ggml_tensor * build_adarms_gemma_layer(ggml_context *         ctx,
                                       const AdaGemmaLayerW & w,
                                       ggml_tensor *          x_in,
                                       ggml_tensor *          positions,
                                       ggml_tensor *          adarms_cond,
                                       const Config &         cfg,
                                       int64_t                seq,
                                       float                  rope_base,
                                       ggml_tensor *          cached_K,
                                       ggml_tensor *          cached_V,
                                       ggml_tensor *          mask,
                                       bool                   use_flash) {
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t qf  = nq * hd;

    ggml_tensor * attn_gate = nullptr;
    ggml_tensor * x_norm    = adarms_norm(ctx, x_in, w.ln_in_W, w.ln_in_b, adarms_cond, cfg, &attn_gate);

    // share one F32->BF16 convert across q/k/v (bit-exact, see build_gemma_layer)
    ggml_tensor * x_norm_bf16 = ggml_cast(ctx, x_norm, GGML_TYPE_BF16);
    ggml_tensor * q           = ggml_mul_mat(ctx, w.Wq, x_norm_bf16);
    ggml_tensor * k           = ggml_mul_mat(ctx, w.Wk, x_norm_bf16);
    ggml_tensor * v           = ggml_mul_mat(ctx, w.Wv, x_norm_bf16);

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq, seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    auto rope_call = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.f, 0.f, 1.f,
                             32.f, 1.f);
    };
    ggml_tensor * q_rope = rope_call(q_h);
    ggml_tensor * k_rope = rope_call(k_h);

    ggml_tensor * K_full = k_rope;
    ggml_tensor * V_full = v_h;
    if (cached_K && cached_V) {
        K_full = ggml_concat(ctx, cached_K, k_rope, 2);
        V_full = ggml_concat(ctx, cached_V, v_h, 2);
    }

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));

    const float   scale   = 1.f / std::sqrt((float) hd);
    ggml_tensor * kqv     = nullptr;
    ggml_tensor * att_pre = nullptr;
    if (use_flash) {
        // Flash attention mirrors llama.cpp's own Gemma path: Q stays F32 (the
        // fused kernels require it) while K/V are cast to fp16 tensor cores.
        // The all-zeros pi0.5 suffix mask (= no masking) is passed as null.
        // Its "permuted" result is (head_dim, n_heads, n_tokens), so no extra
        // permute is needed. Note: the manual path's V is transposed
        // (n_kv, kv_tokens, head_dim) for ggml_mul_mat; flash expects
        // (head_dim, kv_tokens, n_kv) like K.
        ggml_tensor * Vf = ggml_cont(ctx, ggml_permute(ctx, V_full, 0, 2, 1, 3));
        ggml_tensor * Kh = ggml_cast(ctx, K, GGML_TYPE_F16);
        ggml_tensor * Vh = ggml_cast(ctx, Vf, GGML_TYPE_F16);
        kqv              = ggml_flash_attn_ext(ctx, Q, Kh, Vh, nullptr, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
        att_pre = ggml_reshape_2d(ctx, ggml_cont(ctx, kqv), qf, seq);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale, 0.f);
        kqv                = ggml_mul_mat(ctx, V, attn);
        att_pre            = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq);
    }
    ggml_tensor * o_out = ggml_mul_mat(ctx, w.Wo, att_pre);
    ggml_tensor * h1    = gated_residual(ctx, x_in, o_out, attn_gate);

    ggml_tensor * mlp_gate        = nullptr;
    ggml_tensor * x_norm_mlp      = adarms_norm(ctx, h1, w.ln_post_W, w.ln_post_b, adarms_cond, cfg, &mlp_gate);
    // share one F32->BF16 convert across gate/up (bit-exact, see build_gemma_layer)
    ggml_tensor * x_norm_mlp_bf16 = ggml_cast(ctx, x_norm_mlp, GGML_TYPE_BF16);
    ggml_tensor * gate            = ggml_mul_mat(ctx, w.Wgate, x_norm_mlp_bf16);
    ggml_tensor * up              = ggml_mul_mat(ctx, w.Wup, x_norm_mlp_bf16);
    ggml_tensor * inter_t         = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
    ggml_tensor * mlp_out         = ggml_mul_mat(ctx, w.Wdown, inter_t);
    return gated_residual(ctx, h1, mlp_out, mlp_gate);
}

ggml_tensor * build_embed_suffix(ggml_context *    ctx,
                                 const Pi05Model & m,
                                 ggml_tensor *     x,
                                 ggml_tensor *     time_vec,
                                 ggml_tensor **    adarms_cond) {
    // pi0.5 feeds only action tokens to the expert; state is represented in
    // the language prompt as discretized text, matching LeRobot/OpenPI.
    ggml_tensor * action_emb = ggml_add(ctx, ggml_mul_mat(ctx, m.W_ain, x), m.b_ain);
    ggml_tensor * t1         = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, m.W_tm1, time_vec), m.b_tm1));
    ggml_tensor * t2         = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, m.W_tm2, t1), m.b_tm2));
    if (adarms_cond) {
        *adarms_cond = t2;
    }
    return action_emb;
}

bool load_config(const gguf_reader & g, Config & cfg) {
    auto need = [&](const char * k) {
        if (!g.has_key(k)) {
            std::fprintf(stderr, "vla(pi05): gguf missing key %s\n", k);
            return false;
        }
        return true;
    };
    for (const char * k : { "pi05.hidden", "pi05.intermediate", "pi05.n_q_heads", "pi05.n_kv_heads", "pi05.head_dim",
                            "pi05.n_layers", "pi05.expert_h", "pi05.expert_inter", "pi05.chunk_size", "pi05.num_steps",
                            "pi05.max_state_dim", "pi05.max_action_dim", "pi05.real_state_dim", "pi05.real_action_dim",
                            "pi05.tokenizer_max_length", "pi05.min_period", "pi05.max_period" }) {
        if (!need(k)) {
            return false;
        }
    }
    cfg              = Config{};
    cfg.hidden       = g.u32("pi05.hidden");
    cfg.intermediate = g.u32("pi05.intermediate");
    cfg.n_q_heads    = g.u32("pi05.n_q_heads");
    cfg.n_kv_heads   = g.u32("pi05.n_kv_heads");
    cfg.head_dim     = g.u32("pi05.head_dim");
    cfg.n_layers     = g.u32("pi05.n_layers");
    cfg.expert_h     = g.u32("pi05.expert_h");
    cfg.expert_inter = g.u32("pi05.expert_inter");
    cfg.n_suffix     = g.u32("pi05.chunk_size");
    cfg.num_steps    = g.u32("pi05.num_steps");
    if (const char * ns = std::getenv("VLA_PI05_NUM_STEPS"); ns && std::atoi(ns) >= 1 && std::atoi(ns) <= 10) {
        const int old = (int) cfg.num_steps;
        cfg.num_steps = (uint32_t) std::atoi(ns);
        std::printf("vla(pi05): num_steps overridden by env VLA_PI05_NUM_STEPS: %d -> %u\n", old,
                    (unsigned) cfg.num_steps);
    }
    if (const char * cs = std::getenv("VLA_PI05_CHUNK"); cs && std::atoi(cs) >= 1 && std::atoi(cs) <= 50) {
        const int old = (int) cfg.n_suffix;
        cfg.n_suffix  = (uint32_t) std::atoi(cs);
        std::printf("vla(pi05): chunk_size overridden by env VLA_PI05_CHUNK: %d -> %u\n", old, (unsigned) cfg.n_suffix);
    }
    cfg.max_state_dim   = g.u32("pi05.max_state_dim");
    cfg.max_action_dim  = g.u32("pi05.max_action_dim");
    cfg.real_state_dim  = g.u32("pi05.real_state_dim");
    cfg.real_action_dim = g.u32("pi05.real_action_dim");
    cfg.n_lang          = g.u32("pi05.tokenizer_max_length");
    cfg.min_period      = g.f64("pi05.min_period");
    cfg.max_period      = g.f64("pi05.max_period");

    cfg.n_state           = 0;
    cfg.n_img             = 256;
    cfg.q_full_dim        = cfg.n_q_heads * cfg.head_dim;
    cfg.kv_full_dim       = cfg.n_kv_heads * cfg.head_dim;
    cfg.self_attn_every_n = 0;
    cfg.rms_eps           = g.has_key("pi05.rms_norm_eps") ? g.f32("pi05.rms_norm_eps") : 1e-6f;
    cfg.norm_eps          = g.has_key("pi05.norm_eps") ? g.f32("pi05.norm_eps") : 1e-8f;
    cfg.rope_mode         = GGML_ROPE_TYPE_NEOX;
    cfg.rope_n_dims       = (int) cfg.head_dim;
    cfg.rope_freq_base    = g.has_key("pi05.rope_theta") ? (float) g.f64("pi05.rope_theta") : 10000.f;
    cfg.n_prefix          = 0;
    cfg.n_full            = 0;
    return true;
}

bool load_stats(gguf_reader & g, Pi05Model & m) {
    const auto & cfg   = m.cfg;
    m.state_norm_mode  = g.has_key("pi05.state_norm_mode") ? g.str("pi05.state_norm_mode") : "QUANTILES";
    m.action_norm_mode = g.has_key("pi05.action_norm_mode") ? g.str("pi05.action_norm_mode") : "QUANTILES";

    m.state_mean.assign(cfg.real_state_dim, 0.f);
    m.state_std.assign(cfg.real_state_dim, 1.f);
    m.action_mean.assign(cfg.real_action_dim, 0.f);
    m.action_std.assign(cfg.real_action_dim, 1.f);
    m.state_q01.assign(cfg.real_state_dim, -1.f);
    m.state_q99.assign(cfg.real_state_dim, 1.f);
    m.action_q01.assign(cfg.real_action_dim, -1.f);
    m.action_q99.assign(cfg.real_action_dim, 1.f);
    auto read1d = [&](const char * name, std::vector<float> & dst) {
        const ggml_tensor * t = g.meta(name);
        if (!t) {
            std::printf("vla(pi05): %s missing - identity\n", name);
            return;
        }
        if (t->ne[0] != (int64_t) dst.size()) {
            std::printf("vla(pi05): %s dim mismatch - identity\n", name);
            return;
        }
        if (!g.read_raw(name, dst.data())) {
            std::printf("vla(pi05): %s read failed - identity\n", name);
        }
    };
    if (m.state_norm_mode == "MEAN_STD") {
        read1d("state_mean", m.state_mean);
        read1d("state_std", m.state_std);
    } else if (m.state_norm_mode == "QUANTILES") {
        read1d("state_q01", m.state_q01);
        read1d("state_q99", m.state_q99);
    } else {
        std::fprintf(stderr, "vla(pi05): unsupported state_norm_mode '%s'\n", m.state_norm_mode.c_str());
        return false;
    }
    if (m.action_norm_mode == "MEAN_STD") {
        read1d("action_mean", m.action_mean);
        read1d("action_std", m.action_std);
    } else if (m.action_norm_mode == "QUANTILES") {
        read1d("action_q01", m.action_q01);
        read1d("action_q99", m.action_q99);
    } else {
        std::fprintf(stderr, "vla(pi05): unsupported action_norm_mode '%s'\n", m.action_norm_mode.c_str());
        return false;
    }
    std::printf("vla(pi05): normalization state=%s action=%s\n", m.state_norm_mode.c_str(), m.action_norm_mode.c_str());
    return true;
}

}  // namespace

Pi05Model::~Pi05Model() {
    if (galloc_) {
        ggml_gallocr_free(galloc_);
    }
    if (comp_ctx_) {
        ggml_free(comp_ctx_);
    }
    if (weight_buf) {
        ggml_backend_buffer_free(weight_buf);
    }
    if (ctx_weights) {
        ggml_free(ctx_weights);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
    if (cctx) {
        clip_free(cctx);
    }
}

Model * model_load(const std::string & mmproj_path, const std::string & ckpt_path, const std::string & config_path) {
    (void) config_path;

    if (!ends_with(ckpt_path, ".gguf")) {
        std::fprintf(stderr,
                     "vla(pi05): ckpt must be a GGUF produced by src/models/pi05/convert_pi05_to_gguf.py "
                     "(got '%s'); direct .safetensors loading for π0.5 is not yet supported\n",
                     ckpt_path.c_str());
        return nullptr;
    }

    auto m         = std::make_unique<Pi05Model>();
    m->ckpt_path_  = ckpt_path;
    m->matmul_type = GGML_TYPE_BF16;
    if (std::getenv("VLA_PI05_F32_WEIGHTS")) {
        m->matmul_type = GGML_TYPE_F32;
    }
    if (const char * q8 = std::getenv("VLA_PI05_WEIGHT_Q8_0"); q8 && std::atoi(q8) > 0) {
        m->matmul_type = GGML_TYPE_Q8_0;
    }
    if (const char * q4 = std::getenv("VLA_PI05_WEIGHT_Q4_0"); q4 && std::atoi(q4) > 0) {
        m->matmul_type = GGML_TYPE_Q4_0;
    }
    if (const char * fa = std::getenv("VLA_PI05_FLASH_ATTN"); fa && std::atoi(fa) > 0) {
        m->flash_attn_ = true;
        std::printf("vla(pi05): fused flash attention enabled for Gemma/action-expert layers\n");
    }

    gguf_reader g;
    if (!g.open(ckpt_path)) {
        return nullptr;
    }
    if (!g.has_key("pi05.architecture") || g.str("pi05.architecture") != "pi05") {
        std::fprintf(stderr, "vla(pi05): '%s' is not a π0.5 GGUF (pi05.architecture missing/wrong)\n",
                     ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, m->cfg)) {
        return nullptr;
    }
    const Config & cfg     = m->cfg;
    const char *   mm_name = m->matmul_type == GGML_TYPE_F32  ? "F32" :
                             m->matmul_type == GGML_TYPE_Q8_0 ? "Q8_0" :
                             m->matmul_type == GGML_TYPE_Q4_0 ? "Q4_0" :
                                                                "BF16";
    std::printf(
        "vla(pi05): hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_layers=%lld "
        "expert_h=%lld expert_inter=%lld chunk=%lld steps=%d real_state=%lld real_action=%lld "
        "matmul_weights=%s\n",
        (long long) cfg.hidden, (long long) cfg.intermediate, (long long) cfg.n_q_heads, (long long) cfg.n_kv_heads,
        (long long) cfg.head_dim, (long long) cfg.n_layers, (long long) cfg.expert_h, (long long) cfg.expert_inter,
        (long long) cfg.n_suffix, cfg.num_steps, (long long) cfg.real_state_dim, (long long) cfg.real_action_dim,
        mm_name);

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) {
        m->is_cuda = true;
        std::printf("vla(pi05): backend = CUDA (device 0)\n");
    } else {
        std::fprintf(stderr, "vla(pi05): ggml_backend_cuda_init failed; falling back to CPU\n");
    }
#endif
    {
        const unsigned hw = std::thread::hardware_concurrency();
        m->n_threads      = (hw == 0) ? 4 : (int) std::min(hw, 8u);
    }
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) {
            std::fprintf(stderr, "vla(pi05): ggml_backend_cpu_init failed\n");
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(pi05): backend = CPU (%d threads)\n", m->n_threads);
    }

    {
        clip_context_params cp = {};
        cp.use_gpu             = m->is_cuda;
        cp.flash_attn_type     = m->is_cuda ? CLIP_FLASH_ATTN_TYPE_AUTO : CLIP_FLASH_ATTN_TYPE_DISABLED;
        // VisPruner: flash attention hides the attention weights, so force it disabled
        if (std::getenv("VLA_PI05_VIS_PRUNER_TOKENS")) {
            cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
            std::printf("vla(pi05): VisPruner enabled, CLIP flash attention forced OFF\n");
        }
        cp.image_min_tokens  = -1;
        cp.image_max_tokens  = -1;
        cp.warmup            = m->is_cuda;
        cp.cb_eval           = nullptr;
        cp.cb_eval_user_data = nullptr;
        clip_init_result r   = clip_init(mmproj_path.c_str(), cp);
        if (!r.ctx_v) {
            std::fprintf(stderr, "vla(pi05): clip_init failed for %s\n", mmproj_path.c_str());
            return nullptr;
        }
        m->cctx           = r.ctx_v;
        const int img_sz  = clip_get_image_size(m->cctx);
        const int mm_embd = clip_n_mmproj_embd(m->cctx);
        if (img_sz != 224 || mm_embd != (int) cfg.hidden) {
            std::fprintf(stderr, "vla(pi05): mmproj mismatch (image_size=%d mmproj_embd=%d; want 224 / %lld)\n", img_sz,
                         mm_embd, (long long) cfg.hidden);
            return nullptr;
        }
    }

    {
        ggml_init_params wp = { (size_t) 16 * 1024 * 1024, nullptr, true };
        m->ctx_weights      = ggml_init(wp);
        if (!m->ctx_weights) {
            std::fprintf(stderr, "vla(pi05): ggml_init(ctx_weights) failed\n");
            return nullptr;
        }
    }
    ggml_context *             W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;

    auto mk = [&](const char * name, ggml_type type, int n_dims, const int64_t * ne) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "vla(pi05): missing tensor %s\n", name);
            return nullptr;
        }
        ggml_tensor * t = ggml_new_tensor(W, type, n_dims, ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };

    auto mk_mm = [&](const char * name) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "vla(pi05): missing tensor %s\n", name);
            return nullptr;
        }
        return mk(name, m->matmul_type, GGML_MAX_DIMS, gt->ne);
    };
    auto mk_f32 = [&](const char * name) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "vla(pi05): missing tensor %s\n", name);
            return nullptr;
        }
        return mk(name, GGML_TYPE_F32, GGML_MAX_DIMS, gt->ne);
    };

    auto load_layer = [&](const char * tower, int i, GemmaLayerW & lw) -> bool {
        char b[256];
        auto suf = [&](const char * s) {
            std::snprintf(b, sizeof(b), "%s.blk.%d.%s", tower, i, s);
            return b;
        };
        lw.ln_in   = mk_f32(suf("attn_norm.weight"));
        lw.Wq      = mk_mm(suf("attn_q.weight"));
        lw.Wk      = mk_mm(suf("attn_k.weight"));
        lw.Wv      = mk_mm(suf("attn_v.weight"));
        lw.Wo      = mk_mm(suf("attn_o.weight"));
        lw.ln_post = mk_f32(suf("ffn_norm.weight"));
        lw.Wgate   = mk_mm(suf("ffn_gate.weight"));
        lw.Wup     = mk_mm(suf("ffn_up.weight"));
        lw.Wdown   = mk_mm(suf("ffn_down.weight"));
        return lw.ln_in && lw.Wq && lw.Wk && lw.Wv && lw.Wo && lw.ln_post && lw.Wgate && lw.Wup && lw.Wdown;
    };
    auto load_adarms_layer = [&](const char * tower, int i, AdaGemmaLayerW & lw) -> bool {
        char b[256];
        auto suf = [&](const char * s) {
            std::snprintf(b, sizeof(b), "%s.blk.%d.%s", tower, i, s);
            return b;
        };
        // pi0.5 expert norms are adaptive RMSNorms, represented by zero-init
        // dense projections from the time condition to scale/shift/gate.
        lw.ln_in_W   = mk_f32(suf("attn_norm.dense.weight"));
        lw.ln_in_b   = mk_f32(suf("attn_norm.dense.bias"));
        lw.Wo        = mk_mm(suf("attn_o.weight"));
        lw.ln_post_W = mk_f32(suf("ffn_norm.dense.weight"));
        lw.ln_post_b = mk_f32(suf("ffn_norm.dense.bias"));
        lw.Wdown     = mk_mm(suf("ffn_down.weight"));
        lw.Wq        = mk_mm(suf("attn_q.weight"));
        lw.Wk        = mk_mm(suf("attn_k.weight"));
        lw.Wv        = mk_mm(suf("attn_v.weight"));
        lw.Wgate     = mk_mm(suf("ffn_gate.weight"));
        lw.Wup       = mk_mm(suf("ffn_up.weight"));
        return lw.ln_in_W && lw.ln_in_b && lw.Wq && lw.Wk && lw.Wv && lw.Wo && lw.ln_post_W && lw.ln_post_b &&
               lw.Wgate && lw.Wup && lw.Wdown;
    };

    m->pl_layers.resize(cfg.n_layers);
    m->ex_layers.resize(cfg.n_layers);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        if (!load_layer("vlm", (int) i, m->pl_layers[i])) {
            return nullptr;
        }
        if (!load_adarms_layer("aex", (int) i, m->ex_layers[i])) {
            return nullptr;
        }
    }
    m->ex_final_norm_W = mk_f32("aex.output_norm.dense.weight");
    m->ex_final_norm_b = mk_f32("aex.output_norm.dense.bias");
    m->W_ain           = mk_f32("action_in_proj.weight");
    m->b_ain           = mk_f32("action_in_proj.bias");
    m->W_tm1           = mk_f32("time_mlp_in.weight");
    m->b_tm1           = mk_f32("time_mlp_in.bias");
    m->W_tm2           = mk_f32("time_mlp_out.weight");
    m->b_tm2           = mk_f32("time_mlp_out.bias");
    m->W_aout          = mk_f32("action_out_proj.weight");
    m->b_aout          = mk_f32("action_out_proj.bias");
    for (ggml_tensor * t : weights) {
        if (!t) {
            std::fprintf(stderr, "vla(pi05): weight tensor creation failed\n");
            return nullptr;
        }
    }
    if (!m->ex_final_norm_W || !m->ex_final_norm_b || !m->W_ain || !m->b_ain || !m->W_tm1 || !m->b_tm1 || !m->W_tm2 ||
        !m->b_tm2 || !m->W_aout || !m->b_aout) {
        std::fprintf(stderr, "vla(pi05): failed to wire projection / norm tensors\n");
        return nullptr;
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) {
        std::fprintf(stderr, "vla(pi05): ggml_backend_alloc_ctx_tensors failed (out of memory?)\n");
        return nullptr;
    }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type, is_gemma_norm(t->name));
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(pi05): upload size mismatch for %s (%zu vs %zu)\n", t->name, bytes.size(),
                         ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(pi05): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    if (!load_stats(g, *m)) {
        return nullptr;
    }
    std::printf("vla(pi05): model loaded (n_threads=%d)\n", m->n_threads);
    return m.release();
}

void model_free(Model * m) {
    delete m;
}

const Config & model_config(const Model * m) {
    return static_cast<const Pi05Model *>(m)->cfg;
}

const Stats & last_stats(const Model * m) {
    return static_cast<const Pi05Model *>(m)->stats;
}

std::vector<float> predict(Model * m, const Inputs & in) {
    return static_cast<Pi05Model *>(m)->predict(in);
}

PreparedInput prepare(Model * m, const Inputs & in) {
    return static_cast<Pi05Model *>(m)->prepare_inputs(in);
}

std::vector<float> compute(Model * m, const PreparedInput & p) {
    return static_cast<Pi05Model *>(m)->compute_actions(p);
}

PreparedInput Pi05Model::prepare_inputs(const Inputs & in) {
    using clk = std::chrono::high_resolution_clock;
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats = Stats{};
    }

    const Config & cfg       = this->cfg;
    const int64_t  hidden_pl = cfg.hidden;
    const int64_t  chunk     = cfg.n_suffix;

    PreparedInput p;
    int64_t       n_img_tokens = 0;

    if (in.precomputed_img_emb) {
        p.n_img_tokens = (int64_t) in.n_img_views * cfg.n_img;
        p.img_emb_host.assign(in.precomputed_img_emb,
                              in.precomputed_img_emb + (size_t) p.n_img_tokens * hidden_pl);
    } else {
        if (in.n_images < 1 || !in.images) {
            std::fprintf(stderr, "vla(pi05): prepare: no images and no precomputed_img_emb\n");
            p.ok = false;
            p.error = "no images and no precomputed_img_emb";
            return p;
        }
        const int     img_sz          = clip_get_image_size(cctx);
        const size_t  per_pix         = (size_t) 3 * img_sz * img_sz;
        const size_t  per_out         = clip_embd_nbytes_by_img(cctx, img_sz, img_sz) / sizeof(float);
        const int64_t per_view_tokens = (int64_t) (per_out / (size_t) hidden_pl);

        // VisPruner: per-view pruning config (env read once).
        // Multi-view fix: each view is pruned independently to vp_tokens / n_views
        // using its own attention weights (clip_get_last_attn_data is overwritten
        // on every clip_encode_float_image call, so pruning must happen right
        // after each view's encode inside the loop below).
        static int   vp_tokens     = -1;
        static float vp_ratio      = 0.5f;
        static bool  vp_keep_order = false;
        if (vp_tokens < 0) {
            const char * env_t = std::getenv("VLA_PI05_VIS_PRUNER_TOKENS");
            const char * env_r = std::getenv("VLA_PI05_VIS_PRUNER_RATIO");
            const char * env_k = std::getenv("VLA_PI05_VIS_PRUNER_KEEP_ORDER");
            vp_tokens          = env_t ? std::atoi(env_t) : 0;
            vp_ratio           = env_r ? (float) std::atof(env_r) : 0.5f;
            vp_keep_order      = env_k ? (std::atoi(env_k) > 0) : false;
        }
        const int vp_per_view = (vp_tokens > 0) ? std::max(1, vp_tokens / std::max(1, (int) in.n_images)) : 0;

        std::vector<float> hwc(per_pix);
        const auto         tv0 = clk::now();
        p.img_emb_host.clear();
        p.img_emb_host.reserve(per_out * (size_t) in.n_images);

        for (int v = 0; v < in.n_images; ++v) {
            const ImageView & view = in.images[v];
            if (view.w != img_sz || view.h != img_sz) {
                std::fprintf(stderr, "vla(pi05): image[%d] is %dx%d; π0.5 requires %dx%d\n", v, view.w, view.h, img_sz,
                             img_sz);
                p.ok = false;
                p.error = "bad image size";
                return p;
            }
            if (view.format == PixelFormat::U8) {
                const uint8_t * src = static_cast<const uint8_t *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) {
                    hwc[i] = (float) src[i] / 127.5f - 1.0f;
                }
            } else {
                const float * src = static_cast<const float *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) {
                    hwc[i] = src[i] * 2.0f - 1.0f;
                }
            }

            std::vector<float> view_emb(per_out);
            if (!clip_encode_float_image(cctx, n_threads, hwc.data(), img_sz, img_sz, view_emb.data())) {
                std::fprintf(stderr, "vla(pi05): clip_encode_float_image failed (view %d)\n", v);
                p.ok = false;
                p.error = "clip_encode_float_image failed";
                return p;
            }

            // DEBUG dump of raw vision embeddings (env VLA_PI05_DUMP_IMG=1) before pruning
            if (const char * dump_env = std::getenv("VLA_PI05_DUMP_IMG")) {
                if (std::atoi(dump_env) > 0) {
                    char path[256];
                    std::snprintf(path, sizeof(path), "/tmp/pi05_img_emb_view%d.f32", v);
                    FILE * fp = std::fopen(path, "wb");
                    if (fp) {
                        std::fwrite(view_emb.data(), sizeof(float), per_out, fp);
                        std::fclose(fp);
                    }
                    double        s = 0.0, sa = 0.0, s2 = 0.0;
                    const float * p = view_emb.data();
                    for (size_t i = 0; i < per_out; ++i) {
                        s += p[i];
                        sa += std::fabs(p[i]);
                        s2 += (double) p[i] * p[i];
                    }
                    std::printf(
                        "[pi05-debug] view %d: n=%zu mean=%.6f meanabs=%.6f rms=%.6f first=[%.4f %.4f %.4f %.4f]\n", v,
                        per_out, s / (double) per_out, sa / (double) per_out, std::sqrt(s2 / (double) per_out), p[0],
                        p[1], p[2], p[3]);
                }
            }

            // VisPruner: prune this view's tokens using its own attention weights
            // (clip_get_last_attn_data reflects the just-encoded view)
            int64_t v_tokens = per_view_tokens;
            if (vp_per_view > 0 && v_tokens > vp_per_view) {
                int           n_patches = 0, n_heads = 0;
                const float * attn = clip_get_last_attn_data(cctx, &n_patches, &n_heads);
                if (attn && n_patches == (int) v_tokens) {
                    vispruner_prune(view_emb, v_tokens, hidden_pl, attn, n_patches, n_heads, vp_per_view, vp_ratio,
                                    vp_keep_order);
                } else {
                    std::fprintf(stderr, "vla(pi05): VisPruner skipped view %d (attn=%p patches=%d tokens=%lld)\n", v,
                                 (const void *) attn, n_patches, (long long) v_tokens);
                }
            }
            p.img_emb_host.insert(p.img_emb_host.end(), view_emb.begin(), view_emb.end());
            n_img_tokens += v_tokens;
        }
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - tv0).count();
        }
    }
    p.n_img_tokens = n_img_tokens;

    // ── SEG2: language embedding lookup ──
    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "vla(pi05): prepare: empty lang_tokens\n");
        p.ok = false;
        p.error = "empty lang_tokens";
        return p;
    }
    const int64_t          n_lang = in.n_lang;
    std::vector<int32_t>   lang_ids(in.lang_tokens, in.lang_tokens + n_lang);
    p.lang_rows.resize((size_t) n_lang * hidden_pl);
    {
        gguf_reader g;
        if (!g.open(ckpt_path_)) {
            p.ok = false;
            p.error = "gguf_reader open failed";
            return p;
        }
        if (!g.fetch_rows_f32("token_embd.weight", lang_ids, p.lang_rows.data(), hidden_pl)) {
            p.ok = false;
            p.error = "token_embd.weight lookup failed";
            return p;
        }
    }
    p.n_lang = n_lang;

    if (in.noise) p.noise.assign(in.noise, in.noise + (size_t) cfg.max_action_dim * chunk);
    if (in.prev_chunk && in.n_prev_chunk > 0) {
        p.prev_chunk.assign(in.prev_chunk, in.prev_chunk + (size_t) in.n_prev_chunk * cfg.real_action_dim);
        p.n_prev_chunk = in.n_prev_chunk;
    }
    if (in.attention_mask && in.attention_mask_n > 0)
        p.attention_mask.assign(in.attention_mask, in.attention_mask + in.attention_mask_n);
    p.timing_detail = in.timing_detail;
    p.ok = true;
    return p;
}

std::vector<float> Pi05Model::compute_actions(const PreparedInput & p) {
    using clk = std::chrono::high_resolution_clock;

    const Config & cfg       = this->cfg;
    const int64_t  hidden_pl = cfg.hidden;
    const int64_t  hidden_ex = cfg.expert_h;
    const int64_t  chunk     = cfg.n_suffix;
    const int64_t  n_suf     = chunk;
    const int64_t  n_layers  = cfg.n_layers;
    const int64_t  max_ad    = cfg.max_action_dim;
    const int      num_steps = cfg.num_steps;
    const float    dt        = -1.0f / (float) num_steps;
    const float    rope_base = cfg.rope_freq_base;

    const int64_t n_img_tokens = p.n_img_tokens;
    const int64_t n_lang       = p.n_lang;
    const int64_t n_prefix     = n_img_tokens + n_lang;
    const int64_t n_total      = n_prefix + n_suf;

    init_rtc_config();

    // ── Cached compute graph: rebuilt only when the input token counts change ──
    // (server is single-threaded REQ/REP, so one cached graph is safe to reuse)
    // VLA_PI05_NO_CACHE=1 forces a rebuild every request (emulates the original
    // pre-caching behavior, used as an honest baseline for the same codebase).
    static const bool no_cache = []() {
        const char * e = std::getenv("VLA_PI05_NO_CACHE");
        return e && std::atoi(e) > 0;
    }();
    const bool need_rebuild =
        no_cache ||
        (gf_ == nullptr || c_n_img_tokens_ != n_img_tokens || c_n_lang_ != n_lang || c_n_suf_ != n_suf ||
         c_rtc_         != (int64_t) rtc_enabled_ ||
         c_rtc_horizon_ != (int64_t) rtc_horizon_);
    if (need_rebuild) {
        if (galloc_) {
            ggml_gallocr_free(galloc_);
            galloc_ = nullptr;
        }
        if (comp_ctx_) {
            ggml_free(comp_ctx_);
            comp_ctx_ = nullptr;
        }
        gf_          = nullptr;
        x_final_     = nullptr;
        t_image_emb_ = t_lang_emb_ = t_prefix_pos_ = t_x0_ = t_suffix_pos_ = t_full_mask_ = nullptr;
        t_prev_chunk_ = t_rtc_weights_ = nullptr;
        t_time_.clear();

        ggml_init_params cp = { (size_t) 64 * 1024 * 1024, nullptr, true };
        comp_ctx_           = ggml_init(cp);
        if (!comp_ctx_) {
            std::fprintf(stderr, "vla(pi05): ggml_init(ctx_compute) failed\n");
            return {};
        }
        ggml_context * C = comp_ctx_;

        t_image_emb_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_img_tokens);
        ggml_set_input(t_image_emb_);
        t_lang_emb_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_lang);
        ggml_set_input(t_lang_emb_);
        t_prefix_pos_ = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_prefix);
        ggml_set_input(t_prefix_pos_);
        t_x0_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);
        ggml_set_input(t_x0_);
        t_suffix_pos_ = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);
        ggml_set_input(t_suffix_pos_);
        t_full_mask_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_total, n_suf);
        ggml_set_input(t_full_mask_);
        if (rtc_enabled_) {
            t_prev_chunk_  = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);
            ggml_set_input(t_prev_chunk_);
            t_rtc_weights_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);
            ggml_set_input(t_rtc_weights_);
        }
        t_time_.resize(num_steps);
        for (int s = 0; s < num_steps; ++s) {
            t_time_[s] = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_ex, 1);
            ggml_set_input(t_time_[s]);
        }

        const float   lang_scale  = (float) std::sqrt((double) hidden_pl);
        ggml_tensor * prefix_embs = ggml_concat(C, t_image_emb_, ggml_scale(C, t_lang_emb_, lang_scale), 1);

        std::vector<ggml_tensor *> cK(n_layers), cV(n_layers);
        {
            ggml_tensor * h = prefix_embs;
            for (int64_t i = 0; i < n_layers; ++i) {
                h = build_gemma_layer(C, pl_layers[i], h, t_prefix_pos_, cfg, n_prefix, rope_base, nullptr, nullptr,
                                      nullptr, flash_attn_, &cK[i], &cV[i]);
            }
            (void) h;
        }

        ggml_tensor * x_t = t_x0_;
        for (int step = 0; step < num_steps; ++step) {
            ggml_tensor * adarms_cond = nullptr;
            ggml_tensor * h           = build_embed_suffix(C, *this, x_t, t_time_[step], &adarms_cond);
            for (int64_t i = 0; i < n_layers; ++i) {
                h = build_adarms_gemma_layer(C, ex_layers[i], h, t_suffix_pos_, adarms_cond, cfg, n_suf, rope_base,
                                             cK[i], cV[i], t_full_mask_, flash_attn_);
            }
            // The final expert norm is also AdaRMS; its gate output is ignored by OpenPI.
            ggml_tensor * h_final = adarms_norm(C, h, ex_final_norm_W, ex_final_norm_b, adarms_cond, cfg, nullptr);
            ggml_tensor * v_t     = ggml_add(C, ggml_mul_mat(C, W_aout, h_final), b_aout);
            if (rtc_enabled_) {
                // RTC prefix guidance (LeRobot modeling_rtc.denoise_step):
                //   x1_t = x_t - time*v_t
                //   err  = (prev_chunk - x1_t) * weights
                //   v_t -= min(c*inv_r2, max_guidance) * err
                // First-order is EXACT here: LeRobot computes v_t before enabling
                // grad on x_t, so autograd.grad(x1_t, x_t)[0] == err (identity).
                const float time = 1.0f + (float) step * dt;
                ggml_tensor * x1_est = ggml_sub(C, x_t, ggml_scale(C, v_t, time));
                ggml_tensor * diff   = ggml_sub(C, t_prev_chunk_, x1_est);
                ggml_tensor * err    = ggml_mul(C, diff, t_rtc_weights_);
                v_t = ggml_sub(C, v_t, ggml_scale(C, err, rtc_step_guidance_[step]));
            }
            x_t                   = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
        }
        x_final_ = x_t;
        ggml_set_output(x_final_);

        // AdaRMS adds scale/shift/gate nodes on every expert norm, so pi0.5
        // needs a larger graph arena than the plain pi0 action expert.
        gf_ = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf_, x_final_);

        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf_)) {
            std::fprintf(stderr, "vla(pi05): ggml_gallocr_alloc_graph failed (out of memory?)\n");
            if (galloc_) {
                ggml_gallocr_free(galloc_);
                galloc_ = nullptr;
            }
            if (comp_ctx_) {
                ggml_free(comp_ctx_);
                comp_ctx_ = nullptr;
            }
            gf_      = nullptr;
            x_final_ = nullptr;
            return {};
        }
        c_n_img_tokens_ = n_img_tokens;
        c_n_lang_       = n_lang;
        c_n_suf_        = n_suf;
        c_rtc_          = rtc_enabled_ ? 1 : 0;
        c_rtc_horizon_  = rtc_enabled_ ? rtc_horizon_ : 0;
    }

    // ── upload inputs (every request) ──
    ggml_backend_tensor_set(t_image_emb_, p.img_emb_host.data(), 0, ggml_nbytes(t_image_emb_));
    ggml_backend_tensor_set(t_lang_emb_, p.lang_rows.data(), 0, ggml_nbytes(t_lang_emb_));
    {
        std::vector<int32_t> pp(n_prefix);
        for (int64_t i = 0; i < n_prefix; ++i) {
            pp[i] = (int32_t) i;
        }
        ggml_backend_tensor_set(t_prefix_pos_, pp.data(), 0, ggml_nbytes(t_prefix_pos_));
        std::vector<int32_t> sp(n_suf);
        for (int64_t i = 0; i < n_suf; ++i) {
            sp[i] = (int32_t) (n_prefix + i);
        }
        ggml_backend_tensor_set(t_suffix_pos_, sp.data(), 0, ggml_nbytes(t_suffix_pos_));
    }
    {
        std::vector<float> x0h((size_t) max_ad * chunk);
        if (p.noise.size() == (size_t) max_ad * chunk) {
            std::memcpy(x0h.data(), p.noise.data(), x0h.size() * sizeof(float));
        } else {
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto & v : x0h) {
                v = nd(rng);
            }
        }
        // RTC x0 inpainting: seed the leftover prefix into the initial noise so the
        // flow trajectory starts from the executed prefix (VLA_PI05_RTC_X0_INPAINT=1).
        if (rtc_x0_inpaint_ && p.n_prev_chunk > 0) {
            const int64_t L = std::min((int64_t) rtc_horizon_, p.n_prev_chunk);
            for (int64_t t = 0; t < L; ++t) {
                for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
                    const float xr = p.prev_chunk[(size_t) t * cfg.real_action_dim + j];
                    if (action_norm_mode == "MEAN_STD")
                        x0h[(size_t) t * max_ad + j] = (xr - action_mean[j]) / (action_std[j] + cfg.norm_eps);
                    else {
                        const float denom = action_q99[j] - action_q01[j];
                        x0h[(size_t) t * max_ad + j] = 2.0f * (xr - action_q01[j]) / denom - 1.0f;
                    }
                }
            }
        }
        ggml_backend_tensor_set(t_x0_, x0h.data(), 0, ggml_nbytes(t_x0_));
    }
    if (rtc_enabled_) {
        // RTC prev_chunk: real units -> normalized latent, laid out [max_ad, chunk].
        // Rows beyond real_action_dim stay 0 (never guided).
        std::vector<float> prevh((size_t) max_ad * chunk, 0.f);
        if (p.n_prev_chunk > 0) {
            const int64_t nt = std::min(p.n_prev_chunk, chunk);
            for (int64_t t = 0; t < nt; ++t) {
                for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
                    const float xr = p.prev_chunk[(size_t) t * cfg.real_action_dim + j];
                    if (action_norm_mode == "MEAN_STD")
                        prevh[(size_t) t * max_ad + j] = (xr - action_mean[j]) / (action_std[j] + cfg.norm_eps);
                    else {
                        const float denom = action_q99[j] - action_q01[j];
                        prevh[(size_t) t * max_ad + j] = 2.0f * (xr - action_q01[j]) / denom - 1.0f;
                    }
                }
            }
        }
        ggml_backend_tensor_set(t_prev_chunk_, prevh.data(), 0, ggml_nbytes(t_prev_chunk_));

        // RTC weights: schedule weights on real action rows; ALL ZEROS when no
        // leftover -> err == 0 -> v_t untouched (bit-exact baseline preserved).
        // The execution horizon is clamped to the actual leftover length, matching
        // LeRobot denoise_step: execution_horizon = min(config, leftover.shape[1]).
        // Rows beyond the leftover carry zero weight -> no garbage guidance.
        std::vector<float> wgh((size_t) max_ad * chunk, 0.f);
        if (p.n_prev_chunk > 0) {
            const int L = (int) std::min<int64_t>(rtc_horizon_, p.n_prev_chunk);
            const int D = std::min(rtc_delay_, L);
            std::vector<float> sched((size_t) chunk, 0.f);
            for (int i = 0; i < D; ++i) sched[(size_t) i] = 1.f;
            if (L > D) {
                const float denom = (float) (L - D) + 1.0f;
                const float em1   = std::exp(1.0f) - 1.0f;
                for (int i = D; i < L; ++i) {
                    float w = ((float) L - (float) i) / denom;
                    if (rtc_schedule_ == "exp") w = w * std::expm1(w) / em1;
                    sched[(size_t) i] = w;
                }
            }
            for (int64_t t = 0; t < chunk; ++t)
                for (int64_t j = 0; j < cfg.real_action_dim; ++j)
                    wgh[(size_t) t * max_ad + j] = sched[(size_t) t];
        }
        ggml_backend_tensor_set(t_rtc_weights_, wgh.data(), 0, ggml_nbytes(t_rtc_weights_));
    }
    if (!flash_attn_) {
        // Flash attention uses a null mask, so t_full_mask_ is not part of the
        // graph and gallocr does not allocate it; only upload when in use.
        std::vector<float> mk((size_t) n_total * n_suf, 0.f);
        ggml_backend_tensor_set(t_full_mask_, mk.data(), 0, ggml_nbytes(t_full_mask_));
    }
    for (int s = 0; s < num_steps; ++s) {
        const float              timestep = 1.0f + (float) s * dt;
        const std::vector<float> tv       = sinusoidal_time_emb(timestep, hidden_ex, cfg.min_period, cfg.max_period);
        ggml_backend_tensor_set(t_time_[s], tv.data(), 0, ggml_nbytes(t_time_[s]));
    }

    // ── compute + read output ──
    const auto        ti0 = clk::now();
    const ggml_status st  = ggml_backend_graph_compute(backend, gf_);
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();
    }
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(pi05): ggml_backend_graph_compute failed (%d)\n", (int) st);
        return {};
    }

    std::vector<float> out((size_t) chunk * max_ad);
    ggml_backend_tensor_get(x_final_, out.data(), 0, out.size() * sizeof(float));
    for (int64_t t = 0; t < chunk; ++t) {
        float * row = out.data() + (size_t) t * max_ad;
        for (int64_t j = 0; j < cfg.real_action_dim && j < max_ad; ++j) {
            if (action_norm_mode == "MEAN_STD") {
                row[j] = row[j] * (action_std[j] + cfg.norm_eps) + action_mean[j];
            } else {
                const float denom = action_q99[j] - action_q01[j];
                row[j]            = (row[j] + 1.0f) * denom * 0.5f + action_q01[j];
            }
        }
    }
    return out;
}

std::vector<float> Pi05Model::predict(const Inputs & in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    PreparedInput p = prepare_inputs(in);
    if (!p.ok) return {};
    std::vector<float> out = compute_actions(p);
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    }
    return out;
}

}  // namespace vla
