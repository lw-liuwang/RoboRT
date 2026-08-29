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
 * @file fasterwam.cpp
 * @brief FasterWAM (World Action Model) C++ inference engine for RoboRT.
 *
 * FasterWAM (hustvl/FasterWAM) combines a Wan2.2-TI2V-5B video DiT
 * (30 layers, hidden=3072) with a SparseActionDiT action expert (30 layers,
 * hidden=1024) via a Mixture-of-Transformers (MoT) design.
 *
 * Inference flow:
 *   1. prefill_video_cache()  – run the video DiT once (t=0) accumulating
 *      interval K/V tensors; fuse them with softmax-weighted logits at the
 *      8 condition layers [0,4,8,12,16,20,24,28].
 *   2. denoise loop (N=20 steps) – run the action DiT, cross-attending the
 *      cached fused video K/V at each condition layer.
 *   3. Denormalize actions and return the action chunk.
 *
 * Weight naming conventions are fixed by convert_fasterwam_to_gguf.py.
 * Architecture GGUF KV: fasterwam.architecture = "fasterwam"
 */

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "model.h"
#ifdef GGML_USE_CUDA
#    include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vla {
namespace {

// ---------------------------------------------------------------------------
// GGUF reader (identical pattern to hy_vla.cpp)
// ---------------------------------------------------------------------------

struct gguf_reader_fw {
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
            std::fprintf(stderr, "fasterwam: gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "fasterwam: fopen failed for %s\n", path.c_str());
            return false;
        }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }

    ~gguf_reader_fw() {
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

    bool has_key(const char * k) const { return gguf_find_key(gctx, k) >= 0; }

    uint32_t u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }

    float f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }

    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    ggml_type tensor_type(const char * name) const {
        const ggml_tensor * t = meta(name);
        return t ? t->type : GGML_TYPE_COUNT;
    }

    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) {
            std::fprintf(stderr, "fasterwam: missing tensor %s\n", name);
            return false;
        }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) {
            return false;
        }
        return std::fread(buf, 1, nb, fp) == nb;
    }

    // Convert tensor to target ggml_type, returning raw bytes.
    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        const ggml_tensor * t = meta(name);
        if (!t) {
            std::fprintf(stderr, "fasterwam: missing tensor %s\n", name);
            return {};
        }
        if (t->type == target) {
            const int64_t id = gguf_find_tensor(gctx, name);
            if (id < 0) {
                return {};
            }
            const size_t         nb = gguf_get_tensor_size(gctx, id);
            std::vector<uint8_t> out(nb);
            if (!read_raw(name, out.data())) {
                return {};
            }
            return out;
        }
        const int64_t      n = ggml_nelements(t);
        std::vector<float> f32v((size_t) n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32v.data())) {
                return {};
            }
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp((size_t) n);
            if (!read_raw(name, tmp.data())) {
                return {};
            }
            ggml_bf16_to_fp32_row(tmp.data(), f32v.data(), n);
        } else {
            std::fprintf(stderr, "fasterwam: unsupported dtype %d for %s\n", (int) t->type, name);
            return {};
        }
        if (target == GGML_TYPE_F32) {
            std::vector<uint8_t> out((size_t) n * sizeof(float));
            std::memcpy(out.data(), f32v.data(), out.size());
            return out;
        }
        if (target == GGML_TYPE_BF16) {
            std::vector<uint8_t> out((size_t) n * sizeof(ggml_bf16_t));
            ggml_fp32_to_bf16_row(f32v.data(), reinterpret_cast<ggml_bf16_t *>(out.data()), n);
            return out;
        }
        std::fprintf(stderr, "fasterwam: unsupported resident dtype %d for %s\n", (int) target, name);
        return {};
    }

    bool read_to_f32(const char * name, std::vector<float> & out) {
        const ggml_tensor * t = meta(name);
        if (!t) {
            return false;
        }
        out.resize((size_t) ggml_nelements(t));
        if (t->type == GGML_TYPE_F32) {
            return read_raw(name, out.data());
        }
        if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp((size_t) ggml_nelements(t));
            if (!read_raw(name, tmp.data())) {
                return false;
            }
            ggml_bf16_to_fp32_row(tmp.data(), out.data(), ggml_nelements(t));
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Architecture-specific config (not stored in vla::Config)
// ---------------------------------------------------------------------------

struct FasterWAMCfg {
    // Video DiT
    int video_hidden      = 3072;
    int video_ffn_dim     = 14336;
    int video_num_heads   = 24;
    int video_head_dim    = 128;
    int video_num_layers  = 30;
    int video_in_dim      = 16;   // VAE latent channels * patch_size^2 (packed 2x2)
    int video_freq_dim    = 256;  // sinusoidal time embedding dim
    int video_text_dim    = 4096; // text encoder hidden dim (T5)

    // Action DiT
    int action_hidden         = 1024;
    int action_ffn_dim        = 4096;
    int action_num_heads      = 24;   // condition layers
    int action_noncond_heads  = 8;    // non-condition layers
    int action_head_dim       = 128;
    int action_num_layers     = 30;
    int action_dim            = 14;
    int action_freq_dim       = 256;
    int action_text_dim       = 4096;

    // MoT
    int num_condition_layers = 8;
    int condition_layers[8]  = {0, 4, 8, 12, 16, 20, 24, 28};

    // Normalization
    int proprio_dim = 14;
    int state_dim   = 14;

    // Inference
    int   action_horizon       = 32;
    int   num_inference_steps  = 20;
    int   num_video_frames     = 9;
    float infer_shift          = 5.0f;
    int   num_train_timesteps  = 1000;
};

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

// DiT modulation + norm weights for a single block
struct VideoDiTBlockW {
    // Self-attention
    ggml_tensor * sa_q_w       = nullptr;
    ggml_tensor * sa_q_b       = nullptr;
    ggml_tensor * sa_k_w       = nullptr;
    ggml_tensor * sa_k_b       = nullptr;
    ggml_tensor * sa_v_w       = nullptr;
    ggml_tensor * sa_v_b       = nullptr;
    ggml_tensor * sa_o_w       = nullptr;
    ggml_tensor * sa_o_b       = nullptr;
    ggml_tensor * sa_norm_q    = nullptr;  // RMSNorm weight (no bias)
    ggml_tensor * sa_norm_k    = nullptr;
    // FFN
    ggml_tensor * ffn0_w       = nullptr;
    ggml_tensor * ffn0_b       = nullptr;
    ggml_tensor * ffn2_w       = nullptr;
    ggml_tensor * ffn2_b       = nullptr;
    // DiT modulation [1, 6, hidden]
    ggml_tensor * modulation   = nullptr;
    // Cross-attention (text) pre-norm (affine LayerNorm)
    ggml_tensor * norm3_w      = nullptr;
    ggml_tensor * norm3_b      = nullptr;
    // Cross-attention (all video blocks attend text)
    ggml_tensor * ca_q_w       = nullptr;
    ggml_tensor * ca_q_b       = nullptr;
    ggml_tensor * ca_k_w       = nullptr;
    ggml_tensor * ca_k_b       = nullptr;
    ggml_tensor * ca_v_w       = nullptr;
    ggml_tensor * ca_v_b       = nullptr;
    ggml_tensor * ca_o_w       = nullptr;
    ggml_tensor * ca_o_b       = nullptr;
    ggml_tensor * ca_norm_q    = nullptr;
    ggml_tensor * ca_norm_k    = nullptr;
};

// Action DiT block – condition layers have cross_attn to cached video KV
struct ActionDiTBlockW {
    // Self-attention
    ggml_tensor * sa_q_w       = nullptr;
    ggml_tensor * sa_q_b       = nullptr;
    ggml_tensor * sa_k_w       = nullptr;
    ggml_tensor * sa_k_b       = nullptr;
    ggml_tensor * sa_v_w       = nullptr;
    ggml_tensor * sa_v_b       = nullptr;
    ggml_tensor * sa_o_w       = nullptr;
    ggml_tensor * sa_o_b       = nullptr;
    ggml_tensor * sa_norm_q    = nullptr;
    ggml_tensor * sa_norm_k    = nullptr;
    // FFN
    ggml_tensor * ffn0_w       = nullptr;
    ggml_tensor * ffn0_b       = nullptr;
    ggml_tensor * ffn2_w       = nullptr;
    ggml_tensor * ffn2_b       = nullptr;
    // DiT modulation [1, 6, hidden]  (global, not per-token)
    ggml_tensor * modulation   = nullptr;
    // Cross-attention to video KV (condition layers only)
    bool          is_condition = false;
    ggml_tensor * norm3_w      = nullptr;
    ggml_tensor * norm3_b      = nullptr;
    ggml_tensor * ca_q_w       = nullptr;
    ggml_tensor * ca_q_b       = nullptr;
    ggml_tensor * ca_k_w       = nullptr;
    ggml_tensor * ca_k_b       = nullptr;
    ggml_tensor * ca_v_w       = nullptr;
    ggml_tensor * ca_v_b       = nullptr;
    ggml_tensor * ca_o_w       = nullptr;
    ggml_tensor * ca_o_b       = nullptr;
    ggml_tensor * ca_norm_q    = nullptr;
    ggml_tensor * ca_norm_k    = nullptr;
};

struct FasterWAMWeights {
    // Video DiT front-end
    ggml_tensor * vid_patch_emb_w  = nullptr;  // conv3d weight [out, in, t, h, w]
    ggml_tensor * vid_patch_emb_b  = nullptr;
    ggml_tensor * vid_time_emb_0w  = nullptr;  // MLP layer 0
    ggml_tensor * vid_time_emb_0b  = nullptr;
    ggml_tensor * vid_time_emb_2w  = nullptr;  // MLP layer 2
    ggml_tensor * vid_time_emb_2b  = nullptr;
    ggml_tensor * vid_time_proj_w  = nullptr;  // SiLU + Linear → 6*h
    ggml_tensor * vid_time_proj_b  = nullptr;
    ggml_tensor * vid_text_emb_0w  = nullptr;
    ggml_tensor * vid_text_emb_0b  = nullptr;
    ggml_tensor * vid_text_emb_2w  = nullptr;
    ggml_tensor * vid_text_emb_2b  = nullptr;
    // Video DiT head
    ggml_tensor * vid_head_w       = nullptr;
    ggml_tensor * vid_head_b       = nullptr;
    ggml_tensor * vid_head_mod     = nullptr;  // [1,2,hidden]
    // Video KV fusion logits: 8 tensors (j=0: [1], j=1..7: [4] each)
    ggml_tensor * vid_kv_logits[8] = {};
    // Video DiT blocks
    std::vector<VideoDiTBlockW> video_blocks;

    // Action DiT front-end
    ggml_tensor * act_encoder_w    = nullptr;
    ggml_tensor * act_encoder_b    = nullptr;
    ggml_tensor * act_head_w       = nullptr;
    ggml_tensor * act_head_b       = nullptr;
    ggml_tensor * act_time_emb_0w  = nullptr;
    ggml_tensor * act_time_emb_0b  = nullptr;
    ggml_tensor * act_time_emb_2w  = nullptr;
    ggml_tensor * act_time_emb_2b  = nullptr;
    ggml_tensor * act_time_proj_w  = nullptr;
    ggml_tensor * act_time_proj_b  = nullptr;
    ggml_tensor * act_text_emb_0w  = nullptr;
    ggml_tensor * act_text_emb_0b  = nullptr;
    ggml_tensor * act_text_emb_2w  = nullptr;
    ggml_tensor * act_text_emb_2b  = nullptr;
    // Action DiT blocks
    std::vector<ActionDiTBlockW> action_blocks;

    // Proprio encoder
    ggml_tensor * proprio_enc_w    = nullptr;
    ggml_tensor * proprio_enc_b    = nullptr;
};

// ---------------------------------------------------------------------------
// FasterWAMModel
// ---------------------------------------------------------------------------

struct FasterWAMModel final : public Model {
    ~FasterWAMModel() override {
        if (const_buf) {
            ggml_backend_buffer_free(const_buf);
        }
        if (ctx_const) {
            ggml_free(ctx_const);
        }
        if (ctx_step) {
            ggml_free(ctx_step);
        }
        if (ctx_vid_prefill) {
            ggml_free(ctx_vid_prefill);
        }
        if (prefill_galloc) {
            ggml_gallocr_free(prefill_galloc);
        }
        if (ctx_proprio) {
            ggml_free(ctx_proprio);
        }
        if (proprio_galloc) {
            ggml_gallocr_free(proprio_galloc);
        }
        if (ctx_textemb) {
            ggml_free(ctx_textemb);
        }
        if (textemb_galloc) {
            ggml_gallocr_free(textemb_galloc);
        }
        if (weight_buf) {
            ggml_backend_buffer_free(weight_buf);
        }
        if (galloc) {
            ggml_gallocr_free(galloc);
        }
        if (ctx_weights) {
            ggml_free(ctx_weights);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    const Config & config() const override { return cfg; }
    const Stats &  last_stats() const override { return stats; }

    std::vector<float> predict(const Inputs & in) override;
    PreparedInput      prepare(const Inputs & in) override;
    std::vector<float> compute(const PreparedInput & p) override;

    Config         cfg{};
    Stats          stats{};
    FasterWAMCfg   fw_cfg{};
    FasterWAMWeights w{};

    ggml_backend_t         backend     = nullptr;
    ggml_context *         ctx_weights = nullptr;
    ggml_backend_buffer_t  weight_buf  = nullptr;
    mutable ggml_gallocr_t galloc      = nullptr;
    ggml_type              wtype       = GGML_TYPE_BF16;
    bool                   is_cuda     = false;
    int                    n_threads   = 4;

    // Persistent denoising-loop resources (allocated once per predict call,
    // freed in destructor via ctx_step_buf).
    // ctx_step holds the unrolled 20-step graph metadata (host-side).
    mutable ggml_context *        ctx_step     = nullptr;
    mutable ggml_backend_buffer_t ctx_step_buf = nullptr;  // GPU buffer for constant inputs

    // Persistent GPU tensors for constant inputs across denoising steps.
    // Allocated once before the loop; vid_kv, text_ctx, pos_ids are static per predict().
    mutable ggml_context *  ctx_const    = nullptr;  // host-side metadata context
    mutable ggml_tensor *   tc_text      = nullptr;  // action text context [hd, text_seq]
    mutable ggml_tensor *   tc_pos       = nullptr;  // position ids [action_seq]
    mutable ggml_tensor *   tc_vid_k[8] = {};        // video K per cond layer
    mutable ggml_tensor *   tc_vid_v[8] = {};        // video V per cond layer
    mutable ggml_backend_buffer_t const_buf = nullptr;  // GPU storage for the above

    // Persistent GPU tensor holding the action latent across denoising steps.
    // Lives in const_buf (alongside tc_text/tc_pos/tc_vid_*) so gallocr skips it.
    mutable ggml_tensor *   tc_x_raw     = nullptr;  // action latent [action_dim, action_seq]

    // Per-step time embedding tensors (one per step, all in const_buf).
    // Initialized before graph build with pre-computed sinusoidal embeddings.
    static constexpr int MAX_DENOISE_STEPS = 64;
    mutable ggml_tensor * tc_time_emb[MAX_DENOISE_STEPS] = {};  // [freq_dim] per step

    // Unrolled 20-step denoising graph (all steps in one compute).
    // tc_x_raw is the only mutable tensor — updated in-graph via ggml_cpy each step.
    // All time embeddings and delta values are embedded as GPU constants.
    mutable ggml_cgraph *   static_denoise_graph = nullptr;  // built once per predict()
    mutable bool            denoise_graph_valid  = false;    // invalidated when prompt changes

    // Persistent video prefill graph.
    // Built once per (vid_seq, text_seq) shape; only inputs re-uploaded when text changes.
    // Avoids the O(n_nodes) CPU overhead of ggml_init+graph_build+gallocr_alloc per text change.
    mutable ggml_context *  ctx_vid_prefill      = nullptr;  // host metadata for prefill graph
    mutable ggml_cgraph *   static_prefill_graph = nullptr;  // persistent prefill graph
    mutable ggml_gallocr_t  prefill_galloc       = nullptr;  // separate gallocr for prefill graph
    mutable ggml_tensor *   vp_t_vtoken         = nullptr;  // set_input: video tokens
    mutable ggml_tensor *   vp_t_tctx           = nullptr;  // set_input: text context (re-uploaded per text change)
    mutable ggml_tensor *   vp_t_tmod           = nullptr;  // set_input: time modulation (constant after 1st run)
    mutable ggml_tensor *   vp_t_pos            = nullptr;  // set_input: position ids (constant)
    mutable ggml_tensor *   vp_fused_k[8] = {};             // set_output: fused K per cond layer
    mutable ggml_tensor *   vp_fused_v[8] = {};             // set_output: fused V per cond layer
    mutable int  cached_vp_vid_seq = -1;                    // shape keys for static_prefill_graph
    mutable int  cached_vp_text_seq = -1;

    // Cache keys for const_buf / vid_kv invalidation.
    // When text_seq and vid_seq are unchanged, only tc_text is re-uploaded (state-dependent).
    // When text content changes, vid_kv tensors are re-uploaded without reallocating const_buf.
    // const_buf is fully reallocated only when sequence lengths change (shape mismatch).
    mutable std::vector<float> cached_text_ctx_f32;  // last text_ctx used for vid_kv upload
    mutable int  cached_text_seq = -1;               // text_seq that const_buf was sized for
    mutable int  cached_vid_seq  = -1;               // vid_seq that const_buf was sized for

    // Cached video time modulation (vid_t_mod for t=0).
    // vid_t_mod is purely a function of sinusoidal_emb(t=0) and model weights — does NOT depend on
    // text or state. Computed once on first vid_kv_stale call, reused on all subsequent calls.
    // Shape: [6 * video_hidden * vid_seq] (for vid_seq=1: [6 * video_hidden])
    mutable std::vector<float> cached_vid_t_mod;     // precomputed t=0 time modulation for video

    // Persistent proprio_enc graph (state → video_text_dim linear).
    // Rebuilds only when state_dim changes (effectively never).
    mutable ggml_context *  ctx_proprio       = nullptr;
    mutable ggml_cgraph *   proprio_graph     = nullptr;
    mutable ggml_gallocr_t  proprio_galloc    = nullptr;
    mutable ggml_tensor *   pp_prop_in        = nullptr;  // set_input: [state_dim, 1]
    mutable ggml_tensor *   pp_prop_out       = nullptr;  // set_output: [video_text_dim, 1]
    mutable int  cached_pp_state_dim = -1;

    // Persistent act_text_emb graph (text_ext → action_hidden linear+gelu+linear).
    // Rebuilds only when final_text_seq changes (effectively never).
    mutable ggml_context *  ctx_textemb       = nullptr;
    mutable ggml_cgraph *   textemb_graph     = nullptr;
    mutable ggml_gallocr_t  textemb_galloc    = nullptr;
    mutable ggml_tensor *   te_text_in        = nullptr;  // set_input: [video_text_dim, ext_text_seq]
    mutable ggml_tensor *   te_text_out       = nullptr;  // set_output: [action_hidden, ext_text_seq]
    mutable int  cached_te_text_seq = -1;

    std::vector<float> state_mean, state_std, action_mean, action_std;
    std::mt19937       rng{ std::random_device{}() };
};

// ---------------------------------------------------------------------------
// ggml graph helpers
// ---------------------------------------------------------------------------

// Weighted matmul with F32 precision
static inline ggml_tensor * fw_mm(ggml_context * ctx, ggml_tensor * W, ggml_tensor * x) {
    ggml_tensor * r = ggml_mul_mat(ctx, W, x);
    ggml_mul_mat_set_prec(r, GGML_PREC_F32);
    return r;
}

// Linear projection: y = W x + b  (W: [out, in], x: [in, seq])
static inline ggml_tensor * fw_linear(ggml_context * ctx, ggml_tensor * W, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = fw_mm(ctx, W, x);           // [out, seq]
    if (b) {
        ggml_tensor * bv = ggml_reshape_2d(ctx, b, b->ne[0], 1);
        y = ggml_add(ctx, y, bv);
    }
    return y;
}

// affine LayerNorm: norm(x) * w + b  (w/b: [hidden], x: [hidden, seq])
static inline ggml_tensor * fw_ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                                   float eps = 1e-6f) {
    ggml_tensor * n = ggml_norm(ctx, x, eps);
    ggml_tensor * s = ggml_mul(ctx, n, ggml_reshape_2d(ctx, w, w->ne[0], 1));
    return ggml_add(ctx, s, ggml_reshape_2d(ctx, b, b->ne[0], 1));
}

// RMSNorm:  rms_norm(x) * w  (w: [hidden], x: [hidden, seq])
static inline ggml_tensor * fw_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps = 1e-6f) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), ggml_reshape_2d(ctx, w, w->ne[0], 1));
}

// DiT modulate per-token:
//   x:     [hidden, seq]
//   shift: [hidden, seq]  (already sliced from t_mod)
//   scale: [hidden, seq]
// out: x * (1 + scale) + shift
static inline ggml_tensor * fw_modulate(ggml_context * ctx, ggml_tensor * x,
                                         ggml_tensor * shift, ggml_tensor * scale) {
    ggml_tensor * scaled = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    return ggml_add(ctx, scaled, shift);
}

// RoPE for a [head_dim, n_heads, seq] tensor using integer position ids
static inline ggml_tensor * fw_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos,
                                     int n_dims, float freq_base) {
    return ggml_rope_ext(ctx, x, pos, nullptr, n_dims,
                         GGML_ROPE_TYPE_NEOX, 0, freq_base,
                         1.f, 0.f, 1.f, 32.f, 1.f);
}

// Flash attention.
// Inputs in [head_dim, n_heads, seq] layout.
// Permutes to [head_dim, seq, n_heads] (ggml flash_attn_ext format) internally.
// Output: [head_dim, n_heads, seq_q] contiguous — ready for reshape_2d to [nh*hd, seq_q].
static inline ggml_tensor * fw_flash_attn(ggml_context * ctx,
                                           ggml_tensor * q,   // [hd, nh, seq_q]
                                           ggml_tensor * k,   // [hd, nhkv, seq_k]
                                           ggml_tensor * v,   // [hd, nhkv, seq_k]
                                           float scale) {
    // ggml flash_attn_ext expects [head_dim, seq, n_heads, batch] (ne[1]=seq, ne[2]=nh)
    ggml_tensor * Q  = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [hd, seq_q, nh]
    ggml_tensor * K  = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd, seq_k, nhkv]
    ggml_tensor * V  = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [hd, seq_k, nhkv]
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    // fa output: {v.ne[0], Q.ne[2], Q.ne[1], Q.ne[3]} = [hd, nh, seq_q, 1] contiguous
    return fa;  // callers reshape_2d(fa, nh*hd, seq_q)
}

// Sinusoidal time embedding  [1] → [freq_dim]
// Matches Python sinusoidal_embedding_1d: freq_i = t * 10000^(-i/half)
// Output order: [cos(freq_0..half-1), sin(freq_0..half-1)]  (same as Python)
static std::vector<float> sinusoidal_emb_host(float t, int freq_dim,
                                               float min_period = 0.0001f,
                                               float max_period = 1.0f) {
    const int    half = freq_dim / 2;
    const double log_ratio = std::log((double) max_period / (double) min_period);
    std::vector<float> out(freq_dim);
    for (int i = 0; i < half; ++i) {
        const double freq = t * std::exp(-log_ratio * i / (double) half);
        out[i]           = (float) std::cos(freq);  // cos first (matches Python)
        out[i + half]    = (float) std::sin(freq);
    }
    return out;
}

// Make integer position ids tensor [seq] for RoPE
static std::vector<int32_t> make_position_ids(int seq) {
    std::vector<int32_t> ids((size_t) seq);
    for (int i = 0; i < seq; ++i) { ids[i] = i; }
    return ids;
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

static bool fw_load_cfg(const gguf_reader_fw & g, FasterWAMCfg & fc) {
    auto u = [&](const char * k) -> int { return (int) g.u32(k); };
    auto f = [&](const char * k) -> float { return g.f32(k); };

    fc.video_hidden     = u("fasterwam.video_hidden");
    fc.video_ffn_dim    = u("fasterwam.video_ffn_dim");
    fc.video_num_heads  = u("fasterwam.video_num_heads");
    fc.video_head_dim   = u("fasterwam.video_head_dim");
    fc.video_num_layers = u("fasterwam.video_num_layers");
    fc.video_in_dim     = u("fasterwam.video_in_dim");
    fc.video_freq_dim   = u("fasterwam.video_freq_dim");
    fc.video_text_dim   = u("fasterwam.video_text_dim");

    fc.action_hidden        = u("fasterwam.action_hidden");
    fc.action_ffn_dim       = u("fasterwam.action_ffn_dim");
    fc.action_num_heads     = u("fasterwam.action_num_heads");
    fc.action_noncond_heads = u("fasterwam.action_noncond_heads");
    fc.action_head_dim      = u("fasterwam.action_head_dim");
    fc.action_num_layers    = u("fasterwam.action_num_layers");
    fc.action_dim           = u("fasterwam.action_dim");
    fc.action_freq_dim      = u("fasterwam.action_freq_dim");
    fc.action_text_dim      = u("fasterwam.action_text_dim");

    fc.num_condition_layers = u("fasterwam.num_condition_layers");
    for (int j = 0; j < fc.num_condition_layers && j < 8; ++j) {
        char k[64];
        std::snprintf(k, sizeof(k), "fasterwam.condition_layer_%d", j);
        fc.condition_layers[j] = u(k);
    }

    fc.proprio_dim         = u("fasterwam.proprio_dim");
    fc.state_dim           = u("fasterwam.state_dim");
    fc.action_horizon      = u("fasterwam.action_horizon");
    fc.num_inference_steps = u("fasterwam.num_inference_steps");
    fc.num_video_frames    = u("fasterwam.num_video_frames");
    fc.infer_shift         = f("fasterwam.infer_shift");
    fc.num_train_timesteps = u("fasterwam.num_train_timesteps");
    return true;
}

static bool fw_fill_vla_cfg(const FasterWAMCfg & fc, Config & cfg) {
    cfg                 = Config{};
    cfg.hidden          = fc.video_hidden;
    cfg.expert_h        = fc.action_hidden;
    cfg.intermediate    = fc.video_ffn_dim;
    cfg.expert_inter    = fc.action_ffn_dim;
    cfg.n_q_heads       = fc.video_num_heads;
    cfg.n_kv_heads      = fc.video_num_heads;
    cfg.head_dim        = fc.video_head_dim;
    cfg.q_full_dim      = fc.video_num_heads * fc.video_head_dim;
    cfg.kv_full_dim     = cfg.q_full_dim;
    cfg.n_layers        = fc.video_num_layers;
    cfg.n_lang          = 0;
    cfg.n_img           = 0;
    cfg.n_state         = 1;
    cfg.n_suffix        = fc.action_horizon;
    cfg.n_prefix        = 0;
    cfg.n_full          = fc.action_horizon;
    cfg.num_steps       = fc.num_inference_steps;
    cfg.max_state_dim   = fc.state_dim;
    cfg.max_action_dim  = fc.action_dim;
    cfg.real_state_dim  = fc.state_dim;
    cfg.real_action_dim = fc.action_dim;
    cfg.norm_eps        = 1e-6f;
    cfg.rms_eps         = 1e-6f;
    cfg.min_period      = 0.01;
    cfg.max_period      = 1.0;
    cfg.rope_n_dims     = fc.action_head_dim;
    cfg.rope_mode       = GGML_ROPE_TYPE_NEOX;
    cfg.rope_freq_base  = 10000.f;
    return true;
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

static bool fw_load_weights(gguf_reader_fw & g, FasterWAMModel & m) {
    const FasterWAMCfg & fc   = m.fw_cfg;
    FasterWAMWeights &   w    = m.w;
    ggml_context *       wctx = m.ctx_weights;
    const ggml_type      wt   = m.wtype;

    std::vector<ggml_tensor *> tensors;  // all tensors to upload later

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "fasterwam: missing weight tensor %s\n", name);
            return nullptr;
        }
        ggml_tensor * t = ggml_new_tensor(wctx, type, GGML_MAX_DIMS, gt->ne);
        ggml_set_name(t, name);
        tensors.push_back(t);
        return t;
    };
    auto mk_mm  = [&](const char * n) { return mk(n, wt); };
    auto mk_f32 = [&](const char * n) { return mk(n, GGML_TYPE_F32); };

    // --- Video DiT front-end ---
    w.vid_patch_emb_w  = mk_mm("video.patch_embed.weight");
    w.vid_patch_emb_b  = mk_f32("video.patch_embed.bias");
    w.vid_time_emb_0w  = mk_mm("video.time_embed.0.weight");
    w.vid_time_emb_0b  = mk_f32("video.time_embed.0.bias");
    w.vid_time_emb_2w  = mk_mm("video.time_embed.2.weight");
    w.vid_time_emb_2b  = mk_f32("video.time_embed.2.bias");
    w.vid_time_proj_w  = mk_mm("video.time_proj.1.weight");
    w.vid_time_proj_b  = mk_f32("video.time_proj.1.bias");
    w.vid_text_emb_0w  = mk_mm("video.text_embed.0.weight");
    w.vid_text_emb_0b  = mk_f32("video.text_embed.0.bias");
    w.vid_text_emb_2w  = mk_mm("video.text_embed.2.weight");
    w.vid_text_emb_2b  = mk_f32("video.text_embed.2.bias");
    w.vid_head_w       = mk_mm("video.head.weight");
    w.vid_head_b       = mk_f32("video.head.bias");
    w.vid_head_mod     = mk_f32("video.head_mod");

    // Video KV fusion logits (8 tensors)
    for (int j = 0; j < fc.num_condition_layers; ++j) {
        char k[64];
        std::snprintf(k, sizeof(k), "video_kv_fusion_logits.%d", j);
        w.vid_kv_logits[j] = mk_f32(k);
    }

    // --- Video DiT blocks ---
    w.video_blocks.resize((size_t) fc.video_num_layers);
    for (int i = 0; i < fc.video_num_layers; ++i) {
        VideoDiTBlockW & b = w.video_blocks[(size_t) i];
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "video.blk.%d.", i);
        std::string p = pfx;

        // Self-attention
        b.sa_q_w    = mk_mm((p + "self_attn.q.weight").c_str());
        b.sa_q_b    = mk_f32((p + "self_attn.q.bias").c_str());
        b.sa_k_w    = mk_mm((p + "self_attn.k.weight").c_str());
        b.sa_k_b    = mk_f32((p + "self_attn.k.bias").c_str());
        b.sa_v_w    = mk_mm((p + "self_attn.v.weight").c_str());
        b.sa_v_b    = mk_f32((p + "self_attn.v.bias").c_str());
        b.sa_o_w    = mk_mm((p + "self_attn.o.weight").c_str());
        b.sa_o_b    = mk_f32((p + "self_attn.o.bias").c_str());
        b.sa_norm_q = mk_f32((p + "self_attn.norm_q.weight").c_str());
        b.sa_norm_k = mk_f32((p + "self_attn.norm_k.weight").c_str());
        // FFN
        b.ffn0_w    = mk_mm((p + "ffn.0.weight").c_str());
        b.ffn0_b    = mk_f32((p + "ffn.0.bias").c_str());
        b.ffn2_w    = mk_mm((p + "ffn.2.weight").c_str());
        b.ffn2_b    = mk_f32((p + "ffn.2.bias").c_str());
        // Modulation
        b.modulation = mk_f32((p + "modulation").c_str());
        // Cross-attention (all video blocks have text cross-attn)
        b.norm3_w   = mk_f32((p + "norm3.weight").c_str());
        b.norm3_b   = mk_f32((p + "norm3.bias").c_str());
        b.ca_q_w    = mk_mm((p + "cross_attn.q.weight").c_str());
        b.ca_q_b    = mk_f32((p + "cross_attn.q.bias").c_str());
        b.ca_k_w    = mk_mm((p + "cross_attn.k.weight").c_str());
        b.ca_k_b    = mk_f32((p + "cross_attn.k.bias").c_str());
        b.ca_v_w    = mk_mm((p + "cross_attn.v.weight").c_str());
        b.ca_v_b    = mk_f32((p + "cross_attn.v.bias").c_str());
        b.ca_o_w    = mk_mm((p + "cross_attn.o.weight").c_str());
        b.ca_o_b    = mk_f32((p + "cross_attn.o.bias").c_str());
        b.ca_norm_q = mk_f32((p + "cross_attn.norm_q.weight").c_str());
        b.ca_norm_k = mk_f32((p + "cross_attn.norm_k.weight").c_str());
    }

    // --- Action DiT front-end ---
    w.act_encoder_w   = mk_mm("action.action_encoder.weight");
    w.act_encoder_b   = mk_f32("action.action_encoder.bias");
    w.act_head_w      = mk_mm("action.head.weight");
    w.act_head_b      = mk_f32("action.head.bias");
    w.act_time_emb_0w = mk_mm("action.time_embed.0.weight");
    w.act_time_emb_0b = mk_f32("action.time_embed.0.bias");
    w.act_time_emb_2w = mk_mm("action.time_embed.2.weight");
    w.act_time_emb_2b = mk_f32("action.time_embed.2.bias");
    w.act_time_proj_w = mk_mm("action.time_proj.1.weight");
    w.act_time_proj_b = mk_f32("action.time_proj.1.bias");
    w.act_text_emb_0w = mk_mm("action.text_embed.0.weight");
    w.act_text_emb_0b = mk_f32("action.text_embed.0.bias");
    w.act_text_emb_2w = mk_mm("action.text_embed.2.weight");
    w.act_text_emb_2b = mk_f32("action.text_embed.2.bias");

    // --- Action DiT blocks ---
    w.action_blocks.resize((size_t) fc.action_num_layers);
    for (int i = 0; i < fc.action_num_layers; ++i) {
        ActionDiTBlockW & b = w.action_blocks[(size_t) i];
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "action.blk.%d.", i);
        std::string p = pfx;

        // Is this block a condition layer?
        b.is_condition = false;
        for (int j = 0; j < fc.num_condition_layers; ++j) {
            if (fc.condition_layers[j] == i) {
                b.is_condition = true;
                break;
            }
        }

        // Self-attention
        b.sa_q_w    = mk_mm((p + "self_attn.q.weight").c_str());
        b.sa_q_b    = mk_f32((p + "self_attn.q.bias").c_str());
        b.sa_k_w    = mk_mm((p + "self_attn.k.weight").c_str());
        b.sa_k_b    = mk_f32((p + "self_attn.k.bias").c_str());
        b.sa_v_w    = mk_mm((p + "self_attn.v.weight").c_str());
        b.sa_v_b    = mk_f32((p + "self_attn.v.bias").c_str());
        b.sa_o_w    = mk_mm((p + "self_attn.o.weight").c_str());
        b.sa_o_b    = mk_f32((p + "self_attn.o.bias").c_str());
        b.sa_norm_q = mk_f32((p + "self_attn.norm_q.weight").c_str());
        b.sa_norm_k = mk_f32((p + "self_attn.norm_k.weight").c_str());
        // FFN
        b.ffn0_w    = mk_mm((p + "ffn.0.weight").c_str());
        b.ffn0_b    = mk_f32((p + "ffn.0.bias").c_str());
        b.ffn2_w    = mk_mm((p + "ffn.2.weight").c_str());
        b.ffn2_b    = mk_f32((p + "ffn.2.bias").c_str());
        // Modulation
        b.modulation = mk_f32((p + "modulation").c_str());

        if (b.is_condition) {
            b.norm3_w   = mk_f32((p + "norm3.weight").c_str());
            b.norm3_b   = mk_f32((p + "norm3.bias").c_str());
            b.ca_q_w    = mk_mm((p + "cross_attn.q.weight").c_str());
            b.ca_q_b    = mk_f32((p + "cross_attn.q.bias").c_str());
            b.ca_k_w    = mk_mm((p + "cross_attn.k.weight").c_str());
            b.ca_k_b    = mk_f32((p + "cross_attn.k.bias").c_str());
            b.ca_v_w    = mk_mm((p + "cross_attn.v.weight").c_str());
            b.ca_v_b    = mk_f32((p + "cross_attn.v.bias").c_str());
            b.ca_o_w    = mk_mm((p + "cross_attn.o.weight").c_str());
            b.ca_o_b    = mk_f32((p + "cross_attn.o.bias").c_str());
            b.ca_norm_q = mk_f32((p + "cross_attn.norm_q.weight").c_str());
            b.ca_norm_k = mk_f32((p + "cross_attn.norm_k.weight").c_str());
        }
    }

    // --- Proprio encoder ---
    w.proprio_enc_w = mk_mm("proprio_encoder.weight");
    w.proprio_enc_b = mk_f32("proprio_encoder.bias");

    // Allocate GPU/CPU buffer and upload
    m.weight_buf = ggml_backend_alloc_ctx_tensors(wctx, m.backend);
    if (!m.weight_buf) {
        std::fprintf(stderr, "fasterwam: failed to allocate weight buffer\n");
        return false;
    }
    for (ggml_tensor * t : tensors) {
        if (!t) {
            // already printed error above
            return false;
        }
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "fasterwam: upload size mismatch for %s (%zu vs %zu)\n",
                         t->name, bytes.size(), ggml_nbytes(t));
            return false;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    return true;
}

// ---------------------------------------------------------------------------
// Normalization stats loading
// ---------------------------------------------------------------------------

static bool fw_load_stats(gguf_reader_fw & g, FasterWAMModel & m) {
    auto read = [&](const char * name, std::vector<float> & dst, int64_t n, float fill) {
        dst.assign((size_t) n, fill);
        std::vector<float> tmp;
        if (!g.read_to_f32(name, tmp)) {
            return;
        }
        for (size_t i = 0; i < dst.size() && i < tmp.size(); ++i) {
            dst[i] = tmp[i];
        }
    };
    const int sd = m.fw_cfg.state_dim;
    const int ah = m.fw_cfg.action_horizon;
    const int ad = m.fw_cfg.action_dim;
    read("norm.state_mean",  m.state_mean,  sd,      0.f);
    read("norm.state_std",   m.state_std,   sd,      1.f);
    // action stats shape [action_horizon, action_dim], flat row-major
    m.action_mean.assign((size_t) ah * (size_t) ad, 0.f);
    m.action_std.assign((size_t) ah * (size_t) ad, 1.f);
    std::vector<float> tmp;
    if (g.read_to_f32("norm.action_mean", tmp) && (int) tmp.size() >= ah * ad) {
        m.action_mean = std::vector<float>(tmp.begin(), tmp.begin() + (size_t) ah * (size_t) ad);
    }
    if (g.read_to_f32("norm.action_std", tmp) && (int) tmp.size() >= ah * ad) {
        m.action_std = std::vector<float>(tmp.begin(), tmp.begin() + (size_t) ah * (size_t) ad);
    }
    return true;
}

// ---------------------------------------------------------------------------
// ggml graph-based inference
// ---------------------------------------------------------------------------

// Softmax of a length-n float array in-place (used for KV fusion on host)
static void softmax_inplace(float * x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) { max_val = x[i]; }
    }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i) { x[i] /= sum; }
}

// Flow-matching timestep schedule (WanContinuousFlow with infer_shift)
// Returns the N+1 boundary values in [0, num_train_timesteps]
static std::vector<float> make_timestep_schedule(int num_steps, float infer_shift,
                                                  int num_train_timesteps) {
    std::vector<float> schedule(num_steps + 1);
    for (int i = 0; i <= num_steps; ++i) {
        const float t_lin   = (float) i / (float) num_steps;
        const float t_shift = infer_shift * t_lin / (1.f + (infer_shift - 1.f) * t_lin);
        schedule[i] = t_shift * (float) num_train_timesteps;
    }
    return schedule;
}

// ---------------------------------------------------------------------------
// Video prefill helpers
// ---------------------------------------------------------------------------

// Precompute 3D M-RoPE position tensor for video tokens on host.
// Returns a flat int32 vector [vid_seq * 4] where each 4-tuple is [f, h, w, 0].
// ggml_rope_multi MROPE mode expects b as 1D vector of length 4*vid_seq.
static std::vector<int32_t> make_video_pos_ids(int num_frames, int grid_h, int grid_w) {
    const int vid_seq = num_frames * grid_h * grid_w;
    std::vector<int32_t> ids((size_t) vid_seq * 4);
    int idx = 0;
    for (int f = 0; f < num_frames; ++f) {
        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                ids[(size_t) idx * 4 + 0] = f;  // temporal
                ids[(size_t) idx * 4 + 1] = h;  // height
                ids[(size_t) idx * 4 + 2] = w;  // width
                ids[(size_t) idx * 4 + 3] = 0;  // padding (GGML_MROPE_SECTIONS = 4)
                ++idx;
            }
        }
    }
    return ids;
}

// M-RoPE for video: sections = [f_dims, h_dims, w_dims, 0]
// positions: [4*vid_seq] int32 flat (4 values per token: F, H, W, 0)
// x: [head_dim, n_heads, vid_seq]
static inline ggml_tensor * fw_rope_mrope(ggml_context * ctx, ggml_tensor * x,
                                           ggml_tensor * pos,  // [4*vid_seq] i32 (1D)
                                           int head_dim, int f_dims, int h_dims, int w_dims) {
    int sections[GGML_MROPE_SECTIONS] = { f_dims, h_dims, w_dims, 0 };
    const int n_dims = f_dims + h_dims + w_dims;
    return ggml_rope_multi(ctx, x, pos, nullptr,
                           n_dims, sections, GGML_ROPE_TYPE_MROPE,
                           0, 10000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
    (void) head_dim;
}

// ---------------------------------------------------------------------------
// ggml graph: one video DiT block
//
// x:        [video_hidden, vid_seq]    video token sequence
// t_mod:    [6*video_hidden, vid_seq]  per-token modulation from time embed
// pos_ids:  [3, vid_seq] int32         M-RoPE position indices (F/H/W)
// text_ctx: [video_text_dim, text_seq] projected text context
//
// Returns:
//   out_x:   updated x [video_hidden, vid_seq]
//   out_k:   key BEFORE RoPE — NOT used for cache (we cache post-RoPE K below)
//   out_k_rope: key AFTER RoPE [head_dim, n_heads, vid_seq] — cached for KV fusion
//   out_v:   value [head_dim, n_heads, vid_seq]
//
// Mirrors Python DiTBlock.forward() with seperated_timestep=True (per-token t_mod).
// ---------------------------------------------------------------------------
struct VideoDiTBlockOut {
    ggml_tensor * x   = nullptr;  // updated hidden [hd, vid_seq]
    ggml_tensor * k   = nullptr;  // post-RoPE K [hd_q, nh, vid_seq] (for KV fusion cache)
    ggml_tensor * v   = nullptr;  // V [hd_q, nh, vid_seq]
};

static VideoDiTBlockOut build_video_dit_block(
    ggml_context *         ctx,
    const VideoDiTBlockW & bw,
    ggml_tensor *          x,         // [hd, vid_seq]
    ggml_tensor *          t_mod,     // [6*hd, vid_seq]  per-token
    ggml_tensor *          pos_ids,   // [3, vid_seq] int32
    ggml_tensor *          text_ctx,  // [text_hd, text_seq]
    const FasterWAMCfg &   fc
) {
    const int hd      = fc.video_hidden;
    const int nh      = fc.video_num_heads;
    const int hd_q    = fc.video_head_dim;
    const int vid_seq = (int) x->ne[1];
    const float eps   = 1e-6f;
    const float scale = 1.f / std::sqrt((float) hd_q);

    // M-RoPE sections: head_dim=128 → f=43+1=44, h=42, w=42 (≈ 128/3)
    // Python: f_dims = head_dim - 2*(head_dim//3), h_dims=w_dims=head_dim//3
    const int h_dims = hd_q / 3;
    const int w_dims = hd_q / 3;
    const int f_dims = hd_q - 2 * (hd_q / 3);

    // --- Per-token t_mod: [6*hd, vid_seq] → 6 slices of [hd, vid_seq] ---
    // Layout: t_mod[0..hd-1,i]=shift_msa, [hd..2hd-1,i]=scale_msa, etc.
    ggml_tensor * shift_msa = ggml_view_2d(ctx, t_mod, hd, vid_seq, hd * 6 * sizeof(float), (size_t)(0 * hd) * sizeof(float));
    ggml_tensor * scale_msa = ggml_view_2d(ctx, t_mod, hd, vid_seq, hd * 6 * sizeof(float), (size_t)(1 * hd) * sizeof(float));
    ggml_tensor * gate_msa  = ggml_view_2d(ctx, t_mod, hd, vid_seq, hd * 6 * sizeof(float), (size_t)(2 * hd) * sizeof(float));
    ggml_tensor * shift_mlp = ggml_view_2d(ctx, t_mod, hd, vid_seq, hd * 6 * sizeof(float), (size_t)(3 * hd) * sizeof(float));
    ggml_tensor * scale_mlp = ggml_view_2d(ctx, t_mod, hd, vid_seq, hd * 6 * sizeof(float), (size_t)(4 * hd) * sizeof(float));
    ggml_tensor * gate_mlp  = ggml_view_2d(ctx, t_mod, hd, vid_seq, hd * 6 * sizeof(float), (size_t)(5 * hd) * sizeof(float));

    // --- 1. norm1 (no-affine) + modulate → Q/K/V for self-attention ---
    ggml_tensor * x_ln = ggml_norm(ctx, x, eps);      // [hd, vid_seq]
    x_ln = ggml_add(ctx, ggml_mul(ctx, x_ln, scale_msa), shift_msa);   // [hd, vid_seq]

    // Q proj → RMSNorm → M-RoPE
    ggml_tensor * q  = fw_linear(ctx, bw.sa_q_w, bw.sa_q_b, x_ln);    // [nh*hd_q, vid_seq]
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, hd_q, nh, vid_seq);     // [hd_q, nh, vid_seq]
    ggml_tensor * qr = ggml_mul(ctx, ggml_rms_norm(ctx, qh, eps),
                                 ggml_reshape_3d(ctx, bw.sa_norm_q, hd_q, nh, 1));
    qr = fw_rope_mrope(ctx, qr, pos_ids, hd_q, f_dims, h_dims, w_dims);

    // K proj → RMSNorm → M-RoPE
    ggml_tensor * k  = fw_linear(ctx, bw.sa_k_w, bw.sa_k_b, x_ln);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, hd_q, nh, vid_seq);
    ggml_tensor * kr = ggml_mul(ctx, ggml_rms_norm(ctx, kh, eps),
                                 ggml_reshape_3d(ctx, bw.sa_norm_k, hd_q, nh, 1));
    kr = fw_rope_mrope(ctx, kr, pos_ids, hd_q, f_dims, h_dims, w_dims);

    // V proj (no norm/rope)
    ggml_tensor * v  = fw_linear(ctx, bw.sa_v_w, bw.sa_v_b, x_ln);    // [nh*hd_q, vid_seq]
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, hd_q, nh, vid_seq);

    // --- 2. Self-attention ---
    ggml_tensor * fa      = fw_flash_attn(ctx, qr, kr, vh, scale);     // [hd_q, nh, vid_seq]
    ggml_tensor * att_flat = ggml_reshape_2d(ctx, fa, nh * hd_q, vid_seq);

    // --- 3. Output projection + gate residual ---
    ggml_tensor * sa_out = fw_linear(ctx, bw.sa_o_w, bw.sa_o_b, att_flat);  // [hd, vid_seq]
    x = ggml_add(ctx, x, ggml_mul(ctx, sa_out, gate_msa));

    // --- 4. Cross-attention to text context (all video blocks) ---
    {
        const int n_heads_ca = nh;
        const int text_seq   = (int) text_ctx->ne[1];

        // norm3 (affine LayerNorm)
        ggml_tensor * x_ln3 = fw_ln(ctx, x, bw.norm3_w, bw.norm3_b, eps);  // [hd, vid_seq]

        // Q from video tokens
        ggml_tensor * ca_q  = fw_linear(ctx, bw.ca_q_w, bw.ca_q_b, x_ln3);    // [n_heads_ca*hd_q, vid_seq]
        ggml_tensor * ca_qh = ggml_reshape_3d(ctx, ca_q, hd_q, n_heads_ca, vid_seq);
        ggml_tensor * ca_qr = ggml_mul(ctx, ggml_rms_norm(ctx, ca_qh, eps),
                                        ggml_reshape_3d(ctx, bw.ca_norm_q, hd_q, n_heads_ca, 1));

        // K, V from text context
        ggml_tensor * ca_k  = fw_linear(ctx, bw.ca_k_w, bw.ca_k_b, text_ctx);  // [n_heads_ca*hd_q, text_seq]
        ggml_tensor * ca_kh = ggml_reshape_3d(ctx, ca_k, hd_q, n_heads_ca, text_seq);
        ggml_tensor * ca_kr = ggml_mul(ctx, ggml_rms_norm(ctx, ca_kh, eps),
                                        ggml_reshape_3d(ctx, bw.ca_norm_k, hd_q, n_heads_ca, 1));

        ggml_tensor * ca_v  = fw_linear(ctx, bw.ca_v_w, bw.ca_v_b, text_ctx);
        ggml_tensor * ca_vh = ggml_reshape_3d(ctx, ca_v, hd_q, n_heads_ca, text_seq);

        ggml_tensor * ca_fa   = fw_flash_attn(ctx, ca_qr, ca_kr, ca_vh, scale);  // [hd_q, n_heads_ca, vid_seq]
        ggml_tensor * ca_flat = ggml_reshape_2d(ctx, ca_fa, n_heads_ca * hd_q, vid_seq);
        ggml_tensor * ca_out  = fw_linear(ctx, bw.ca_o_w, bw.ca_o_b, ca_flat);
        x = ggml_add(ctx, x, ca_out);
    }

    // --- 5. FFN: norm2 + modulate → FFN → gate residual ---
    ggml_tensor * x_ln_ffn = ggml_norm(ctx, x, eps);
    x_ln_ffn = ggml_add(ctx, ggml_mul(ctx, x_ln_ffn, scale_mlp), shift_mlp);

    ggml_tensor * ffn_up   = fw_linear(ctx, bw.ffn0_w, bw.ffn0_b, x_ln_ffn);
    ggml_tensor * ffn_act  = ggml_gelu(ctx, ffn_up);
    ggml_tensor * ffn_down = fw_linear(ctx, bw.ffn2_w, bw.ffn2_b, ffn_act);

    x = ggml_add(ctx, x, ggml_mul(ctx, ffn_down, gate_mlp));

    // Make kr and vh contiguous so they can be set as graph outputs and read back.
    ggml_tensor * k_cont = ggml_cont(ctx, kr);
    ggml_tensor * v_cont = ggml_cont(ctx, vh);

    VideoDiTBlockOut out;
    out.x = x;
    out.k = k_cont;
    out.v = v_cont;
    return out;
}

// ---------------------------------------------------------------------------
// prefill_video_cache_gpu:
//   Runs video DiT blocks up to the last condition layer (inclusive) on GPU,
//   accumulates interval K/V, fuses at 8 condition layers via softmax(logits)-weighted sum.
//   Layers beyond the last condition layer are skipped (their x output is unused).
//   Returns 8 fused KV pairs [{k:[hd_q,nh,vid_seq], v:[hd_q,nh,vid_seq]}]
//   as host float vectors.
//
// Input video latent: zeros (vid_seq=patch-embedded video tokens, BF16/F32)
// For the smoke test (no real video), we use all-zero video latent with
// vid_seq = 1 (no patch embedding run).
// When real video frames are provided, compute actual vid_seq and run conv3d.
// ---------------------------------------------------------------------------
struct VideoKVPair {
    std::vector<float> k;  // [hd_q * nh * vid_seq]
    std::vector<float> v;
};

// Build and execute the video prefill graph.
// video_tokens_f32: [video_hidden, vid_seq] (already patch-embedded + text/time conditioned)
// Returns 8 fused KV pairs.
static std::vector<VideoKVPair> prefill_video_cache_gpu(
    ggml_backend_t           backend,
    ggml_gallocr_t           galloc,
    const FasterWAMWeights & wts,
    const FasterWAMCfg &     fc,
    const std::vector<float>& video_tokens_f32,  // [vid_hidden * vid_seq]
    const std::vector<float>& text_ctx_f32,       // [video_text_dim * text_seq]
    const std::vector<float>& t_mod_f32,          // [6*vid_hidden * vid_seq]  per-token
    const std::vector<int32_t>& pos_ids_host,     // [3 * vid_seq]
    int                       vid_seq,
    int                       text_seq
) {
    const int hd    = fc.video_hidden;
    const int nh    = fc.video_num_heads;
    const int hd_q  = fc.video_head_dim;
    const int ncl   = fc.num_condition_layers;

    // Build condition_intervals (like Python _build_condition_intervals)
    // condition_layers = [0,4,8,12,16,20,24,28]
    // intervals[j] = layers from prev_cond_layer+1 to condition_layer[j] inclusive
    // interval sizes: [1,4,4,4,4,4,4,4]
    std::vector<std::vector<int>> intervals(ncl);
    int prev = -1;
    for (int j = 0; j < ncl; ++j) {
        const int cl = fc.condition_layers[j];
        for (int li = prev + 1; li <= cl; ++li) {
            intervals[j].push_back(li);
        }
        prev = cl;
    }

    // We need one graph per layer (accumulate kv_interval across layers, fuse at condition layers).
    // To keep it simple: build a graph that processes ALL 30 layers in one pass,
    // outputs 8*2 fused KV tensors (mark as ggml_set_output).
    // Since interval fusion (softmax + weighted sum) is small enough to do on GPU too,
    // we embed the fusion inside the graph for condition layers.

    const ggml_init_params p{ (size_t) 512 * 1024 * 1024, nullptr, true };
    ggml_context * vctx = ggml_init(p);
    if (!vctx) { return {}; }

    // Inputs
    ggml_tensor * t_vtoken = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, hd, vid_seq);
    ggml_set_name(t_vtoken, "vid_tokens");
    ggml_set_input(t_vtoken);

    ggml_tensor * t_tctx = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, fc.video_text_dim, text_seq);
    ggml_set_name(t_tctx, "vid_text_ctx");
    ggml_set_input(t_tctx);

    // t_mod per-token: [6*hd, vid_seq]  — stored as [6*hd, vid_seq] in ggml
    ggml_tensor * t_tmod = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, hd * 6, vid_seq);
    ggml_set_name(t_tmod, "vid_t_mod");
    ggml_set_input(t_tmod);

    // M-RoPE positions: [4*vid_seq] int32 (1D flat: 4 values per token)
    ggml_tensor * t_pos = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, (int64_t) vid_seq * 4);
    ggml_set_name(t_pos, "vid_pos");
    ggml_set_input(t_pos);

    // Text embedding for video: Linear(text_dim→vid_hidden)+GELU+Linear
    ggml_tensor * text_emb = fw_linear(vctx, wts.vid_text_emb_0w, wts.vid_text_emb_0b, t_tctx);
    text_emb = ggml_gelu(vctx, text_emb);
    text_emb = fw_linear(vctx, wts.vid_text_emb_2w, wts.vid_text_emb_2b, text_emb);  // [hid, text_seq]

    // Run 30 video DiT blocks.
    // For KV fusion: accumulate interval_kv (k/v per layer in interval),
    // at each condition layer j fuse using vid_kv_logits[j].
    // Fused KV for condition layer j: sum_i(softmax(logits[j])[i] * interval_kv[i].k)
    // We track this as ggml ops: for each interval of size n,
    //   weights = softmax(logits[j])  (n scalars, host-computed from GPU weights)
    //   fused_k = sum_i(weights[i] * k_i)
    // Since logits are small (≤4 elements), we scale each k_i by a scalar constant.

    // Helper: fuse interval K or V vectors with learned weights.
    // We compute softmax of logits on host (since logits are tiny learnable params),
    // then construct a weighted sum graph node.
    // kv_list: list of ggml tensors [hd_q, nh, vid_seq]
    // logits_tensor: ggml_tensor [n] (F32 weights from GGUF)
    // Since logits are constant weights (not input-dependent), we read them off
    // GPU once at build time ... but we can't read from GPU during graph build.
    // Instead: embed the fusion directly using ggml_scale + ggml_add in the graph.
    // We read the logit VALUES from the CPU-side weight metadata during graph construction,
    // compute softmax on host, and use ggml_scale(k_i, w_i) + accumulate.

    // Read logit values from GPU tensors to host once (they are constant weights).
    // This is a one-time cost per predict() call.
    std::vector<std::vector<float>> logits_host(ncl);
    for (int j = 0; j < ncl; ++j) {
        int n = (int) intervals[j].size();
        logits_host[j].resize((size_t) n);
        ggml_backend_tensor_get(wts.vid_kv_logits[j],
                                logits_host[j].data(), 0,
                                (size_t) n * sizeof(float));
        softmax_inplace(logits_host[j].data(), n);
    }

    ggml_tensor * x = t_vtoken;  // [hd, vid_seq]

    // interval_k/v: K and V tensors from each layer in the current interval
    std::vector<ggml_tensor *> interval_k, interval_v;
    int cur_interval_j = 0;  // which condition layer we're building toward

    // fused_kv outputs (pointers to graph output tensors)
    std::vector<ggml_tensor *> fused_k_out(ncl, nullptr);
    std::vector<ggml_tensor *> fused_v_out(ncl, nullptr);

    // Only run up to and including the last condition layer.
    // Layers beyond fc.condition_layers[ncl-1] contribute no KV output and can be skipped.
    const int last_cond_layer = fc.condition_layers[ncl - 1];
    for (int li = 0; li <= last_cond_layer; ++li) {
        const VideoDiTBlockW & bw = wts.video_blocks[(size_t) li];
        VideoDiTBlockOut blk = build_video_dit_block(vctx, bw, x, t_tmod, t_pos, text_emb, fc);
        x = blk.x;
        interval_k.push_back(blk.k);
        interval_v.push_back(blk.v);

        // Check if this layer is a condition layer
        bool is_cond = false;
        for (int j = 0; j < ncl; ++j) {
            if (fc.condition_layers[j] == li) {
                is_cond = true;
                cur_interval_j = j;
                break;
            }
        }

        if (is_cond) {
            // Fuse: fused_k = sum_i(w_i * interval_k[i])
            const std::vector<float> & wghts = logits_host[cur_interval_j];
            const int n_interval = (int) interval_k.size();

            ggml_tensor * fk = ggml_scale(vctx, interval_k[0], wghts[0]);
            ggml_tensor * fv = ggml_scale(vctx, interval_v[0], wghts[0]);
            for (int ii = 1; ii < n_interval; ++ii) {
                fk = ggml_add(vctx, fk, ggml_scale(vctx, interval_k[(size_t)ii], wghts[(size_t)ii]));
                fv = ggml_add(vctx, fv, ggml_scale(vctx, interval_v[(size_t)ii], wghts[(size_t)ii]));
            }
            // Mark as outputs
            fk = ggml_cont(vctx, fk);
            fv = ggml_cont(vctx, fv);
            ggml_set_output(fk);
            ggml_set_output(fv);
            fused_k_out[cur_interval_j] = fk;
            fused_v_out[cur_interval_j] = fv;

            // Reset interval for next segment
            interval_k.clear();
            interval_v.clear();
        }
    }

    // Build the graph with all fused KV as outputs.
    ggml_cgraph * vg = ggml_new_graph_custom(vctx, 65536, false);
    for (int j = 0; j < ncl; ++j) {
        if (fused_k_out[j]) { ggml_build_forward_expand(vg, fused_k_out[j]); }
        if (fused_v_out[j]) { ggml_build_forward_expand(vg, fused_v_out[j]); }
    }

    if (!ggml_gallocr_alloc_graph(galloc, vg)) {
        std::fprintf(stderr, "fasterwam: video prefill galloc alloc failed\n");
        ggml_free(vctx);
        return {};
    }

    // Upload inputs (async: same stream as graph_compute, ordering guaranteed)
    ggml_backend_tensor_set_async(backend, t_vtoken, video_tokens_f32.data(), 0, ggml_nbytes(t_vtoken));
    ggml_backend_tensor_set_async(backend, t_tctx,  text_ctx_f32.data(),     0, ggml_nbytes(t_tctx));
    ggml_backend_tensor_set_async(backend, t_tmod,  t_mod_f32.data(),         0, ggml_nbytes(t_tmod));
    ggml_backend_tensor_set_async(backend, t_pos,   pos_ids_host.data(),      0, ggml_nbytes(t_pos));

    ggml_backend_graph_compute(backend, vg);

    // Read back fused KV
    const size_t kv_elem = (size_t) hd_q * (size_t) nh * (size_t) vid_seq;
    std::vector<VideoKVPair> result((size_t) ncl);
    for (int j = 0; j < ncl; ++j) {
        result[j].k.resize(kv_elem);
        result[j].v.resize(kv_elem);
        if (fused_k_out[j] && fused_v_out[j]) {
            ggml_backend_tensor_get(fused_k_out[j], result[j].k.data(), 0, kv_elem * sizeof(float));
            ggml_backend_tensor_get(fused_v_out[j], result[j].v.data(), 0, kv_elem * sizeof(float));
        }
    }

    ggml_free(vctx);
    return result;
}

// ---------------------------------------------------------------------------
// ggml graph: one action DiT block
//
// x:       [action_hidden, action_seq]
// t_mod:   [6*action_hidden] (global modulation from time embed)
// vid_k/v: [head_dim, n_heads, video_seq] or nullptr (condition layers only)
// returns updated x
// ---------------------------------------------------------------------------
// build_action_dit_block:
//   x:       [hd, seq]          action token sequence
//   t_mod:   [6*hd, 1]          global time modulation (before adding block.modulation)
//   pos_ids: [seq] int32        position ids for RoPE
//   vid_k:   [hd_q, nh_ca, vid_seq]  pre-fused video K (condition layers only, else nullptr)
//   vid_v:   [hd_q, nh_ca, vid_seq]  pre-fused video V
//   text_ctx: [text_hidden, text_seq]  text context for cross-attn (condition layers only)
//
// Mirrors Python SparseActionDiT/SparseMoT forward:
//   Condition block (DiTBlock):
//     1. modulation = (block.modulation + t_mod).chunk(6)
//     2. q,k,v from norm1(x)+modulate → sa projections → RMSNorm → RoPE
//     3. Mixed attn: cat([vid_k, k_action], seq) / cat([vid_v, v_action], seq), 24 heads
//     4. x = gate(x, gate_msa, sa.o(mixed_out))
//     5. x += cross_attn(norm3(x), text_ctx) using 24 heads
//     6. x = gate(x, gate_mlp, ffn(modulate(norm2(x), shift_mlp, scale_mlp)))
//   Non-condition block (ActionOnlyDiTBlock):
//     1. modulation = (block.modulation + t_mod).chunk(6)
//     2-4. same but self-attn only (action tokens), using noncond_heads
//     5. (no cross-attn)
//     6. FFN same
static ggml_tensor * build_action_dit_block(
    ggml_context *         ctx,
    const ActionDiTBlockW & bw,
    ggml_tensor *          x,         // [hd, seq]
    ggml_tensor *          t_mod,     // [6*hd, 1]  (global, WITHOUT block.modulation)
    ggml_tensor *          pos_ids,   // [seq] int32
    ggml_tensor *          vid_k,     // [hd_q, nh_ca, vid_seq] or nullptr
    ggml_tensor *          vid_v,
    ggml_tensor *          text_ctx,  // [text_hd, text_seq] or nullptr
    const FasterWAMCfg &   fc
) {
    const int hd      = fc.action_hidden;
    const int nh      = bw.is_condition ? fc.action_num_heads : fc.action_noncond_heads;
    const int hd_q    = fc.action_head_dim;
    const int seq     = fc.action_horizon;
    const float eps   = 1e-6f;
    const float scale = 1.f / std::sqrt((float) hd_q);

    // --- Add block.modulation to global t_mod, then slice into 6 vectors ---
    // bw.modulation: [hd, 6, 1] in ggml = reshaped to [6*hd, 1]
    ggml_tensor * mod_flat   = ggml_reshape_2d(ctx, bw.modulation, hd * 6, 1);
    ggml_tensor * t_mod_blk  = ggml_add(ctx, mod_flat, t_mod);   // [6*hd, 1]

    ggml_tensor * shift_msa = ggml_view_2d(ctx, t_mod_blk, hd, 1, hd * sizeof(float), 0 * hd * sizeof(float));
    ggml_tensor * scale_msa = ggml_view_2d(ctx, t_mod_blk, hd, 1, hd * sizeof(float), 1 * hd * sizeof(float));
    ggml_tensor * gate_msa  = ggml_view_2d(ctx, t_mod_blk, hd, 1, hd * sizeof(float), 2 * hd * sizeof(float));
    ggml_tensor * shift_mlp = ggml_view_2d(ctx, t_mod_blk, hd, 1, hd * sizeof(float), 3 * hd * sizeof(float));
    ggml_tensor * scale_mlp = ggml_view_2d(ctx, t_mod_blk, hd, 1, hd * sizeof(float), 4 * hd * sizeof(float));
    ggml_tensor * gate_mlp  = ggml_view_2d(ctx, t_mod_blk, hd, 1, hd * sizeof(float), 5 * hd * sizeof(float));

    // --- 1. Norm1 (no-affine) + modulate → Q/K/V for self-attention ---
    ggml_tensor * x_ln = ggml_norm(ctx, x, eps);
    x_ln = ggml_add(ctx, ggml_mul(ctx, x_ln, scale_msa), shift_msa);   // [hd, seq]

    // Q proj → RMSNorm → RoPE
    ggml_tensor * q  = fw_linear(ctx, bw.sa_q_w, bw.sa_q_b, x_ln);    // [nh*hd_q, seq]
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, hd_q, nh, seq);
    ggml_tensor * qr = ggml_mul(ctx, ggml_rms_norm(ctx, qh, eps),
                                 ggml_reshape_3d(ctx, bw.sa_norm_q, hd_q, nh, 1));
    qr = fw_rope(ctx, qr, pos_ids, hd_q, 10000.f);                     // [hd_q, nh, seq]

    // K proj → RMSNorm → RoPE
    ggml_tensor * k  = fw_linear(ctx, bw.sa_k_w, bw.sa_k_b, x_ln);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, hd_q, nh, seq);
    ggml_tensor * kr = ggml_mul(ctx, ggml_rms_norm(ctx, kh, eps),
                                 ggml_reshape_3d(ctx, bw.sa_norm_k, hd_q, nh, 1));
    kr = fw_rope(ctx, kr, pos_ids, hd_q, 10000.f);

    // V proj (no norm/rope)
    ggml_tensor * v  = fw_linear(ctx, bw.sa_v_w, bw.sa_v_b, x_ln);
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, hd_q, nh, seq);

    // --- 2. Flash attention: for condition blocks, concat video KV with action KV ---
    ggml_tensor * fa;
    if (bw.is_condition && vid_k && vid_v) {
        // Mixed attention: K = [vid_k, k_action], V = [vid_v, v_action], Q = q_action (24 heads)
        // vid_k/v: [hd_q, nh, vid_seq]   kr/vh: [hd_q, nh, seq]
        // cat along seq dim (ggml dim 2):
        const int vid_seq = (int) vid_k->ne[2];
        const int tot_seq = vid_seq + seq;
        // cat K: [hd_q, nh, vid_seq+seq]
        ggml_tensor * k_cat = ggml_concat(ctx, vid_k, kr, 2);  // [hd_q, nh, vid_seq+seq]
        ggml_tensor * v_cat = ggml_concat(ctx, vid_v, vh, 2);
        fa = fw_flash_attn(ctx, qr, k_cat, v_cat, scale);       // [hd_q, nh, seq]
        (void) tot_seq;
    } else {
        // Non-condition: pure self-attention
        fa = fw_flash_attn(ctx, qr, kr, vh, scale);             // [hd_q, nh, seq]
    }

    ggml_tensor * att_flat = ggml_reshape_2d(ctx, fa, nh * hd_q, seq);  // [nh*hd_q, seq]

    // --- 3. Output projection + gate residual ---
    ggml_tensor * sa_out = fw_linear(ctx, bw.sa_o_w, bw.sa_o_b, att_flat);  // [hd, seq]
    x = ggml_add(ctx, x, ggml_mul(ctx, sa_out, gate_msa));

    // --- 4. Cross-attention to text context (condition blocks / DiTBlock only) ---
    if (bw.is_condition && text_ctx) {
        const int n_heads_ca = fc.action_num_heads;  // 24
        const int text_seq   = (int) text_ctx->ne[1];

        // norm3 (affine LayerNorm)
        ggml_tensor * x_ln3 = fw_ln(ctx, x, bw.norm3_w, bw.norm3_b, eps);  // [hd, seq]

        // Q from action tokens
        ggml_tensor * ca_q  = fw_linear(ctx, bw.ca_q_w, bw.ca_q_b, x_ln3);    // [n_heads_ca*hd_q, seq]
        ggml_tensor * ca_qh = ggml_reshape_3d(ctx, ca_q, hd_q, n_heads_ca, seq);
        ggml_tensor * ca_qr = ggml_mul(ctx, ggml_rms_norm(ctx, ca_qh, eps),
                                        ggml_reshape_3d(ctx, bw.ca_norm_q, hd_q, n_heads_ca, 1));

        // K, V from text context
        ggml_tensor * ca_k  = fw_linear(ctx, bw.ca_k_w, bw.ca_k_b, text_ctx);  // [n_heads_ca*hd_q, text_seq]
        ggml_tensor * ca_kh = ggml_reshape_3d(ctx, ca_k, hd_q, n_heads_ca, text_seq);
        ggml_tensor * ca_kr = ggml_mul(ctx, ggml_rms_norm(ctx, ca_kh, eps),
                                        ggml_reshape_3d(ctx, bw.ca_norm_k, hd_q, n_heads_ca, 1));

        ggml_tensor * ca_v  = fw_linear(ctx, bw.ca_v_w, bw.ca_v_b, text_ctx);
        ggml_tensor * ca_vh = ggml_reshape_3d(ctx, ca_v, hd_q, n_heads_ca, text_seq);

        ggml_tensor * ca_fa   = fw_flash_attn(ctx, ca_qr, ca_kr, ca_vh, scale);  // [hd_q, n_heads_ca, seq]
        ggml_tensor * ca_flat = ggml_reshape_2d(ctx, ca_fa, n_heads_ca * hd_q, seq);
        ggml_tensor * ca_out  = fw_linear(ctx, bw.ca_o_w, bw.ca_o_b, ca_flat);
        x = ggml_add(ctx, x, ca_out);
    }

    // --- 5. FFN: norm2 + modulate → FFN → gate residual ---
    ggml_tensor * x_ln_ffn = ggml_norm(ctx, x, eps);
    x_ln_ffn = ggml_add(ctx, ggml_mul(ctx, x_ln_ffn, scale_mlp), shift_mlp);

    const int ffn_dim = fc.action_ffn_dim;
    ggml_tensor * ffn_up   = fw_linear(ctx, bw.ffn0_w, bw.ffn0_b, x_ln_ffn);
    ggml_tensor * ffn_act  = ggml_gelu(ctx, ffn_up);
    ggml_tensor * ffn_down = fw_linear(ctx, bw.ffn2_w, bw.ffn2_b, ffn_act);   // [hd, seq]

    x = ggml_add(ctx, x, ggml_mul(ctx, ffn_down, gate_mlp));
    return x;
}

// ---------------------------------------------------------------------------
// FasterWAMModel::predict  — main entry point (GPU ggml graph inference)
// ---------------------------------------------------------------------------
std::vector<float> FasterWAMModel::predict(const Inputs & in) {
    const auto t0 = std::chrono::steady_clock::now();

    const FasterWAMCfg & fc = fw_cfg;

    // -----------------------------------------------------------------------
    // 1. State normalization (host)
    // -----------------------------------------------------------------------
    std::vector<float> state_norm((size_t) fc.state_dim, 0.f);
    if (in.state) {
        for (int i = 0; i < fc.state_dim; ++i) {
            const float m = (i < (int) state_mean.size()) ? state_mean[i] : 0.f;
            const float s = (i < (int) state_std.size())  ? state_std[i]  : 1.f;
            state_norm[i] = (in.state[i] - m) / (s + 1e-8f);
        }
    }

    // -----------------------------------------------------------------------
    // 2. Text context
    // precomputed_img_emb carries T5 text embeddings [n_img_views, video_text_dim]
    // -----------------------------------------------------------------------
    int text_seq  = (in.precomputed_img_emb && in.n_img_views > 0) ? in.n_img_views : 1;
    std::vector<float> text_ctx_f32((size_t) text_seq * (size_t) fc.video_text_dim, 0.f);
    if (in.precomputed_img_emb && in.n_img_views > 0) {
        std::memcpy(text_ctx_f32.data(), in.precomputed_img_emb,
                    std::min(text_ctx_f32.size(),
                             (size_t) in.n_img_views * (size_t) fc.video_text_dim) * sizeof(float));
    }

    // -----------------------------------------------------------------------
    // 3. Initial action latent (encode noise via action_encoder on GPU graph)
    // -----------------------------------------------------------------------
    const int action_seq    = fc.action_horizon;
    const int action_hidden = fc.action_hidden;
    const int action_dim    = fc.action_dim;

    std::vector<float> raw_noise_f32((size_t) action_seq * (size_t) action_dim);
    if (in.noise) {
        std::memcpy(raw_noise_f32.data(), in.noise, raw_noise_f32.size() * sizeof(float));
    } else {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float & v : raw_noise_f32) { v = nd(rng); }
    }

    // -----------------------------------------------------------------------
    // 4. Prefill video KV cache (GPU ggml graph)
    //
    //    Input: precomputed_img_emb carries T5 text embeddings [text_seq, video_text_dim]
    //    OR no video (n_img_views=0): use single-token zero video latent as fallback.
    //
    //    Video latent shape: [16 ch, num_frames, H/2, W/2] (after VAE)
    //    Patch size = (1,2,2), so vid_seq = num_frames * (H/4) * (W/4)
    //    For no-video fallback: vid_seq = 1 (single zero token), bypass conv3d.
    //
    //    Caching: vid_kv depends only on text_ctx_f32 (not on robot state).
    //    When text_ctx_f32 is unchanged vs cached_text_ctx_f32 (pre-action-projection),
    //    skip prefill_video_cache_gpu and reuse cached vid_kv_k/v directly.
    // -----------------------------------------------------------------------
    const auto t_prefill_start = std::chrono::steady_clock::now();

    const int nh_ca = fc.action_num_heads;  // 24
    const int hd_q  = fc.action_head_dim;   // 128

    // Ensure gallocr exists
    if (!galloc) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    // Determine video sequence length.
    // For smoke test (no video input), use 1 zero token.
    const int patch_t = 1, patch_h = 2, patch_w = 2;  // video patch size
    // num_video_frames latent frames at half-spatial resolution
    // latent shape: [T=num_video_frames, H_lat, W_lat, C=16]
    // For a real 480×480 input: H_lat=60, W_lat=60, patch_h=2→30, patch_w=2→30
    // For smoke test: use vid_seq=1 (no video).
    const bool has_video = (in.images != nullptr && in.n_images > 0);
    int vid_seq;
    std::vector<float> vid_kv_k_flat, vid_kv_v_flat;  // flat storage for all conditions

    // vid_kv_k/v: per-condition host KV vectors used only for the no-video (zero) fallback path.
    // For the video path, KV is copied GPU→GPU from vp_fused_k/v; host vectors stay empty.
    std::vector<std::vector<float>> vid_kv_k(fc.num_condition_layers);
    std::vector<std::vector<float>> vid_kv_v(fc.num_condition_layers);

    // vid_seq is fully determined by input shape and doesn't change when text changes,
    // but we need it before checking the cache to size the kv vectors.
    if (!has_video) {
        vid_seq = 1;
    } else {
        // TODO: derive from actual video input shape
        vid_seq = 1;
    }

    // Check if vid_kv cache is valid for current text input.
    const bool vid_kv_stale = (text_ctx_f32 != cached_text_ctx_f32);
    if (vid_kv_stale) {
        std::vector<VideoKVPair> vid_kv_pairs;

        if (!has_video) {
            // No video: use 1 zero token (minimal placeholder, fast to run)
            vid_kv_pairs.resize((size_t) fc.num_condition_layers);
            (void)(size_t)(hd_q * nh_ca * vid_seq);  // suppress unused

            // Build a minimal video prefill with 1 zero token (no conv3d, just DiT blocks)
            const std::vector<float> zero_vtokens((size_t) fc.video_hidden * (size_t) vid_seq, 0.f);

            // Per-token t_mod: compute at t=0.
            // Python: token_timesteps all 0 (first frame = 0, and for single token also 0)
            // sinusoidal_emb(0, freq_dim) → Linear(SiLU(Linear)) → time_projection
            // vid_t_mod does NOT depend on text or state (only on weights and t=0).
            // Cached in cached_vid_t_mod after first computation.
            if (cached_vid_t_mod.empty()) {
                // First time: compute via GPU graph
                std::vector<float> vid_t_mod_f32((size_t) fc.video_hidden * 6 * (size_t) vid_seq, 0.f);
                {
                    const ggml_init_params ctx_p{ (size_t) 16 * 1024 * 1024, nullptr, true };
                    ggml_context * tctx = ggml_init(ctx_p);
                    if (tctx) {
                        // sinusoidal emb of t=0: [freq_dim]
                        ggml_tensor * tsemb = ggml_new_tensor_1d(tctx, GGML_TYPE_F32, fc.video_freq_dim);
                        ggml_set_input(tsemb);
                        ggml_tensor * t0 = fw_linear(tctx, w.vid_time_emb_0w, w.vid_time_emb_0b, tsemb);
                        ggml_tensor * t1 = ggml_silu(tctx, t0);
                        ggml_tensor * t2 = fw_linear(tctx, w.vid_time_emb_2w, w.vid_time_emb_2b, t1);
                        // time_projection: SiLU(t2) → Linear → [6*hidden]
                        ggml_tensor * t3 = ggml_silu(tctx, t2);
                        ggml_tensor * tp = fw_linear(tctx, w.vid_time_proj_w, w.vid_time_proj_b, t3);
                        ggml_tensor * tp_c = ggml_cont(tctx, tp);
                        ggml_set_output(tp_c);
                        ggml_cgraph * tg = ggml_new_graph(tctx);
                        ggml_build_forward_expand(tg, tp_c);
                        if (ggml_gallocr_alloc_graph(galloc, tg)) {
                            // t=0 → sinusoidal_emb_host(0, freq_dim)
                            std::vector<float> semb_host = sinusoidal_emb_host(0.f, fc.video_freq_dim);
                            ggml_backend_tensor_set_async(backend, tsemb, semb_host.data(), 0, ggml_nbytes(tsemb));
                            ggml_backend_graph_compute(backend, tg);
                            // tp shape: [6*hidden, 1] → replicate vid_seq times (vid_seq=1 here)
                            std::vector<float> tp_f32((size_t) fc.video_hidden * 6);
                            ggml_backend_tensor_get(tp_c, tp_f32.data(), 0, ggml_nbytes(tp_c));
                            // expand to [6*hidden, vid_seq] (vid_seq=1: just copy)
                            for (int s = 0; s < vid_seq; ++s) {
                                std::memcpy(vid_t_mod_f32.data() + (size_t) s * (size_t)(fc.video_hidden * 6),
                                            tp_f32.data(), (size_t)(fc.video_hidden * 6) * sizeof(float));
                            }
                        }
                        ggml_free(tctx);
                    }
                }
                cached_vid_t_mod = vid_t_mod_f32;
            }
            const std::vector<float> & vid_t_mod_f32 = cached_vid_t_mod;

            // M-RoPE positions: [4*vid_seq] int32 (1D flat: 4 values per token)
            std::vector<int32_t> vid_pos_ids = make_video_pos_ids(1, 1, 1);  // [4]

            // Text context for video: same text_ctx_f32 (raw T5 embeddings, not action-projected)
            // Use persistent prefill graph: build once per (vid_seq, text_seq) shape,
            // re-upload only t_tctx on text changes.
            const bool prefill_shape_changed = (vid_seq != cached_vp_vid_seq || text_seq != cached_vp_text_seq);
            if (prefill_shape_changed || !static_prefill_graph) {
                // Free previous graph context if shapes changed.
                if (ctx_vid_prefill) {
                    ggml_free(ctx_vid_prefill);
                    ctx_vid_prefill = nullptr;
                    static_prefill_graph = nullptr;
                }
                if (!prefill_galloc) {
                    prefill_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                }

                // Build the persistent prefill context and graph.
                const ggml_init_params vp_params{ (size_t) 512 * 1024 * 1024, nullptr, true };
                ctx_vid_prefill = ggml_init(vp_params);
                if (!ctx_vid_prefill) {
                    std::fprintf(stderr, "fasterwam: ggml_init failed for ctx_vid_prefill\n");
                    return {};
                }

                // Create input tensors (will be re-uploaded on each text change or first run).
                vp_t_vtoken = ggml_new_tensor_2d(ctx_vid_prefill, GGML_TYPE_F32, fc.video_hidden, vid_seq);
                ggml_set_name(vp_t_vtoken, "vid_tokens");
                ggml_set_input(vp_t_vtoken);

                vp_t_tctx = ggml_new_tensor_2d(ctx_vid_prefill, GGML_TYPE_F32, fc.video_text_dim, text_seq);
                ggml_set_name(vp_t_tctx, "vid_text_ctx");
                ggml_set_input(vp_t_tctx);

                vp_t_tmod = ggml_new_tensor_2d(ctx_vid_prefill, GGML_TYPE_F32, fc.video_hidden * 6, vid_seq);
                ggml_set_name(vp_t_tmod, "vid_t_mod");
                ggml_set_input(vp_t_tmod);

                vp_t_pos = ggml_new_tensor_1d(ctx_vid_prefill, GGML_TYPE_I32, (int64_t) vid_seq * 4);
                ggml_set_name(vp_t_pos, "vid_pos");
                ggml_set_input(vp_t_pos);

                // Build text embedding branch.
                ggml_tensor * text_emb = fw_linear(ctx_vid_prefill, w.vid_text_emb_0w, w.vid_text_emb_0b, vp_t_tctx);
                text_emb = ggml_gelu(ctx_vid_prefill, text_emb);
                text_emb = fw_linear(ctx_vid_prefill, w.vid_text_emb_2w, w.vid_text_emb_2b, text_emb);

                // Build condition intervals and read logit weights.
                const int ncl = fc.num_condition_layers;
                std::vector<std::vector<int>> intervals(ncl);
                {
                    int prev = -1;
                    for (int j = 0; j < ncl; ++j) {
                        const int cl = fc.condition_layers[j];
                        for (int li = prev + 1; li <= cl; ++li) { intervals[j].push_back(li); }
                        prev = cl;
                    }
                }
                std::vector<std::vector<float>> logits_host(ncl);
                for (int j = 0; j < ncl; ++j) {
                    int n = (int) intervals[j].size();
                    logits_host[j].resize((size_t) n);
                    ggml_backend_tensor_get(w.vid_kv_logits[j], logits_host[j].data(), 0, (size_t) n * sizeof(float));
                    softmax_inplace(logits_host[j].data(), n);
                }

                // Run video DiT blocks up to the last condition layer, fusing K/V at condition layers.
                ggml_tensor * x = vp_t_vtoken;
                std::vector<ggml_tensor *> interval_k, interval_v;
                int cur_interval_j = 0;
                std::vector<ggml_tensor *> fused_k_out(ncl, nullptr);
                std::vector<ggml_tensor *> fused_v_out(ncl, nullptr);

                // Only iterate up to and including the last condition layer.
                // Layers beyond fc.condition_layers[ncl-1] don't contribute any KV output
                // and their x update is discarded — skip them to save GPU compute.
                const int last_cond_layer = fc.condition_layers[ncl - 1];
                for (int li = 0; li <= last_cond_layer; ++li) {
                    const VideoDiTBlockW & bw = w.video_blocks[(size_t) li];
                    VideoDiTBlockOut blk = build_video_dit_block(ctx_vid_prefill, bw, x, vp_t_tmod, vp_t_pos, text_emb, fc);
                    x = blk.x;
                    interval_k.push_back(blk.k);
                    interval_v.push_back(blk.v);

                    bool is_cond = false;
                    for (int j = 0; j < ncl; ++j) {
                        if (fc.condition_layers[j] == li) { is_cond = true; cur_interval_j = j; break; }
                    }
                    if (is_cond) {
                        const std::vector<float> & wghts = logits_host[cur_interval_j];
                        const int n_interval = (int) interval_k.size();
                        ggml_tensor * fk = ggml_scale(ctx_vid_prefill, interval_k[0], wghts[0]);
                        ggml_tensor * fv = ggml_scale(ctx_vid_prefill, interval_v[0], wghts[0]);
                        for (int ii = 1; ii < n_interval; ++ii) {
                            fk = ggml_add(ctx_vid_prefill, fk, ggml_scale(ctx_vid_prefill, interval_k[(size_t)ii], wghts[(size_t)ii]));
                            fv = ggml_add(ctx_vid_prefill, fv, ggml_scale(ctx_vid_prefill, interval_v[(size_t)ii], wghts[(size_t)ii]));
                        }
                        fk = ggml_cont(ctx_vid_prefill, fk);
                        fv = ggml_cont(ctx_vid_prefill, fv);
                        ggml_set_output(fk);
                        ggml_set_output(fv);
                        fused_k_out[cur_interval_j] = fk;
                        fused_v_out[cur_interval_j] = fv;
                        vp_fused_k[cur_interval_j] = fk;
                        vp_fused_v[cur_interval_j] = fv;
                        interval_k.clear();
                        interval_v.clear();
                    }
                }

                // Build the graph.
                static_prefill_graph = ggml_new_graph_custom(ctx_vid_prefill, 65536, false);
                for (int j = 0; j < ncl; ++j) {
                    if (fused_k_out[j]) { ggml_build_forward_expand(static_prefill_graph, fused_k_out[j]); }
                    if (fused_v_out[j]) { ggml_build_forward_expand(static_prefill_graph, fused_v_out[j]); }
                }

                if (!ggml_gallocr_alloc_graph(prefill_galloc, static_prefill_graph)) {
                    std::fprintf(stderr, "fasterwam: video prefill galloc alloc failed\n");
                    ggml_free(ctx_vid_prefill);
                    ctx_vid_prefill = nullptr;
                    static_prefill_graph = nullptr;
                    return {};
                }

                // Upload constant inputs (vtoken=zeros, t_mod, pos are constant).
                // Use async: same stream as graph_compute ensures ordering.
                ggml_backend_tensor_set_async(backend, vp_t_vtoken, zero_vtokens.data(), 0, ggml_nbytes(vp_t_vtoken));
                ggml_backend_tensor_set_async(backend, vp_t_tmod,   vid_t_mod_f32.data(), 0, ggml_nbytes(vp_t_tmod));
                ggml_backend_tensor_set_async(backend, vp_t_pos,    vid_pos_ids.data(),   0, ggml_nbytes(vp_t_pos));

                cached_vp_vid_seq  = vid_seq;
                cached_vp_text_seq = text_seq;
            }

            // Re-upload only the text context (changes every text-changed call).
            // Async: same stream as graph_compute, ordering guaranteed.
            ggml_backend_tensor_set_async(backend, vp_t_tctx, text_ctx_f32.data(), 0, ggml_nbytes(vp_t_tctx));

            // Run the persistent prefill graph.
            ggml_backend_graph_compute(backend, static_prefill_graph);

            // vp_fused_k/v now contain the computed KV tensors on GPU.
            // The G→G copy into tc_vid_k/v is done below in Section 6 once const_buf exists.
            // Do NOT read back to host here — avoid the G→H→G roundtrip entirely.
            // vid_kv_k/v host vectors stay empty; Section 6 G→G path will be used.
        } else {
            // TODO: real video path with conv3d patch embedding.
            // For now fall through to zero KV: tc_vid_k/v will be zero-filled on first alloc.
            // vp_fused_k[0] will be null here, so Section 6 G→G copy won't fire;
            // vid_kv_k/v host zeros are used instead.
            const size_t kv_e0 = (size_t) hd_q * (size_t) nh_ca * (size_t) vid_seq;
            for (int ci = 0; ci < fc.num_condition_layers; ++ci) {
                vid_kv_k[ci].assign(kv_e0, 0.f);
                vid_kv_v[ci].assign(kv_e0, 0.f);
            }
        }
        // (no-video path already populated vid_kv_k/v above;
        //  video path leaves them empty — G→G copy in Section 6 handles the upload.)
    }

    const auto t_prefill_end = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // 5. Build action text context + position ids
    // -----------------------------------------------------------------------

    // 5a. Proprio encoding: state_norm → Linear(state_dim→video_text_dim) → [video_text_dim]
    // Persistent graph: built once per state_dim (shape never changes at runtime).
    std::vector<float> proprio_token_f32((size_t) fc.video_text_dim, 0.f);
    {
        const bool pp_shape_changed = (fc.state_dim != cached_pp_state_dim);
        if (pp_shape_changed || !proprio_graph) {
            if (ctx_proprio) { ggml_free(ctx_proprio); ctx_proprio = nullptr; proprio_graph = nullptr; }
            if (!proprio_galloc) {
                proprio_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            }
            const ggml_init_params pp{ (size_t) 2 * 1024 * 1024, nullptr, true };
            ctx_proprio = ggml_init(pp);
            if (!ctx_proprio) { return {}; }
            pp_prop_in = ggml_new_tensor_2d(ctx_proprio, GGML_TYPE_F32, fc.state_dim, 1);
            ggml_set_input(pp_prop_in);
            ggml_tensor * prop_out = fw_linear(ctx_proprio, w.proprio_enc_w, w.proprio_enc_b, pp_prop_in);
            pp_prop_out = ggml_cont(ctx_proprio, prop_out);
            ggml_set_output(pp_prop_out);
            proprio_graph = ggml_new_graph(ctx_proprio);
            ggml_build_forward_expand(proprio_graph, pp_prop_out);
            if (!ggml_gallocr_alloc_graph(proprio_galloc, proprio_graph)) { return {}; }
            cached_pp_state_dim = fc.state_dim;
        }
        ggml_backend_tensor_set_async(backend, pp_prop_in, state_norm.data(), 0, ggml_nbytes(pp_prop_in));
        ggml_backend_graph_compute(backend, proprio_graph);
        ggml_backend_tensor_get(pp_prop_out, proprio_token_f32.data(), 0, ggml_nbytes(pp_prop_out));
    }

    // 5b. Append proprio token to text context → ext_ctx [text_seq+1, video_text_dim]
    const int ext_text_seq = text_seq + 1;
    std::vector<float> text_ctx_ext((size_t) ext_text_seq * fc.video_text_dim);
    std::memcpy(text_ctx_ext.data(), text_ctx_f32.data(), text_ctx_f32.size() * sizeof(float));
    std::memcpy(text_ctx_ext.data() + text_ctx_f32.size(),
                proprio_token_f32.data(), proprio_token_f32.size() * sizeof(float));

    // 5c. Text embedding: Linear(video_text_dim→action_hidden) + GELU + Linear → [action_hidden, ext_text_seq]
    // Persistent graph: built once per ext_text_seq shape.
    std::vector<float> action_text_ctx_f32((size_t) ext_text_seq * fc.action_hidden);
    {
        const bool te_shape_changed = (ext_text_seq != cached_te_text_seq);
        if (te_shape_changed || !textemb_graph) {
            if (ctx_textemb) { ggml_free(ctx_textemb); ctx_textemb = nullptr; textemb_graph = nullptr; }
            if (!textemb_galloc) {
                textemb_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            }
            const ggml_init_params tp{ (size_t) 4 * 1024 * 1024, nullptr, true };
            ctx_textemb = ggml_init(tp);
            if (!ctx_textemb) { return {}; }
            te_text_in = ggml_new_tensor_2d(ctx_textemb, GGML_TYPE_F32, fc.video_text_dim, ext_text_seq);
            ggml_set_input(te_text_in);
            ggml_tensor * te0 = fw_linear(ctx_textemb, w.act_text_emb_0w, w.act_text_emb_0b, te_text_in);
            ggml_tensor * te1 = ggml_gelu(ctx_textemb, te0);
            ggml_tensor * te2 = fw_linear(ctx_textemb, w.act_text_emb_2w, w.act_text_emb_2b, te1);
            te_text_out = ggml_cont(ctx_textemb, te2);
            ggml_set_output(te_text_out);
            textemb_graph = ggml_new_graph(ctx_textemb);
            ggml_build_forward_expand(textemb_graph, te_text_out);
            if (!ggml_gallocr_alloc_graph(textemb_galloc, textemb_graph)) { return {}; }
            cached_te_text_seq = ext_text_seq;
        }
        ggml_backend_tensor_set_async(backend, te_text_in, text_ctx_ext.data(), 0, ggml_nbytes(te_text_in));
        ggml_backend_graph_compute(backend, textemb_graph);
        ggml_backend_tensor_get(te_text_out, action_text_ctx_f32.data(), 0, ggml_nbytes(te_text_out));
    }
    const int final_text_seq = ext_text_seq;

    // 5d. Position ids for action tokens [0..action_seq-1]
    const std::vector<int32_t> pos_ids_host = make_position_ids(action_seq);

    // -----------------------------------------------------------------------
    // 5e. Flow-matching timestep schedule + pre-computed time embeddings
    // -----------------------------------------------------------------------
    const auto timesteps = make_timestep_schedule(fc.num_inference_steps, fc.infer_shift,
                                                   fc.num_train_timesteps);

    // Pre-compute all sinusoidal time embeddings before the loop (avoids 20× alloc+trig).
    std::vector<std::vector<float>> time_emb_by_step((size_t) fc.num_inference_steps);
    for (int step = fc.num_inference_steps - 1; step >= 0; --step) {
        const float t_curr = timesteps[(size_t) step + 1];
        time_emb_by_step[(size_t) step] = sinusoidal_emb_host(t_curr, fc.action_freq_dim);
    }

    // -----------------------------------------------------------------------
    // 6. Action denoising loop — ggml GPU graph per step (optimized)
    //
    //    Key optimizations vs. naive per-step ggml_init/free:
    //    a) ctx_step is reset with ggml_reset() (O(1)) instead of 256 MB
    //       malloc+free each step.
    //    b) Constant inputs (text_ctx, pos_ids, vid_kv × 16) are uploaded
    //       to GPU once before the loop into persistent ggml tensors stored
    //       in const_buf.  These tensors act like weights — they are
    //       referenced directly in the graph and never re-uploaded.
    //    c) gallocr_reserve() is called after the first graph build so that
    //       subsequent alloc_graph() calls are O(n) pointer-only assignments.
    //    d) const_buf persists across predict() calls; vid_kv is only
    //       re-uploaded when text_ctx changes; the graph is rebuilt only
    //       on const_buf reallocation (pointer invalidation).
    // -----------------------------------------------------------------------
    const auto t_denoise_start = std::chrono::steady_clock::now();

    // --- (a) Persistent step context ---
    // The unrolled graph is built once (when !denoise_graph_valid) into ctx_step below.
    // ctx_step is only needed at graph build time — see the denoise_graph_valid block below.

    // --- (b) Upload constant tensors into persistent GPU buffer ---
    // const_buf is allocated once per (text_seq, vid_seq) shape combination and persists across
    // predict() calls.  On each call we selectively re-upload only what changed:
    //   - tc_pos, tc_time_emb[]: constant forever — uploaded only on first allocation.
    //   - tc_vid_k/v: depend on text_ctx content — re-uploaded when text changes.
    //   - tc_text: state-dependent (text + proprio) — re-uploaded every call.
    //   - tc_x_raw: noise — uploaded just before graph_compute (further below).
    // static_denoise_graph is rebuilt only when const_buf is newly allocated (pointer change).

    const bool shape_changed = (final_text_seq != cached_text_seq || vid_seq != cached_vid_seq);
    if (shape_changed) {
        // Shapes changed: free everything and reallocate.
        if (ctx_const) { ggml_free(ctx_const);    ctx_const = nullptr; }
        if (const_buf) { ggml_backend_buffer_free(const_buf); const_buf = nullptr; }
        denoise_graph_valid = false;
        cached_text_seq = -1;
        cached_vid_seq  = -1;
    }

    if (!const_buf) {
        // Meta-context for the constant tensor descriptors (host side only).
        ctx_const = ggml_init({ (size_t) 4 * 1024 * 1024, nullptr, true });
        if (!ctx_const) {
            std::fprintf(stderr, "fasterwam: ggml_init failed for ctx_const\n");
            return {};
        }

        // Compute total bytes needed for all constant tensors.
        const size_t text_bytes   = (size_t) final_text_seq * (size_t) fc.action_hidden * sizeof(float);
        const size_t pos_bytes    = (size_t) action_seq * sizeof(int32_t);
        const size_t kv_bytes     = (size_t) hd_q * (size_t) nh_ca * (size_t) vid_seq * sizeof(float);
        const size_t x_raw_bytes  = (size_t) action_seq * (size_t) action_dim * sizeof(float);
        const size_t time_emb_bytes = (size_t) fc.num_inference_steps * (size_t) fc.action_freq_dim * sizeof(float);
        const size_t total_const_bytes = text_bytes + pos_bytes
                                       + (size_t) fc.num_condition_layers * 2 * kv_bytes
                                       + x_raw_bytes + time_emb_bytes;

        // Allocate one GPU buffer for all constants.
        const_buf = ggml_backend_alloc_buffer(backend, total_const_bytes + 4096 /*alignment pad*/);
        if (!const_buf) {
            std::fprintf(stderr, "fasterwam: failed to allocate const_buf\n");
            return {};
        }
        ggml_tallocr ta_const = ggml_tallocr_new(const_buf);

        // Create tensor descriptors (no data upload yet — done selectively below).
        tc_text = ggml_new_tensor_2d(ctx_const, GGML_TYPE_F32, fc.action_hidden, final_text_seq);
        ggml_tallocr_alloc(&ta_const, tc_text);

        tc_pos = ggml_new_tensor_1d(ctx_const, GGML_TYPE_I32, action_seq);
        ggml_tallocr_alloc(&ta_const, tc_pos);

        for (int ci = 0; ci < fc.num_condition_layers; ++ci) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "tc_vid_k_%d", ci);
            std::snprintf(vname, sizeof(vname), "tc_vid_v_%d", ci);
            tc_vid_k[ci] = ggml_new_tensor_3d(ctx_const, GGML_TYPE_F32, hd_q, nh_ca, vid_seq);
            ggml_set_name(tc_vid_k[ci], kname);
            ggml_tallocr_alloc(&ta_const, tc_vid_k[ci]);
            tc_vid_v[ci] = ggml_new_tensor_3d(ctx_const, GGML_TYPE_F32, hd_q, nh_ca, vid_seq);
            ggml_set_name(tc_vid_v[ci], vname);
            ggml_tallocr_alloc(&ta_const, tc_vid_v[ci]);
        }

        tc_x_raw = ggml_new_tensor_2d(ctx_const, GGML_TYPE_F32, action_dim, action_seq);
        ggml_set_name(tc_x_raw, "tc_x_raw");
        ggml_tallocr_alloc(&ta_const, tc_x_raw);

        GGML_ASSERT(fc.num_inference_steps <= MAX_DENOISE_STEPS);
        for (int step = 0; step < fc.num_inference_steps; ++step) {
            char tname[32];
            std::snprintf(tname, sizeof(tname), "tc_time_%d", step);
            tc_time_emb[step] = ggml_new_tensor_1d(ctx_const, GGML_TYPE_F32, fc.action_freq_dim);
            ggml_set_name(tc_time_emb[step], tname);
            ggml_tallocr_alloc(&ta_const, tc_time_emb[step]);
        }

        // Upload truly-constant tensors once — pos ids and time embeddings never change.
        // Async is fine: these uploads happen before graph build/alloc which forces ordering anyway.
        ggml_backend_tensor_set_async(backend, tc_pos, pos_ids_host.data(), 0, ggml_nbytes(tc_pos));
        for (int step = 0; step < fc.num_inference_steps; ++step) {
            ggml_backend_tensor_set_async(backend, tc_time_emb[step],
                                    time_emb_by_step[(size_t) step].data(),
                                    0, ggml_nbytes(tc_time_emb[step]));
        }

        cached_text_seq = final_text_seq;
        cached_vid_seq  = vid_seq;
        // cached_text_ctx_f32 is already empty — force vid_kv upload on this call.
        // (vid_kv_stale will be true because cached_text_ctx_f32 was cleared by the shape change
        //  path or is empty on the very first call.)
    }

    // Upload tc_vid_k/v when the text content has changed (or on first allocation).
    // Prefer GPU→GPU copy from vp_fused_k/v (no host stall).
    // Fall back to host upload (vid_kv_k/v zeros) only for the no-video path.
    if (vid_kv_stale) {
        if (vp_fused_k[0] && tc_vid_k[0]) {
            // Video-prefill path: GPU→GPU direct copy on the compute stream.
            // ggml_backend_tensor_copy_async uses cuda_ctx->stream() with no sync — truly async.
            for (int ci = 0; ci < fc.num_condition_layers; ++ci) {
                if (vp_fused_k[ci] && tc_vid_k[ci]) {
                    ggml_backend_tensor_copy_async(backend, backend, vp_fused_k[ci], tc_vid_k[ci]);
                }
                if (vp_fused_v[ci] && tc_vid_v[ci]) {
                    ggml_backend_tensor_copy_async(backend, backend, vp_fused_v[ci], tc_vid_v[ci]);
                }
            }
        } else if (tc_vid_k[0]) {
            // No-video path: upload zeros from host (first call or shape change only).
            for (int ci = 0; ci < fc.num_condition_layers; ++ci) {
                if (!vid_kv_k[ci].empty()) {
                    ggml_backend_tensor_set_async(backend, tc_vid_k[ci], vid_kv_k[(size_t)ci].data(), 0, ggml_nbytes(tc_vid_k[ci]));
                    ggml_backend_tensor_set_async(backend, tc_vid_v[ci], vid_kv_v[(size_t)ci].data(), 0, ggml_nbytes(tc_vid_v[ci]));
                }
            }
        }
        // else: tc_vid_k[0] is null — const_buf not yet allocated; skip upload.
        // This can't happen: const_buf is allocated above before reaching here.
        cached_text_ctx_f32 = text_ctx_f32;  // cache the raw T5 text (vid_kv dependency)
    }

    // Always re-upload tc_text: it embeds the current robot state (proprio-conditioned).
    // Use async variant: enqueues H2D copy on cuda_ctx->stream(), same stream as graph_compute,
    // so ordering is preserved automatically with no host-blocking cudaStreamSynchronize.
    ggml_backend_tensor_set_async(backend, tc_text, action_text_ctx_f32.data(), 0, ggml_nbytes(tc_text));

    // --- Helper lambda: build one denoising step in dctx, using a pre-loaded GPU time emb tensor ---
    // t_time_const: GPU tensor [freq_dim] already holding the sinusoidal emb for this step.
    // delta_val: Euler step size (baked into ggml_scale op_params — constant in the graph).
    // Returns x_updated (view of tc_x_raw after the Euler step).
    auto build_one_step = [&](ggml_context * dctx,
                               ggml_tensor * t_time_const,
                               float delta_val) -> ggml_tensor * {
        // tc_x_raw is the pre-allocated GPU latent — referenced directly, not as set_input.
        ggml_tensor * t_x = fw_linear(dctx, w.act_encoder_w, w.act_encoder_b, tc_x_raw);

        // Time embedding MLP: Linear→SiLU→Linear→SiLU→Linear(6*hd)
        ggml_tensor * te0   = fw_linear(dctx, w.act_time_emb_0w, w.act_time_emb_0b, t_time_const);
        ggml_tensor * te1   = ggml_silu(dctx, te0);
        ggml_tensor * te2   = fw_linear(dctx, w.act_time_emb_2w, w.act_time_emb_2b, te1);
        ggml_tensor * tp0   = ggml_silu(dctx, te2);
        ggml_tensor * t_mod = fw_linear(dctx, w.act_time_proj_w, w.act_time_proj_b, tp0);
        ggml_tensor * t_mod_2d = ggml_reshape_2d(dctx, t_mod, action_hidden * 6, 1);

        // Action DiT blocks — constant KV refs via tc_vid_k/v (no set_input needed).
        ggml_tensor * x = t_x;
        int cond_idx = 0;
        for (int li = 0; li < fc.action_num_layers; ++li) {
            const ActionDiTBlockW & bw = w.action_blocks[(size_t) li];
            ggml_tensor * vk = nullptr;
            ggml_tensor * vv = nullptr;
            if (bw.is_condition) {
                vk = tc_vid_k[cond_idx];
                vv = tc_vid_v[cond_idx];
                ++cond_idx;
            }
            x = build_action_dit_block(dctx, bw, x, t_mod_2d, tc_pos, vk, vv, tc_text, fc);
        }

        // Head projection — produces velocity prediction
        ggml_tensor * pred   = fw_linear(dctx, w.act_head_w, w.act_head_b, x);
        ggml_tensor * pred_c = ggml_cont(dctx, pred);

        // In-graph Euler update: x_raw += delta * velocity (delta is a compile-time constant in graph)
        // Use ggml_add_inplace to avoid the explicit ggml_cpy node — saves 20 copy kernel launches
        // across the unrolled loop and removes a graph-level data-dependency edge per step.
        ggml_tensor * scale_v   = ggml_scale(dctx, pred_c, delta_val);
        ggml_tensor * x_updated = ggml_add_inplace(dctx, tc_x_raw, scale_v);
        return x_updated;
    };

    // --- Build unrolled 20-step denoising graph in ctx_step ---
    // All 20 steps are chained in one graph: step N reads tc_x_raw, writes it back via ggml_cpy,
    // then step N+1 reads the same tc_x_raw. One ggml_backend_graph_compute call runs all steps.
    // Time embeddings and deltas are embedded as GPU constants in const_buf (no per-step H2D).
    // The graph can be captured by GGML_CUDA_GRAPHS since no properties change between calls.
    // Graph is built once (on const_buf allocation) and reused on all subsequent predict() calls.
    if (!denoise_graph_valid) {
        if (!ctx_step) {
            // ctx_step needs to hold metadata for ~20 steps × ~30 blocks × ~50 ops ≈ 30000 tensors.
            // Use 512 MB to be safe (host-side metadata only; no GPU allocation here).
            const ggml_init_params ctx_step_params{ (size_t) 512 * 1024 * 1024, nullptr, true };
            ctx_step = ggml_init(ctx_step_params);
            if (!ctx_step) {
                std::fprintf(stderr, "fasterwam: ggml_init failed for ctx_step\n");
                return {};
            }
        } else {
            ggml_reset(ctx_step);
        }

        // Compute deltas for each step (needed to embed as constants in the graph).
        std::vector<float> step_deltas((size_t) fc.num_inference_steps);
        for (int step = fc.num_inference_steps - 1; step >= 0; --step) {
            const float t_curr     = timesteps[(size_t) step + 1];
            const float t_next     = timesteps[(size_t) step];
            const float sigma_curr = t_curr / (float) fc.num_train_timesteps;
            const float sigma_next = t_next / (float) fc.num_train_timesteps;
            step_deltas[(size_t) step] = sigma_next - sigma_curr;  // negative
        }

        // Build the unrolled graph: chain all steps, each updating tc_x_raw in-place.
        {
            // ~20 steps × 8192 nodes/step + safety margin
            const size_t graph_nodes = (size_t) fc.num_inference_steps * 8192 + 4096;
            static_denoise_graph = ggml_new_graph_custom(ctx_step, graph_nodes, false);

            ggml_tensor * last_out = nullptr;
            for (int step = fc.num_inference_steps - 1; step >= 0; --step) {
                last_out = build_one_step(ctx_step,
                                         tc_time_emb[(size_t) step],
                                         step_deltas[(size_t) step]);
            }
            ggml_set_output(last_out);
            ggml_build_forward_expand(static_denoise_graph, last_out);

            // Reserve + allocate GPU scratch.
            ggml_gallocr_reserve(galloc, static_denoise_graph);
            ggml_gallocr_alloc_graph(galloc, static_denoise_graph);
        }
        denoise_graph_valid = true;
    }

    // --- Run the unrolled denoising graph ---
    // Upload initial noise once; all N steps execute in a single ggml_backend_graph_compute.
    // Use async variant for the same stream-ordering guarantee (no host stall).
    const auto t_xset0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set_async(backend, tc_x_raw, raw_noise_f32.data(), 0, ggml_nbytes(tc_x_raw));
    const auto t_xset1 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, static_denoise_graph);
    const auto t_gcomp = std::chrono::steady_clock::now();

    // Read final latent back to host once (single D2H after all N steps).
    std::vector<float> action_raw_f32((size_t) action_seq * action_dim);
    ggml_backend_tensor_get(tc_x_raw, action_raw_f32.data(), 0, ggml_nbytes(tc_x_raw));
    const auto t_xget = std::chrono::steady_clock::now();
    if (const char * dbg = std::getenv("VLA_FASTERWAM_TIMING_DETAIL")) {
        (void) dbg;
        auto msf = [](auto a, auto b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };
        std::fprintf(stderr, "  [denoise detail] tc_text_set=?  x_set=%.3f ms  graph_compute=%.3f ms  x_get=%.3f ms\n",
                     msf(t_xset0, t_xset1), msf(t_xset1, t_gcomp), msf(t_gcomp, t_xget));
    }

    const auto t_denoise_end = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // 7. action_raw_f32 is now the clean latent in action_dim space — no further decoding needed
    // -----------------------------------------------------------------------
    std::vector<float> action_out = action_raw_f32;

    // Denormalize
    for (int t = 0; t < action_seq; ++t) {
        float * row = action_out.data() + (size_t) t * (size_t) action_dim;
        for (int j = 0; j < action_dim; ++j) {
            const size_t stat_i = (size_t) t * (size_t) action_dim + (size_t) j;
            const float m = (stat_i < action_mean.size()) ? action_mean[stat_i] : 0.f;
            const float s = (stat_i < action_std.size())  ? action_std[stat_i]  : 1.f;
            row[j] = row[j] * (s + 1e-8f) + m;
        }
    }

    // -----------------------------------------------------------------------
    // 8. Stats
    // -----------------------------------------------------------------------
    const auto t1 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    stats.ms_total     = ms(t0, t1);
    stats.ms_prefill   = ms(t_prefill_start, t_prefill_end);
    stats.ms_denoise   = ms(t_denoise_start, t_denoise_end);
    stats.ms_inference = stats.ms_prefill + stats.ms_denoise;

    return action_out;
}

// ---------------------------------------------------------------------------
// FasterWAMModel::prepare  — Phase 1 of two-phase inference
// Runs: state_norm, text_ctx, proprio graph, textemb graph, video prefill graph.
// Stores results in PreparedInput for use by compute().
//
// img_emb_host  → action_text_ctx_f32 (action hidden, f32)
// n_img_tokens  → final_text_seq (int64_t cast)
// noise         → raw_noise_f32
// ---------------------------------------------------------------------------
PreparedInput FasterWAMModel::prepare(const Inputs & in) {
    PreparedInput p;

    const FasterWAMCfg & fc = fw_cfg;
    const auto t0 = std::chrono::steady_clock::now();

    // 1. State normalization
    std::vector<float> state_norm((size_t) fc.state_dim, 0.f);
    if (in.state) {
        for (int i = 0; i < fc.state_dim; ++i) {
            const float m = (i < (int) state_mean.size()) ? state_mean[i] : 0.f;
            const float s = (i < (int) state_std.size())  ? state_std[i]  : 1.f;
            state_norm[i] = (in.state[i] - m) / (s + 1e-8f);
        }
    }

    // 2. Text context
    int text_seq = (in.precomputed_img_emb && in.n_img_views > 0) ? in.n_img_views : 1;
    std::vector<float> text_ctx_f32((size_t) text_seq * (size_t) fc.video_text_dim, 0.f);
    if (in.precomputed_img_emb && in.n_img_views > 0) {
        std::memcpy(text_ctx_f32.data(), in.precomputed_img_emb,
                    std::min(text_ctx_f32.size(),
                             (size_t) in.n_img_views * (size_t) fc.video_text_dim) * sizeof(float));
    }

    // 3. Raw noise (copy or generate)
    std::vector<float> raw_noise_f32((size_t) fc.action_horizon * (size_t) fc.action_dim);
    if (in.noise) {
        std::memcpy(raw_noise_f32.data(), in.noise, raw_noise_f32.size() * sizeof(float));
    } else {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float & v : raw_noise_f32) { v = nd(rng); }
    }

    // 4. Video prefill (same logic as predict())
    if (!galloc) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    const int nh_ca = fc.action_num_heads;
    const int hd_q  = fc.action_head_dim;

    const bool has_video = (in.images != nullptr && in.n_images > 0);
    const int vid_seq = 1;  // TODO: real video path

    const bool vid_kv_stale = (text_ctx_f32 != cached_text_ctx_f32);
    if (vid_kv_stale) {
        if (!has_video) {
            // Build/reuse the prefill graph (warm-up guard: predict() must have run at least once)
            const bool vp_shape_changed = (vid_seq != cached_vp_vid_seq || text_seq != cached_vp_text_seq);
            if (vp_shape_changed || !static_prefill_graph) {
                p.ok    = false;
                p.error = "fasterwam: prepare() called before prefill graph is warmed up; use predict() for first call";
                return p;
            }

            ggml_backend_tensor_set_async(backend, vp_t_tctx, text_ctx_f32.data(), 0, ggml_nbytes(vp_t_tctx));
            ggml_backend_graph_compute(backend, static_prefill_graph);
        }
    }

    // 5. Proprio + textemb graphs
    std::vector<float> proprio_token_f32((size_t) fc.video_text_dim, 0.f);
    {
        const bool pp_shape_changed = (fc.state_dim != cached_pp_state_dim);
        if (pp_shape_changed || !proprio_graph) {
            p.ok    = false;
            p.error = "fasterwam: prepare() called before proprio graph is warmed up; use predict() for first call";
            return p;
        }
        ggml_backend_tensor_set_async(backend, pp_prop_in, state_norm.data(), 0, ggml_nbytes(pp_prop_in));
        ggml_backend_graph_compute(backend, proprio_graph);
        ggml_backend_tensor_get(pp_prop_out, proprio_token_f32.data(), 0, ggml_nbytes(pp_prop_out));
    }

    const int ext_text_seq = text_seq + 1;
    std::vector<float> text_ctx_ext((size_t) ext_text_seq * fc.video_text_dim);
    std::memcpy(text_ctx_ext.data(), text_ctx_f32.data(), text_ctx_f32.size() * sizeof(float));
    std::memcpy(text_ctx_ext.data() + text_ctx_f32.size(),
                proprio_token_f32.data(), proprio_token_f32.size() * sizeof(float));

    std::vector<float> action_text_ctx_f32((size_t) ext_text_seq * fc.action_hidden);
    {
        const bool te_shape_changed = (ext_text_seq != cached_te_text_seq);
        if (te_shape_changed || !textemb_graph) {
            p.ok    = false;
            p.error = "fasterwam: prepare() called before textemb graph is warmed up; use predict() for first call";
            return p;
        }
        ggml_backend_tensor_set_async(backend, te_text_in, text_ctx_ext.data(), 0, ggml_nbytes(te_text_in));
        ggml_backend_graph_compute(backend, textemb_graph);
        ggml_backend_tensor_get(te_text_out, action_text_ctx_f32.data(), 0, ggml_nbytes(te_text_out));
    }

    // Update cached_text_ctx_f32 now (same as predict() does on first vid_kv_stale pass)
    if (vid_kv_stale) {
        cached_text_ctx_f32 = text_ctx_f32;
    }

    // Package inter-phase data
    p.img_emb_host  = std::move(action_text_ctx_f32);
    p.n_img_tokens  = (int64_t) (text_seq + 1);   // == final_text_seq
    p.noise         = std::move(raw_noise_f32);
    p.ok            = true;

    (void) t0;  // timing not tracked in prepare; server will track wall time
    return p;
}

// ---------------------------------------------------------------------------
// FasterWAMModel::compute  — Phase 2 of two-phase inference
// Runs: const_buf management, tc_text upload, denoise graph, D2H, denormalize.
// ---------------------------------------------------------------------------
std::vector<float> FasterWAMModel::compute(const PreparedInput & p) {
    if (!p.ok || p.img_emb_host.empty() || p.noise.empty()) {
        return {};
    }

    const FasterWAMCfg & fc = fw_cfg;
    const auto t0 = std::chrono::steady_clock::now();

    const std::vector<float> & action_text_ctx_f32 = p.img_emb_host;
    const int final_text_seq = (int) p.n_img_tokens;
    const std::vector<float> & raw_noise_f32 = p.noise;

    const int action_seq    = fc.action_horizon;
    const int action_dim    = fc.action_dim;

    // vid_seq must match what was used in prepare()
    const int vid_seq = (int) cached_vid_seq;
    if (vid_seq <= 0) {
        // const_buf not yet allocated — shouldn't happen if prepare() ran first
        return {};
    }

    const auto t_denoise_start = std::chrono::steady_clock::now();

    // Check shape consistency (text_seq must match what const_buf was sized for)
    if (final_text_seq != cached_text_seq || !const_buf) {
        // const_buf shape mismatch — can't proceed without rebuilding, fall back to predict()
        return {};
    }

    // vid_kv: the prefill ran in prepare(), so vp_fused_k/v already hold fresh data.
    // Check if we need to upload (text changed in prepare() means vid_kv_stale was true there).
    // cached_text_ctx_f32 was already updated in prepare(), so vid_kv_stale is now false here.
    // However, the G→G copy from vp_fused_k/v → tc_vid_k/v was NOT done in prepare() because
    // const_buf may not have existed yet at that point. Do it now if vp_fused_k[0] != nullptr.
    // We detect "fresh prefill in this prepare/compute pair" by comparing action_text_ctx_f32
    // with what we'd expect from tc_text — instead use a simpler heuristic: always do the
    // G→G copy when prepare() was called (p.ok == true implies prepare ran the prefill).
    // Since this is always cheap (G→G, no sync), just do it unconditionally.
    if (vp_fused_k[0] && tc_vid_k[0]) {
        for (int ci = 0; ci < fc.num_condition_layers; ++ci) {
            if (vp_fused_k[ci] && tc_vid_k[ci]) {
                ggml_backend_tensor_copy_async(backend, backend, vp_fused_k[ci], tc_vid_k[ci]);
            }
            if (vp_fused_v[ci] && tc_vid_v[ci]) {
                ggml_backend_tensor_copy_async(backend, backend, vp_fused_v[ci], tc_vid_v[ci]);
            }
        }
    }

    // Upload tc_text (state-dependent, changes every call)
    ggml_backend_tensor_set_async(backend, tc_text, action_text_ctx_f32.data(), 0, ggml_nbytes(tc_text));

    // Build denoise graph if needed (should already exist from previous predict()/compute() calls)
    if (!denoise_graph_valid) {
        return {};  // Graph not built; caller must use predict() to warm up first
    }

    // Upload initial noise and run
    ggml_backend_tensor_set_async(backend, tc_x_raw, raw_noise_f32.data(), 0, ggml_nbytes(tc_x_raw));
    ggml_backend_graph_compute(backend, static_denoise_graph);

    // D2H
    std::vector<float> action_raw_f32((size_t) action_seq * action_dim);
    ggml_backend_tensor_get(tc_x_raw, action_raw_f32.data(), 0, ggml_nbytes(tc_x_raw));

    const auto t_denoise_end = std::chrono::steady_clock::now();

    // Denormalize
    std::vector<float> action_out = action_raw_f32;
    for (int t = 0; t < action_seq; ++t) {
        float * row = action_out.data() + (size_t) t * (size_t) action_dim;
        for (int j = 0; j < action_dim; ++j) {
            const size_t stat_i = (size_t) t * (size_t) action_dim + (size_t) j;
            const float m = (stat_i < action_mean.size()) ? action_mean[stat_i] : 0.f;
            const float s = (stat_i < action_std.size())  ? action_std[stat_i]  : 1.f;
            row[j] = row[j] * (s + 1e-8f) + m;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    stats.ms_total     = ms(t0, t1);
    stats.ms_prefill   = 0.f;   // prefill ran in prepare()
    stats.ms_denoise   = ms(t_denoise_start, t_denoise_end);
    stats.ms_inference = stats.ms_denoise;

    return action_out;
}

}  // namespace (anonymous)

// ---------------------------------------------------------------------------
// Factory (not in the anonymous namespace — must be visible to model.cpp)
// ---------------------------------------------------------------------------
std::unique_ptr<Model> fasterwam_create(const std::string & /*mmproj_path*/,
                                         const std::string & ckpt_path,
                                         const std::string & /*config_path*/) {
    if (ckpt_path.size() < 5 ||
        std::strcmp(ckpt_path.c_str() + ckpt_path.size() - 5, ".gguf") != 0) {
        std::fprintf(stderr, "fasterwam: checkpoint must be a .gguf file\n");
        return nullptr;
    }

    gguf_reader_fw g;
    if (!g.open(ckpt_path)) {
        return nullptr;
    }
    if (!g.has_key("fasterwam.architecture") || g.str("fasterwam.architecture") != "fasterwam") {
        std::fprintf(stderr, "fasterwam: GGUF missing or wrong fasterwam.architecture key\n");
        return nullptr;
    }

    auto m = std::make_unique<FasterWAMModel>();

    // Weight dtype
    if (const char * e = std::getenv("VLA_FASTERWAM_WEIGHT_DTYPE")) {
        if (std::strcmp(e, "f32") == 0 || std::strcmp(e, "F32") == 0) {
            m->wtype = GGML_TYPE_F32;
        } else {
            m->wtype = GGML_TYPE_BF16;
        }
    }

    // Backend
#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) {
        m->is_cuda = true;
        std::printf("fasterwam: backend = CUDA (device 0)\n");
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    m->n_threads = (hw == 0) ? 4 : (int) std::min(hw, 8u);
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) {
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("fasterwam: backend = CPU (%d threads)\n", m->n_threads);
    }

    // Load config
    if (!fw_load_cfg(g, m->fw_cfg)) {
        return nullptr;
    }

    // Allow runtime override of inference steps via env var
    if (const char * e = std::getenv("VLA_FASTERWAM_NUM_STEPS")) {
        const int n = std::atoi(e);
        if (n >= 1 && n <= FasterWAMModel::MAX_DENOISE_STEPS) {
            m->fw_cfg.num_inference_steps = n;
        }
    }

    if (!fw_fill_vla_cfg(m->fw_cfg, m->cfg)) {
        return nullptr;
    }

    // Allocate ggml weight context
    ggml_init_params wp{ (size_t) 64 * 1024 * 1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) {
        return nullptr;
    }

    // Load weights
    if (!fw_load_weights(g, *m)) {
        return nullptr;
    }

    // Load normalization stats
    if (!fw_load_stats(g, *m)) {
        return nullptr;
    }

    std::printf("fasterwam: model loaded from %s\n"
                "  video: %d layers h=%d  action: %d layers h=%d  horizon=%d  steps=%d\n",
                ckpt_path.c_str(),
                m->fw_cfg.video_num_layers, m->fw_cfg.video_hidden,
                m->fw_cfg.action_num_layers, m->fw_cfg.action_hidden,
                m->fw_cfg.action_horizon, m->fw_cfg.num_inference_steps);

    return m;
}

}  // namespace vla
