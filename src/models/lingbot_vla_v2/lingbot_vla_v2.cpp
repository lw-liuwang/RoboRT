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
 * @file lingbot_vla_v2.cpp
 * @brief LingBot-VLA-v2-6B (Robbyant) architecture implementation: model
 *        struct, weight loading, two-phase inference, factory.
 *
 * Pipeline (mirrors the HF reference modeling_lingbot_vla.py/v2):
 *   1. prepare(): Qwen3-VL vision tower (mtmd clip, qwen3vl_merger projector
 *      with fused deepstack outputs) on 3x256x256 views + language embedding
 *      row lookup.  The clip output is [192 tokens, 10240]: per token the first
 *      2560 values are the merger embedding, the remaining 3x2560 are the
 *      deepstack ViT-5/11/17 features.
 *   2. compute(): host-side prefix assembly (3x[vs,64 img tokens,ve] + lang +
 *      8 current + 8 future task queries), prefix pass over the VLM tower
 *      (causal mask, IMROPE, KV cache), then a 10-step Euler denoise loop of
 *      the action expert over the 51-token suffix (1 state + 50 action) with
 *      the prefix K/V concatenated each step, finally de-normalisation.
 *
 * Env overrides:
 *   VLA_LINGBOT_V2_F32_WEIGHTS   keep all matmul weights in F32.
 *   VLA_LINGBOT_V2_FLASH=1       attention via ggml_flash_attn_ext (CUDA FA).
 *   VLA_LINGBOT_V2_FA_F16=0      (FA only) keep the joint K/V cache in F32
 *                                instead of the default bit-exact F16.
 *   VLA_LINGBOT_V2_FA_COPY=1     (FA only) materialise the Q/K/V permutes
 *                                instead of relying on the kernel strides.
 *   VLA_LINGBOT_V2_NUM_STEPS=N   denoising steps (default 10).
 *   VLA_LINGBOT_V2_SEED=N        deterministic initial noise.
 *   VLA_LINGBOT_V2_NO_CACHE=1    rebuild the compute graph every call.
 *   VLA_LINGBOT_V2_DUMP_DIR=path dump alignment tensors as .npy files
 *                                (key names mirror export_lingbot_vla_v2_
 *                                reference.py; compare with compare_lingbot_
 *                                vla_v2_ref.py).
 */

#include "clip.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model.h"
#include "lingbot_vla_v2_gguf.h"
#include "lingbot_vla_v2_graph.h"
#include "lingbot_vla_v2_npy.h"
#include "lingbot_vla_v2_weights.h"
#ifdef GGML_USE_CUDA
#    include "ggml-cuda.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace vla {

// ────────────────────────────────────────────────────────────────────────────
// Model constants (lingbot-vla-v2-6b / robotwin embodiment)
// ────────────────────────────────────────────────────────────────────────────
namespace {

constexpr int    kNViews         = 3;    // camera views per request.
constexpr int64_t kVidTokens     = 64;   // merger tokens per view (256/16/2)^2.
constexpr int64_t kViewTokens    = 66;   // vs + 64 img tokens + ve.
constexpr int64_t kNumTaskTokens = 8;    // query bank rows (current / future).
constexpr int64_t kSuffixState   = 1;    // state token in the suffix.
constexpr int     kVisionStartId = 151652;
constexpr int     kVisionEndId   = 151653;
constexpr int64_t kNDeepstack    = 3;    // ViT layers 5/11/17 -> LLM layers 0/1/2.

/// Sinusoidal time embedding (pi05-compatible: [sin..., cos...] layout).
std::vector<float> sinusoidal_time_emb(double t, int64_t dim, double min_p, double max_p) {
    const int64_t      half = dim / 2;
    std::vector<float> out(dim);
    for (int64_t i = 0; i < half; ++i) {
        const double frac   = (half == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(half - 1);
        const double period = min_p * std::pow(max_p / min_p, frac);
        const double s      = (2.0 * M_PI / period) * t;
        out[i]              = static_cast<float>(std::sin(s));
        out[half + i]       = static_cast<float>(std::cos(s));
    }
    return out;
}

}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// LingBotVlaV2Model
// ────────────────────────────────────────────────────────────────────────────

struct LingBotVlaV2Model final : public Model {
    ~LingBotVlaV2Model() override;

    const Config & config() const override { return cfg; }

    const Stats & last_stats() const override { return stats; }

    std::vector<float> predict(const Inputs & in) override;

    PreparedInput prepare(const Inputs & in) override { return prepare_inputs(in); }

    std::vector<float> compute(const PreparedInput & p) override { return compute_actions(p); }

    PreparedInput      prepare_inputs(const Inputs & in);
    std::vector<float> compute_actions(const PreparedInput & p);
    void               dump_debug(const PreparedInput & p, int64_t n_prefix, int64_t n_suf);

    Stats      stats{};
    Config     cfg{};
    std::mutex stats_mu_;

    clip_ctx *            cctx       = nullptr;
    ggml_backend_t        backend    = nullptr;
    bool                  is_cuda    = false;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_context *        ctx_weights = nullptr;
    std::string           ckpt_path_;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    std::vector<lingbot_vla_v2::VlmLayerW>    vlm_layers;
    std::vector<lingbot_vla_v2::ExpertLayerW> ex_layers;
    ggml_tensor *                            aex_output_norm = nullptr;

    // flow-matching heads
    ggml_tensor * W_state = nullptr, *b_state = nullptr;  // state_proj 55->768
    ggml_tensor * W_ain   = nullptr, *b_ain   = nullptr;  // action_in_proj 55->768
    ggml_tensor * W_mi    = nullptr, *b_mi    = nullptr;  // action_time_mlp_in 1536->768
    ggml_tensor * W_mo    = nullptr, *b_mo    = nullptr;  // action_time_mlp_out 768->768
    ggml_tensor * W_aout  = nullptr, *b_aout  = nullptr;  // action_out_proj 768->55

    // prefix task-query banks (frozen projections, [2560, 8] each)
    ggml_tensor * q_cur = nullptr;
    ggml_tensor * q_fut = nullptr;

    // ── Cached compute graph ─────────────────────────────────────────────
    ggml_context *                comp_ctx_ = nullptr;
    ggml_cgraph *                 gf_       = nullptr;
    ggml_gallocr_t                galloc_   = nullptr;
    ggml_tensor *                 t_prefix_emb_ = nullptr;  // [2560, P]
    std::vector<ggml_tensor *>    t_ds_;                    // 3x [2560, P]
    ggml_tensor *                 t_prefix_pos_ = nullptr;  // I32 [4P]
    ggml_tensor *                 t_suffix_pos_ = nullptr;  // I32 [4*51]
    ggml_tensor *                 t_prefix_mask_ = nullptr; // F32 [P, P]
    ggml_tensor *                 t_suffix_mask_ = nullptr; // F32 [P+51, 51]
    ggml_tensor *                 t_state_ = nullptr;       // F32 [55, 1]
    ggml_tensor *                 t_x0_    = nullptr;       // F32 [55, 50]
    std::vector<ggml_tensor *>    t_time_;                  // 10x F32 [768, 1]
    ggml_tensor *                 x_final_ = nullptr;
    int64_t                       c_n_lang_ = -1;

    // ── Normalisation statistics (bounds_99_woclip, 55-dim canonical) ────
    std::vector<float> state_q01, state_q99, action_q01, action_q99;

    // ── Host-stashed proprioception from predict() (normalised on use) ────
    std::vector<float> last_state_;  // [real_state_dim]

    // ── Host-cached token embedding rows (vision start/end) ──────────────
    std::vector<float> vs_emb_, ve_emb_;  // 2560 each

    // ── Layer / MoE parameters from KV ────────────────────────────────────
    lingbot_vla_v2::LayerParams lp{};
    lingbot_vla_v2::MoeParams   mp{};

    // ── Prefix mask mode: causal (robotwin.yaml) or bidirectional ────────
    bool vlm_causal_ = true;

    // ── Raw (robot) dim -> canonical 55-dim offset maps ──────────────────
    // RoboTwin raw order [armL6, gripL, armR6, gripR]: arm->canon[0:12],
    // grippers->canon[28:30].  Identity fallback when the GGUF has no maps.
    std::vector<int32_t> state_raw_map_;
    std::vector<int32_t> action_raw_map_;

    // ── Numerical-alignment dumps (VLA_LINGBOT_V2_DUMP_DIR) ──────────────
    // Taps registered while building the graph; keep-alive views (appended
    // last) stop the gallocr from recycling the tapped buffers, so the data
    // survives until after ggml_backend_graph_compute and is then written
    // to .npy files named after the reference export keys.
    struct DebugTap {
        std::string   name;
        ggml_tensor * t;
        bool          kv_layout;  ///< [hd, nkv, seq] dumped as (seq, nkv*hd).
    };
    std::vector<DebugTap> dbg_taps_;
    std::string           dump_dir_;
    ggml_tensor *         vlm_output_norm = nullptr;  ///< vlm.output_norm.weight (dump-only).

    // ── Noise RNG ─────────────────────────────────────────────────────────
    std::mt19937 rng{ []() -> std::mt19937::result_type {
        const char * e = std::getenv("VLA_LINGBOT_V2_SEED");
        return e ? static_cast<std::mt19937::result_type>(std::strtoul(e, nullptr, 10)) : std::random_device{}();
    }() };

    int n_threads = 4;
};

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────
namespace {

bool load_config(const lingbot_vla_v2::GgufReader & g, Config & cfg, LingBotVlaV2Model & m) {
    const char * P = "lingbot_vla_v2.";
    auto need = [&](const char * k) {
        if (!g.has_key(k)) {
            std::fprintf(stderr, "lingbot_vla_v2: gguf missing key %s\n", k);
            return false;
        }
        return true;
    };
    for (const char * k : { "lingbot_vla_v2.hidden", "lingbot_vla_v2.n_q_heads", "lingbot_vla_v2.n_kv_heads",
                            "lingbot_vla_v2.head_dim", "lingbot_vla_v2.n_layers", "lingbot_vla_v2.expert_h",
                            "lingbot_vla_v2.moe_n_experts", "lingbot_vla_v2.moe_top_k", "lingbot_vla_v2.chunk_size",
                            "lingbot_vla_v2.num_steps", "lingbot_vla_v2.max_state_dim", "lingbot_vla_v2.max_action_dim",
                            "lingbot_vla_v2.time_dim", "lingbot_vla_v2.min_period", "lingbot_vla_v2.max_period",
                            "lingbot_vla_v2.state_arm_dim", "lingbot_vla_v2.state_effector_dim" }) {
        if (!need(k)) {
            return false;
        }
    }

    cfg            = Config{};
    cfg.hidden     = g.u32("lingbot_vla_v2.hidden");
    cfg.intermediate = g.has_key("lingbot_vla_v2.intermediate") ? g.u32("lingbot_vla_v2.intermediate") : 9728;
    cfg.n_q_heads  = g.u32("lingbot_vla_v2.n_q_heads");
    cfg.n_kv_heads = g.u32("lingbot_vla_v2.n_kv_heads");
    cfg.head_dim   = g.u32("lingbot_vla_v2.head_dim");
    cfg.n_layers   = g.u32("lingbot_vla_v2.n_layers");
    cfg.expert_h   = g.u32("lingbot_vla_v2.expert_h");
    cfg.expert_inter = g.has_key("lingbot_vla_v2.moe_intermediate") ? g.u32("lingbot_vla_v2.moe_intermediate") : 512;

    cfg.n_img       = kVidTokens;
    cfg.n_lang      = 0;  // per-request
    cfg.n_state     = kSuffixState;
    cfg.n_suffix    = g.u32("lingbot_vla_v2.chunk_size");
    cfg.num_steps   = g.u32("lingbot_vla_v2.num_steps");

    if (const char * ns = std::getenv("VLA_LINGBOT_V2_NUM_STEPS");
        ns != nullptr && std::atoi(ns) >= 1 && std::atoi(ns) <= 10) {
        std::printf("lingbot_vla_v2: num_steps overridden by env: %d -> %d\n", cfg.num_steps, std::atoi(ns));
        cfg.num_steps = static_cast<uint32_t>(std::atoi(ns));
    }

    cfg.max_state_dim  = g.u32("lingbot_vla_v2.max_state_dim");
    cfg.max_action_dim = g.u32("lingbot_vla_v2.max_action_dim");
    const uint32_t arm_dim = g.u32("lingbot_vla_v2.state_arm_dim");
    const uint32_t eff_dim = g.u32("lingbot_vla_v2.state_effector_dim");
    cfg.real_state_dim  = arm_dim + eff_dim;
    cfg.real_action_dim = arm_dim + eff_dim;
    cfg.n_prefix = 0;  // per-request
    cfg.n_full   = 0;

    cfg.min_period = g.f64("lingbot_vla_v2.min_period");
    cfg.max_period = g.f64("lingbot_vla_v2.max_period");
    m.vlm_causal_  = g.has_key("lingbot_vla_v2.vlm_causal") ? (g.u32("lingbot_vla_v2.vlm_causal") != 0) : true;

    // Raw (robot) -> canonical dim maps; identity fallback for embodiments
    // whose live dims already sit at the canonical front.
    m.state_raw_map_  = g.arr_i32("lingbot_vla_v2.state_raw_map");
    m.action_raw_map_ = g.arr_i32("lingbot_vla_v2.action_raw_map");
    for (auto * mp : { &m.state_raw_map_, &m.action_raw_map_ }) {
        if (mp->size() != static_cast<size_t>(cfg.real_state_dim)) {
            if (!mp->empty()) {
                std::fprintf(stderr, "lingbot_vla_v2: raw map size mismatch (%zu vs %lld) - using identity\n",
                             mp->size(), static_cast<long long>(cfg.real_state_dim));
            }
            mp->resize(cfg.real_state_dim);
            for (int64_t j = 0; j < cfg.real_state_dim; ++j) {
                (*mp)[j] = static_cast<int32_t>(j);
            }
        }
    }
    cfg.rms_eps    = g.has_key("lingbot_vla_v2.rms_norm_eps") ? g.f32("lingbot_vla_v2.rms_norm_eps") : 1e-6f;
    cfg.norm_eps   = g.has_key("lingbot_vla_v2.expert_norm_eps") ? g.f32("lingbot_vla_v2.expert_norm_eps") : 1e-6f;
    cfg.q_full_dim  = cfg.n_q_heads * cfg.head_dim;
    cfg.kv_full_dim = cfg.n_kv_heads * cfg.head_dim;
    cfg.rope_n_dims   = static_cast<int>(cfg.head_dim);
    cfg.rope_mode     = GGML_ROPE_TYPE_IMROPE;
    cfg.rope_freq_base = g.has_key("lingbot_vla_v2.rope_theta")
                             ? static_cast<float>(g.f64("lingbot_vla_v2.rope_theta"))
                             : 5e6f;

    // layer / MoE params for the graph builders
    m.lp.hidden     = cfg.hidden;
    m.lp.n_q_heads  = cfg.n_q_heads;
    m.lp.n_kv_heads = cfg.n_kv_heads;
    m.lp.head_dim   = cfg.head_dim;
    m.lp.rms_eps    = cfg.rms_eps;
    m.lp.rope_theta = cfg.rope_freq_base;
    m.lp.use_bf16   = true;

    m.mp.n_experts = g.u32("lingbot_vla_v2.moe_n_experts");
    m.mp.top_k     = g.u32("lingbot_vla_v2.moe_top_k");
    m.mp.scale     = g.has_key("lingbot_vla_v2.moe_routed_scaling_factor")
                         ? g.f32("lingbot_vla_v2.moe_routed_scaling_factor")
                         : 4.0f;
    m.mp.use_shared_gate = g.has_key("lingbot_vla_v2.moe_use_shared_expert_gate")
                               ? (g.u32("lingbot_vla_v2.moe_use_shared_expert_gate") != 0)
                               : false;
    (void) P;
    return true;
}

bool load_norm_stats(lingbot_vla_v2::GgufReader & g, LingBotVlaV2Model & m) {
    const int64_t dim = m.cfg.max_state_dim;
    m.state_q01.assign(dim, -1.f);
    m.state_q99.assign(dim, 1.f);
    m.action_q01.assign(dim, -1.f);
    m.action_q99.assign(dim, 1.f);

    auto read1d = [&](const char * name, std::vector<float> & dst) {
        const ggml_tensor * t = g.meta(name);
        if (!t || t->ne[0] != dim || !g.read_raw(name, dst.data())) {
            std::fprintf(stderr, "lingbot_vla_v2: norm stat %s missing/mismatched\n", name);
            return false;
        }
        return true;
    };
    if (!read1d("state_q01", m.state_q01) || !read1d("state_q99", m.state_q99) ||
        !read1d("action_q01", m.action_q01) || !read1d("action_q99", m.action_q99)) {
        return false;
    }
    std::printf("lingbot_vla_v2: normalization: bounds_99_woclip (55-dim, live=%lld)\n",
                static_cast<long long>(m.cfg.real_state_dim));
    return true;
}

}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ────────────────────────────────────────────────────────────────────────────

LingBotVlaV2Model::~LingBotVlaV2Model() {
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

// ────────────────────────────────────────────────────────────────────────────
// Factory
// ────────────────────────────────────────────────────────────────────────────

std::unique_ptr<Model> lingbot_vla_v2_create(const std::string & mmproj_path,
                                             const std::string & ckpt_path,
                                             const std::string & config_path) {
    (void) config_path;

    if (ckpt_path.size() < 5 || ckpt_path.compare(ckpt_path.size() - 5, 5, ".gguf") != 0) {
        std::fprintf(stderr, "lingbot_vla_v2: ckpt must be a GGUF from convert_lingbot_vla_v2_to_gguf.py\n");
        return nullptr;
    }

    auto m         = std::make_unique<LingBotVlaV2Model>();
    m->ckpt_path_  = ckpt_path;
    m->matmul_type = GGML_TYPE_BF16;
    if (std::getenv("VLA_LINGBOT_V2_F32_WEIGHTS")) {
        m->matmul_type = GGML_TYPE_F32;
        m->lp.use_bf16 = false;
    }
    if (const char * dd = std::getenv("VLA_LINGBOT_V2_DUMP_DIR")) {
        m->dump_dir_ = dd;
    }
    if (std::getenv("VLA_LINGBOT_V2_FLASH")) {
        m->lp.use_flash = true;
    }

    lingbot_vla_v2::GgufReader g;
    if (!g.open(ckpt_path)) {
        return nullptr;
    }
    if (!g.has_key("lingbot_vla_v2.architecture") || g.str("lingbot_vla_v2.architecture") != "lingbot_vla_v2") {
        std::fprintf(stderr, "lingbot_vla_v2: '%s' is not a lingbot_vla_v2 GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, m->cfg, *m)) {
        return nullptr;
    }
    const Config & cfg = m->cfg;
    std::printf("lingbot_vla_v2: hidden=%lld heads=%lldq/%lldkv x%lld n_layers=%lld expert_h=%lld "
                "moe=%lldx%lld(scale %.1f) chunk=%lld steps=%d real=%lld/%lld matmul=%s flash=%d\n",
                static_cast<long long>(cfg.hidden), static_cast<long long>(cfg.n_q_heads),
                static_cast<long long>(cfg.n_kv_heads), static_cast<long long>(cfg.head_dim),
                static_cast<long long>(cfg.n_layers), static_cast<long long>(cfg.expert_h),
                static_cast<long long>(m->mp.n_experts), static_cast<long long>(m->mp.top_k), m->mp.scale,
                static_cast<long long>(cfg.n_suffix), cfg.num_steps, static_cast<long long>(cfg.real_state_dim),
                static_cast<long long>(cfg.real_action_dim),
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16", m->lp.use_flash ? 1 : 0);

    // ── Backend selection ─────────────────────────────────────────────────
#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) {
        m->is_cuda = true;
        std::printf("lingbot_vla_v2: backend = CUDA (device 0)\n");
    } else {
        std::fprintf(stderr, "lingbot_vla_v2: ggml_backend_cuda_init failed; falling back to CPU\n");
    }
#endif
    {
        const unsigned hw = std::thread::hardware_concurrency();
        m->n_threads      = (hw == 0) ? 4 : static_cast<int>(std::min(hw, 8u));
    }
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) {
            std::fprintf(stderr, "lingbot_vla_v2: ggml_backend_cpu_init failed\n");
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("lingbot_vla_v2: backend = CPU (%d threads)\n", m->n_threads);
    }

    // ── Vision tower (Qwen3-VL + qwen3vl_merger projector) ────────────────
    {
        clip_context_params cp = {};
        cp.use_gpu             = m->is_cuda;
        cp.flash_attn_type     = m->is_cuda ? CLIP_FLASH_ATTN_TYPE_AUTO : CLIP_FLASH_ATTN_TYPE_DISABLED;
        cp.image_min_tokens    = -1;
        cp.image_max_tokens    = -1;
        cp.warmup              = m->is_cuda;
        cp.cb_eval             = nullptr;
        cp.cb_eval_user_data   = nullptr;
        clip_init_result r     = clip_init(mmproj_path.c_str(), cp);
        if (!r.ctx_v) {
            std::fprintf(stderr, "lingbot_vla_v2: clip_init failed for %s\n", mmproj_path.c_str());
            return nullptr;
        }
        m->cctx          = r.ctx_v;
        const int img_sz = clip_get_image_size(m->cctx);
        const int mm_dim = clip_n_mmproj_embd(m->cctx);
        // mmproj output = 2560 * (1 + 3 deepstack) per token.
        if (img_sz != 256 || mm_dim != static_cast<int>(cfg.hidden * (1 + kNDeepstack))) {
            std::fprintf(stderr,
                         "lingbot_vla_v2: mmproj mismatch (image_size=%d mmproj_embd=%d; want 256/%lld)\n", img_sz,
                         mm_dim, static_cast<long long>(cfg.hidden * (1 + kNDeepstack)));
            return nullptr;
        }
        const int64_t per_view_tokens =
            static_cast<int64_t>(clip_embd_nbytes_by_img(m->cctx, 256, 256) / sizeof(float)) / mm_dim;
        if (per_view_tokens != kVidTokens) {
            std::fprintf(stderr, "lingbot_vla_v2: unexpected %lld vision tokens per view (want %lld)\n",
                         static_cast<long long>(per_view_tokens), static_cast<long long>(kVidTokens));
            return nullptr;
        }
    }

    // ── Weight context ────────────────────────────────────────────────────
    {
        ggml_init_params wp = { static_cast<size_t>(32) * 1024 * 1024, nullptr, true };
        m->ctx_weights      = ggml_init(wp);
        if (!m->ctx_weights) {
            std::fprintf(stderr, "lingbot_vla_v2: ggml_init(ctx_weights) failed\n");
            return nullptr;
        }
    }
    ggml_context *             W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "lingbot_vla_v2: missing tensor %s\n", name);
            return nullptr;
        }
        ggml_tensor * t = ggml_new_tensor(W, type, GGML_MAX_DIMS, gt->ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };
    auto mk_mm = [&](const char * name) -> ggml_tensor * { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) -> ggml_tensor * { return mk(name, GGML_TYPE_F32); };

    const int64_t n_layers = cfg.n_layers;

    auto load_vlm_layer = [&](int i, lingbot_vla_v2::VlmLayerW & lw) -> bool {
        char b[256];
        auto suf = [&](const char * s) {
            std::snprintf(b, sizeof(b), "vlm.blk.%d.%s", i, s);
            return b;
        };
        lw.attn_norm = mk_f32(suf("attn_norm.weight"));
        lw.Wq        = mk_mm(suf("attn_q.weight"));
        lw.q_norm    = mk_f32(suf("attn_q_norm.weight"));
        lw.Wk        = mk_mm(suf("attn_k.weight"));
        lw.k_norm    = mk_f32(suf("attn_k_norm.weight"));
        lw.Wv        = mk_mm(suf("attn_v.weight"));
        lw.Wo        = mk_mm(suf("attn_o.weight"));
        lw.ffn_norm  = mk_f32(suf("ffn_norm.weight"));
        lw.Wgate     = mk_mm(suf("ffn_gate.weight"));
        lw.Wup       = mk_mm(suf("ffn_up.weight"));
        lw.Wdown     = mk_mm(suf("ffn_down.weight"));
        return lw.attn_norm && lw.Wq && lw.q_norm && lw.Wk && lw.k_norm && lw.Wv && lw.Wo && lw.ffn_norm &&
               lw.Wgate && lw.Wup && lw.Wdown;
    };

    auto load_expert_layer = [&](int i, lingbot_vla_v2::ExpertLayerW & lw) -> bool {
        char b[256];
        auto suf = [&](const char * s) {
            std::snprintf(b, sizeof(b), "aex.blk.%d.%s", i, s);
            return b;
        };
        auto adanorm = [&](const char * norm_name, lingbot_vla_v2::AdanormW & aw) -> bool {
            aw.w       = mk_f32((std::string(suf(norm_name)) + ".weight").c_str());
            aw.gamma_w = mk_mm((std::string(suf(norm_name)) + ".gamma_w").c_str());
            aw.gamma_b = mk_f32((std::string(suf(norm_name)) + ".gamma_b").c_str());
            aw.beta_w  = mk_mm((std::string(suf(norm_name)) + ".beta_w").c_str());
            aw.beta_b  = mk_f32((std::string(suf(norm_name)) + ".beta_b").c_str());
            return aw.w && aw.gamma_w && aw.gamma_b && aw.beta_w && aw.beta_b;
        };
        if (!adanorm("attn_norm", lw.attn) || !adanorm("ffn_norm", lw.ffn)) {
            return false;
        }
        lw.Wq = mk_mm(suf("attn_q.weight"));
        lw.bq = mk_f32(suf("attn_q.bias"));
        lw.Wk = mk_mm(suf("attn_k.weight"));
        lw.bk = mk_f32(suf("attn_k.bias"));
        lw.Wv = mk_mm(suf("attn_v.weight"));
        lw.bv = mk_f32(suf("attn_v.bias"));
        lw.Wo = mk_mm(suf("attn_o.weight"));
        lw.moe_gate     = mk_f32(suf("moe_gate.weight"));
        lw.experts_bias = mk_f32(suf("experts_bias"));
        lw.exp_gate     = mk_mm(suf("experts_gate.weight"));
        lw.exp_up       = mk_mm(suf("experts_up.weight"));
        lw.exp_down     = mk_mm(suf("experts_down.weight"));
        lw.sh_gate      = mk_mm(suf("shared_expert_gate.weight"));
        lw.sh_up        = mk_mm(suf("shared_expert_up.weight"));
        lw.sh_down      = mk_mm(suf("shared_expert_down.weight"));
        // Optional sigmoid gate (use_shared_expert_gate); absent -> ungated.
        lw.sh_router    = m->mp.use_shared_gate ? mk_f32(suf("shared_expert_router.weight")) : nullptr;
        return lw.Wq && lw.bq && lw.Wk && lw.bk && lw.Wv && lw.bv && lw.Wo && lw.moe_gate && lw.experts_bias &&
               lw.exp_gate && lw.exp_up && lw.exp_down && lw.sh_gate && lw.sh_up && lw.sh_down &&
               (!m->mp.use_shared_gate || lw.sh_router);
    };

    m->vlm_layers.resize(n_layers);
    m->ex_layers.resize(n_layers);
    for (int64_t i = 0; i < n_layers; ++i) {
        if (!load_vlm_layer(static_cast<int>(i), m->vlm_layers[i]) ||
            !load_expert_layer(static_cast<int>(i), m->ex_layers[i])) {
            return nullptr;
        }
    }

    m->aex_output_norm = mk_f32("aex.output_norm.weight");
    m->W_state         = mk_f32("state_proj.weight");
    m->b_state         = mk_f32("state_proj.bias");
    m->W_ain           = mk_f32("action_in_proj.weight");
    m->b_ain           = mk_f32("action_in_proj.bias");
    m->W_mi            = mk_f32("action_time_mlp_in.weight");
    m->b_mi            = mk_f32("action_time_mlp_in.bias");
    m->W_mo            = mk_f32("action_time_mlp_out.weight");
    m->b_mo            = mk_f32("action_time_mlp_out.bias");
    m->W_aout          = mk_f32("action_out_proj.weight");
    m->b_aout          = mk_f32("action_out_proj.bias");
    m->q_cur           = mk_f32("prefix_query_current");
    m->q_fut           = mk_f32("prefix_query_future");
    // VLM final norm: only needed by the alignment dumps (the prefix pass
    // output is otherwise unused once the KV cache is filled).
    if (g.meta("vlm.output_norm.weight")) {
        m->vlm_output_norm = mk_f32("vlm.output_norm.weight");
    }

    for (ggml_tensor * t : weights) {
        if (!t) {
            std::fprintf(stderr, "lingbot_vla_v2: weight tensor creation failed\n");
            return nullptr;
        }
    }
    if (!m->aex_output_norm || !m->W_state || !m->b_state || !m->W_ain || !m->b_ain || !m->W_mi || !m->b_mi ||
        !m->W_mo || !m->b_mo || !m->W_aout || !m->b_aout || !m->q_cur || !m->q_fut) {
        std::fprintf(stderr, "lingbot_vla_v2: failed to wire projection/query tensors\n");
        return nullptr;
    }

    // ── Host-side constants: vision start/end embedding rows ──────────────
    {
        m->vs_emb_.assign(cfg.hidden, 0.f);
        m->ve_emb_.assign(cfg.hidden, 0.f);
        if (!g.fetch_rows_f32("token_embd.weight", { kVisionStartId }, m->vs_emb_.data(), cfg.hidden) ||
            !g.fetch_rows_f32("token_embd.weight", { kVisionEndId }, m->ve_emb_.data(), cfg.hidden)) {
            std::fprintf(stderr, "lingbot_vla_v2: cannot fetch vision token embeddings\n");
            return nullptr;
        }
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) {
        std::fprintf(stderr, "lingbot_vla_v2: ggml_backend_alloc_ctx_tensors failed (OOM?)\n");
        return nullptr;
    }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "lingbot_vla_v2: upload size mismatch for %s (%zu vs %zu)\n", t->name, bytes.size(),
                         ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("lingbot_vla_v2: resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    if (!load_norm_stats(g, *m)) {
        return nullptr;
    }
    std::printf("lingbot_vla_v2: model loaded (n_threads=%d)\n", m->n_threads);
    return m;
}

// ────────────────────────────────────────────────────────────────────────────
// prepare_inputs: vision tower + language embedding lookup
// ────────────────────────────────────────────────────────────────────────────

PreparedInput LingBotVlaV2Model::prepare_inputs(const Inputs & in) {
    using clk = std::chrono::high_resolution_clock;
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats = Stats{};
    }

    const Config & cfg = this->cfg;

    PreparedInput p;

    // ── Vision tower (3 views, 256x256) ───────────────────────────────────
    if (in.precomputed_img_emb) {
        if (in.n_img_views != kNViews) {
            std::fprintf(stderr, "lingbot_vla_v2: expected %d views, got %d\n", kNViews, in.n_img_views);
            p.ok    = false;
            p.error = "bad view count";
            return p;
        }
        p.n_img_tokens = static_cast<int64_t>(kNViews) * kVidTokens;
        p.img_emb_host.assign(in.precomputed_img_emb,
                              in.precomputed_img_emb + static_cast<size_t>(p.n_img_tokens) *
                                                             (cfg.hidden * (1 + kNDeepstack)));
    } else {
        if (in.n_images != kNViews || !in.images) {
            std::fprintf(stderr, "lingbot_vla_v2: prepare: need exactly %d images (got %d)\n", kNViews, in.n_images);
            p.ok    = false;
            p.error = "need exactly 3 images";
            return p;
        }
        const int     img_sz    = clip_get_image_size(cctx);
        const size_t  per_pix   = static_cast<size_t>(3) * img_sz * img_sz;
        const int     mm_dim    = clip_n_mmproj_embd(cctx);
        const size_t  per_out   = clip_embd_nbytes_by_img(cctx, img_sz, img_sz) / sizeof(float);
        const int64_t per_tok   = static_cast<int64_t>(per_out) / mm_dim;

        std::vector<float> hwc(per_pix);
        const auto         tv0 = clk::now();
        p.img_emb_host.clear();
        p.img_emb_host.reserve(per_out * kNViews);

        for (int v = 0; v < kNViews; ++v) {
            const ImageView & view = in.images[v];
            if (view.w != img_sz || view.h != img_sz) {
                std::fprintf(stderr, "lingbot_vla_v2: image[%d] is %dx%d; requires %dx%d\n", v, view.w, view.h, img_sz,
                             img_sz);
                p.ok    = false;
                p.error = "bad image size";
                return p;
            }
            // Qwen3-VL normalisation (mean=std=0.5): u8/127.5 - 1.
            if (view.format == PixelFormat::U8) {
                const uint8_t * src = static_cast<const uint8_t *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) {
                    hwc[i] = static_cast<float>(src[i]) / 127.5f - 1.f;
                }
            } else {
                const float * src = static_cast<const float *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) {
                    hwc[i] = src[i] * 2.f - 1.f;
                }
            }

            std::vector<float> view_emb(per_out);
            if (!clip_encode_float_image(cctx, n_threads, hwc.data(), img_sz, img_sz, view_emb.data())) {
                std::fprintf(stderr, "lingbot_vla_v2: clip_encode_float_image failed (view %d)\n", v);
                p.ok    = false;
                p.error = "clip_encode_float_image failed";
                return p;
            }
            p.img_emb_host.insert(p.img_emb_host.end(), view_emb.begin(), view_emb.end());
        }
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - tv0).count();
        }
        p.n_img_tokens = static_cast<int64_t>(kNViews) * per_tok;
    }

    // ── Language embedding lookup ─────────────────────────────────────────
    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "lingbot_vla_v2: prepare: empty lang_tokens\n");
        p.ok    = false;
        p.error = "empty lang_tokens";
        return p;
    }
    const int64_t        n_lang = in.n_lang;
    std::vector<int32_t> lang_ids(in.lang_tokens, in.lang_tokens + n_lang);
    p.lang_rows.resize(static_cast<size_t>(n_lang) * cfg.hidden);
    {
        lingbot_vla_v2::GgufReader g2;
        if (!g2.open(ckpt_path_)) {
            p.ok    = false;
            p.error = "gguf_reader open failed";
            return p;
        }
        if (!g2.fetch_rows_f32("token_embd.weight", lang_ids, p.lang_rows.data(), cfg.hidden)) {
            p.ok    = false;
            p.error = "token_embd.weight lookup failed";
            return p;
        }
    }
    p.n_lang = n_lang;

    if (in.noise) {
        p.noise.assign(in.noise, in.noise + static_cast<size_t>(cfg.max_action_dim) * cfg.n_suffix);
    }
    p.timing_detail = in.timing_detail;
    p.ok            = true;
    return p;
}

// ────────────────────────────────────────────────────────────────────────────
// compute_actions: host assembly + cached graph + denoise + de-normalise
// ────────────────────────────────────────────────────────────────────────────

std::vector<float> LingBotVlaV2Model::compute_actions(const PreparedInput & p) {
    using clk = std::chrono::high_resolution_clock;

    const Config & cfg       = this->cfg;
    const int64_t  hidden    = cfg.hidden;
    const int64_t  chunk     = cfg.n_suffix;
    const int64_t  n_suf     = kSuffixState + chunk;  // 51
    const int64_t  n_layers  = cfg.n_layers;
    const int64_t  max_ad    = cfg.max_action_dim;
    const int      num_steps = cfg.num_steps;
    const float    dt        = -1.f / static_cast<float>(num_steps);

    const int64_t n_lang   = p.n_lang;
    const int64_t n_prefix = kNViews * kViewTokens + n_lang + 2 * kNumTaskTokens;  // 214 + n

    // ── Cached compute graph ──────────────────────────────────────────────
    static const bool no_cache = []() {
        const char * e = std::getenv("VLA_LINGBOT_V2_NO_CACHE");
        return e && std::atoi(e) > 0;
    }();
    const bool need_rebuild = no_cache || !gf_ || c_n_lang_ != n_lang;

    if (need_rebuild) {
        if (galloc_) {
            ggml_gallocr_free(galloc_);
            galloc_ = nullptr;
        }
        if (comp_ctx_) {
            ggml_free(comp_ctx_);
            comp_ctx_ = nullptr;
        }
        gf_            = nullptr;
        x_final_       = nullptr;
        t_prefix_emb_  = nullptr;
        t_ds_.clear();
        t_prefix_pos_ = t_suffix_pos_ = t_prefix_mask_ = t_suffix_mask_ = nullptr;
        t_state_ = t_x0_ = nullptr;
        t_time_.clear();
        dbg_taps_.clear();

        ggml_init_params cp = { static_cast<size_t>(192) * 1024 * 1024, nullptr, true };
        comp_ctx_           = ggml_init(cp);
        if (!comp_ctx_) {
            std::fprintf(stderr, "lingbot_vla_v2: ggml_init(ctx_compute) failed\n");
            return {};
        }
        ggml_context * C = comp_ctx_;

        // Alignment-dump taps (no-op unless VLA_LINGBOT_V2_DUMP_DIR is set).
        const bool dump = !dump_dir_.empty();
        auto tap = [&](const std::string & name, ggml_tensor * t, bool kv_layout = false) {
            if (!dump || !t || !ggml_is_contiguous(t)) {
                return;
            }
            dbg_taps_.push_back({ name, t, kv_layout });
        };

        t_prefix_emb_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, n_prefix);
        ggml_set_input(t_prefix_emb_);
        t_ds_.resize(kNDeepstack);
        for (int64_t l = 0; l < kNDeepstack; ++l) {
            t_ds_[l] = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, n_prefix);
            ggml_set_input(t_ds_[l]);
        }
        t_prefix_pos_ = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4 * n_prefix);
        ggml_set_input(t_prefix_pos_);
        t_suffix_pos_ = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4 * n_suf);
        ggml_set_input(t_suffix_pos_);
        t_prefix_mask_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_prefix, n_prefix);
        ggml_set_input(t_prefix_mask_);
        t_suffix_mask_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_prefix + n_suf, n_suf);
        ggml_set_input(t_suffix_mask_);
        t_state_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, 1);
        ggml_set_input(t_state_);
        t_x0_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);
        ggml_set_input(t_x0_);
        t_time_.resize(num_steps);
        for (int s = 0; s < num_steps; ++s) {
            t_time_[s] = ggml_new_tensor_2d(C, GGML_TYPE_F32, cfg.expert_h, 1);
            ggml_set_input(t_time_[s]);
        }

        // The MoE builders register intermediate views with
        // ggml_build_forward_expand, so the graph must exist before the nodes.
        gf_ = ggml_new_graph_custom(C, 65536, false);

        // ── Prefix pass: VLM tower fills the KV cache ─────────────────────
        std::vector<ggml_tensor *> cK(n_layers);
        std::vector<ggml_tensor *> cV(n_layers);
        // F16 joint K/V caches for the denoise loop (see fa_f16_kv() in
        // lingbot_vla_v2_graph.cpp).  Each is [hd, n_kv, n_prefix + n_suf]: the
        // prefix slice is staged here once per layer, the suffix tail is
        // overwritten by build_expert_layer on every denoise step, so no
        // full-length prefix+suffix concat is ever materialised.  The dumps tap
        // the joint K/V, which only exists on the concat path, so a dump run
        // always builds the F32 cache.
        const bool f16_kv = lp.use_flash && !dump && lingbot_vla_v2::fa_f16_kv();
        std::vector<ggml_tensor *> kv_cK;
        std::vector<ggml_tensor *> kv_cV;
        {
            ggml_tensor * h = t_prefix_emb_;
            for (int64_t il = 0; il < n_layers; ++il) {
                if (il <= 3 || il == n_layers - 1) {
                    tap("prefix.hid_in_L" + std::to_string(il), h);
                }
                h = lingbot_vla_v2::build_vlm_layer(C, vlm_layers[il], h, t_prefix_pos_, t_prefix_mask_, lp,
                                                    n_prefix, f16_kv, &cK[il], &cV[il]);
                tap("prefix.k_L" + std::to_string(il), cK[il], true);
                tap("prefix.v_L" + std::to_string(il), cV[il], true);
                if (il < kNDeepstack) {
                    // Deepstack: ViT-5/11/17 features injected on the image
                    // token positions after the layer output.
                    h = ggml_add(C, h, t_ds_[il]);
                }
                if (cK[il]->type == GGML_TYPE_F16) {
                    ggml_tensor * kc = ggml_new_tensor_3d(C, GGML_TYPE_F16, cK[il]->ne[0], cK[il]->ne[1],
                                                          cK[il]->ne[2] + n_suf);
                    ggml_tensor * vc = ggml_new_tensor_3d(C, GGML_TYPE_F16, cV[il]->ne[0], cV[il]->ne[1],
                                                          cV[il]->ne[2] + n_suf);
                    ggml_tensor * kd = ggml_cpy(C, cK[il],
                        ggml_view_3d(C, kc, cK[il]->ne[0], cK[il]->ne[1], cK[il]->ne[2], kc->nb[1], kc->nb[2], 0));
                    ggml_tensor * vd = ggml_cpy(C, cV[il],
                        ggml_view_3d(C, vc, cV[il]->ne[0], cV[il]->ne[1], cV[il]->ne[2], vc->nb[1], vc->nb[2], 0));
                    ggml_build_forward_expand(gf_, kd);  // side effect: not consumed
                    ggml_build_forward_expand(gf_, vd);
                    kv_cK.push_back(kc);
                    kv_cV.push_back(vc);
                }
            }
            tap("prefix.pre_final_out", h);
        }

        // ── Denoise loop: action expert over the 51-token suffix ──────────
        const bool kv_f16 = !kv_cK.empty();
        ggml_tensor * x_t = t_x0_;
        for (int step = 0; step < num_steps; ++step) {
            tap("s" + std::to_string(step) + ".x_in", x_t);
            ggml_tensor * h = lingbot_vla_v2::build_embed_suffix(C, x_t, t_state_, t_time_[step], W_state, b_state,
                                                                 W_ain, b_ain, W_mi, b_mi, W_mo, b_mo);
            tap("s" + std::to_string(step) + ".suffix_embs", h);
            for (int64_t il = 0; il < n_layers; ++il) {
                if (step == 0) {
                    tap("s0.hid_in_E" + std::to_string(il), h);
                }
                ggml_tensor * kfull = nullptr;
                ggml_tensor * vfull = nullptr;
                // The joint K/V taps only make sense on the concat path (the F16
                // cache holds the last step's suffix once the graph has run).
                const bool tap_kv = step == 0 && dump && !kv_f16;
                h = lingbot_vla_v2::build_expert_layer(C, gf_, ex_layers[il], h, t_suffix_pos_, t_time_[step],
                                                        kv_f16 ? kv_cK[il] : cK[il],
                                                        kv_f16 ? kv_cV[il] : cV[il], t_suffix_mask_, lp, mp, n_suf,
                                                        tap_kv ? &kfull : nullptr,
                                                        tap_kv ? &vfull : nullptr);
                if (step == 0) {
                    tap("s0.k_L" + std::to_string(il), kfull, true);
                    tap("s0.v_L" + std::to_string(il), vfull, true);
                }
            }
            // Final plain RMSNorm over all 51 rows, then drop the state row.
            ggml_tensor * h_final = ggml_mul(C, ggml_rms_norm(C, h, cfg.rms_eps), aex_output_norm);
            const size_t  row_b   = static_cast<size_t>(cfg.expert_h) * sizeof(float);
            ggml_tensor * h50 =
                ggml_view_2d(C, h_final, cfg.expert_h, chunk, h_final->nb[1], row_b);
            ggml_tensor * v_t = ggml_add(C, ggml_mul_mat(C, W_aout, h50), b_aout);
            tap("s" + std::to_string(step) + ".v", v_t);
            x_t               = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
        }
        x_final_ = x_t;
        ggml_set_output(x_final_);
        ggml_build_forward_expand(gf_, x_final_);

        if (!dbg_taps_.empty()) {
            // Keep-alive views: registered after every consumer so the tapped
            // buffers survive to the end of the graph.  gallocr recycles a
            // tensor only when its child count AND view count both hit zero;
            // each view (never freed, owns no storage) pins its source.
            std::vector<ggml_tensor *> keep;
            keep.reserve(dbg_taps_.size() + 8 + t_ds_.size() + t_time_.size());
            for (const auto & d : dbg_taps_) {
                keep.push_back(d.t);
            }
            keep.push_back(t_prefix_emb_);
            for (ggml_tensor * t : t_ds_) {
                keep.push_back(t);
            }
            keep.insert(keep.end(), { t_prefix_pos_, t_suffix_pos_, t_prefix_mask_, t_suffix_mask_, t_state_, t_x0_ });
            keep.insert(keep.end(), t_time_.begin(), t_time_.end());
            for (ggml_tensor * t : keep) {
                if (t && ggml_is_contiguous(t)) {
                    ggml_build_forward_expand(gf_, ggml_view_1d(C, t, 1, 0));
                }
            }
        }

        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf_)) {
            std::fprintf(stderr, "lingbot_vla_v2: ggml_gallocr_alloc_graph failed (OOM?)\n");
            if (galloc_) {
                ggml_gallocr_free(galloc_);
                galloc_ = nullptr;
            }
            ggml_free(comp_ctx_);
            comp_ctx_ = nullptr;
            gf_       = nullptr;
            x_final_  = nullptr;
            return {};
        }
        c_n_lang_ = n_lang;
    }

    // ── Upload inputs (every request) ────────────────────────────────────
    const int mm_dim  = static_cast<int>(hidden * (1 + kNDeepstack));
    const int64_t n_vtok = kVidTokens;  // merger tokens per view

    // Prefix embeddings + deepstack maps.
    {
        std::vector<float> emb(static_cast<size_t>(n_prefix) * hidden, 0.f);
        std::vector<float> ds(static_cast<size_t>(kNDeepstack) * n_prefix * hidden, 0.f);
        const float * clip = p.img_emb_host.data();

        for (int k = 0; k < kNViews; ++k) {
            const int64_t base = static_cast<int64_t>(k) * kViewTokens;
            std::memcpy(emb.data() + static_cast<size_t>(base) * hidden, vs_emb_.data(), hidden * sizeof(float));
            std::memcpy(emb.data() + static_cast<size_t>(base + kViewTokens - 1) * hidden, ve_emb_.data(),
                        hidden * sizeof(float));
            for (int64_t j = 0; j < n_vtok; ++j) {
                // merger output: first `hidden` values of each clip token row
                const float * src = clip + (static_cast<size_t>(k) * n_vtok + j) * mm_dim;
                std::memcpy(emb.data() + static_cast<size_t>(base + 1 + j) * hidden, src, hidden * sizeof(float));
                // deepstack: the next 3 x `hidden` values
                for (int64_t l = 0; l < kNDeepstack; ++l) {
                    std::memcpy(ds.data() +
                                    (static_cast<size_t>(l) * n_prefix + static_cast<size_t>(base + 1 + j)) * hidden,
                                src + static_cast<size_t>(1 + l) * hidden, hidden * sizeof(float));
                }
            }
        }
        std::memcpy(emb.data() + static_cast<size_t>(kNViews * kViewTokens) * hidden, p.lang_rows.data(),
                    static_cast<size_t>(n_lang) * hidden * sizeof(float));
        // task queries: current then future, 8 rows each ([hidden, 8] row-major).
        const size_t q_rows = static_cast<size_t>(kNumTaskTokens) * hidden;
        {
            std::vector<float> qc(q_rows), qf(q_rows);
            ggml_backend_tensor_get(q_cur, qc.data(), 0, q_rows * sizeof(float));
            ggml_backend_tensor_get(q_fut, qf.data(), 0, q_rows * sizeof(float));
            std::memcpy(emb.data() + static_cast<size_t>(kNViews * kViewTokens + n_lang) * hidden, qc.data(),
                        q_rows * sizeof(float));
            std::memcpy(emb.data() + static_cast<size_t>(kNViews * kViewTokens + n_lang + kNumTaskTokens) * hidden,
                        qf.data(), q_rows * sizeof(float));
        }
        ggml_backend_tensor_set(t_prefix_emb_, emb.data(), 0, ggml_nbytes(t_prefix_emb_));
        for (int64_t l = 0; l < kNDeepstack; ++l) {
            ggml_backend_tensor_set(t_ds_[l], ds.data() + static_cast<size_t>(l) * n_prefix * hidden, 0,
                                    ggml_nbytes(t_ds_[l]));
        }
    }

    // Positions: 4 planes per token, the e plane mirrors the t plane.
    {
        std::vector<int32_t> pp(static_cast<size_t>(4) * n_prefix);
        for (int k = 0; k < kNViews; ++k) {
            const int64_t base = static_cast<int64_t>(k) * kViewTokens;
            const int32_t vs_p = 10 * k;
            for (int pl = 0; pl < 4; ++pl) {
                pp[static_cast<size_t>(pl) * n_prefix + base] = vs_p;
                pp[static_cast<size_t>(pl) * n_prefix + base + kViewTokens - 1] = 10 * k + 9;
            }
            for (int64_t j = 0; j < n_vtok; ++j) {
                const int32_t t_v = 10 * k + 1;
                const int32_t h_v = 10 * k + 1 + static_cast<int32_t>(j / 8);
                const int32_t w_v = 10 * k + 1 + static_cast<int32_t>(j % 8);
                const int64_t idx = base + 1 + j;
                pp[idx]                                = t_v;
                pp[static_cast<size_t>(n_prefix) + idx] = h_v;
                pp[static_cast<size_t>(2 * n_prefix) + idx] = w_v;
                pp[static_cast<size_t>(3 * n_prefix) + idx] = t_v;
            }
        }
        const int64_t lang_base = kNViews * kViewTokens;
        for (int64_t i = 0; i < n_lang + 2 * kNumTaskTokens; ++i) {
            const int32_t v = static_cast<int32_t>(30 + i);
            for (int pl = 0; pl < 4; ++pl) {
                pp[static_cast<size_t>(pl) * n_prefix + lang_base + i] = v;
            }
        }
        ggml_backend_tensor_set(t_prefix_pos_, pp.data(), 0, ggml_nbytes(t_prefix_pos_));

        std::vector<int32_t> sp(static_cast<size_t>(4) * n_suf);
        const int32_t        s0 = static_cast<int32_t>(n_lang + 46);
        for (int64_t i = 0; i < n_suf; ++i) {
            for (int pl = 0; pl < 4; ++pl) {
                sp[static_cast<size_t>(pl) * n_suf + i] = s0 + static_cast<int32_t>(i);
            }
        }
        ggml_backend_tensor_set(t_suffix_pos_, sp.data(), 0, ggml_nbytes(t_suffix_pos_));
    }

    // Attention masks (0 = allow, -inf = block).
    {
        const float      ninf = -std::numeric_limits<float>::infinity();
        std::vector<float> pm(static_cast<size_t>(n_prefix) * n_prefix);
        for (int64_t q = 0; q < n_prefix; ++q) {
            for (int64_t kv = 0; kv < n_prefix; ++kv) {
                // vlm_causal (robotwin.yaml): causal prefix.  vlm_causal=false
                // configs use all-zero att_masks -> fully bidirectional prefix.
                pm[static_cast<size_t>(q) * n_prefix + kv] =
                    (!vlm_causal_ || kv <= q) ? 0.f : ninf;
            }
        }
        ggml_backend_tensor_set(t_prefix_mask_, pm.data(), 0, ggml_nbytes(t_prefix_mask_));

        std::vector<float> sm(static_cast<size_t>(n_prefix + n_suf) * n_suf, 0.f);
        for (int64_t q = 0; q < n_suf; ++q) {
            for (int64_t kv = 0; kv < n_prefix + n_suf; ++kv) {
                if (q == 0) {
                    // state row: prefix + itself only
                    sm[static_cast<size_t>(q) * (n_prefix + n_suf) + kv] = (kv <= n_prefix) ? 0.f : ninf;
                }
                // action rows attend to everything (already 0)
            }
        }
        ggml_backend_tensor_set(t_suffix_mask_, sm.data(), 0, ggml_nbytes(t_suffix_mask_));
    }

    // Proprioception: raw robot dims -> canonical 55-dim layout + bounds
    // normalisation.  bounds_99_woclip: (v - q01) / (q99 - q01 + 1e-6) * 2 - 1
    // (stats live in canonical positions; dead dims stay 0).
    {
        std::vector<float> st(max_ad, 0.f);
        if (last_state_.size() == static_cast<size_t>(cfg.real_state_dim)) {
            for (int64_t j = 0; j < cfg.real_state_dim; ++j) {
                const int64_t c    = state_raw_map_[j];
                const float   denom = state_q99[c] - state_q01[c] + 1e-6f;
                st[c]              = 2.f * (last_state_[j] - state_q01[c]) / denom - 1.f;
            }
        }
        ggml_backend_tensor_set(t_state_, st.data(), 0, ggml_nbytes(t_state_));
    }

    // Initial noise x0.
    {
        std::vector<float> x0h(static_cast<size_t>(max_ad) * chunk);
        if (p.noise.size() == static_cast<size_t>(max_ad) * chunk) {
            std::memcpy(x0h.data(), p.noise.data(), x0h.size() * sizeof(float));
        } else {
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto & v : x0h) {
                v = nd(rng);
            }
        }
        ggml_backend_tensor_set(t_x0_, x0h.data(), 0, ggml_nbytes(t_x0_));
    }

    // Time embeddings: t = 1, 0.9, ..., 1/num_steps.
    for (int s = 0; s < num_steps; ++s) {
        const float              timestep = 1.f + static_cast<float>(s) * dt;
        const std::vector<float> tv =
            sinusoidal_time_emb(timestep, cfg.expert_h, cfg.min_period, cfg.max_period);
        ggml_backend_tensor_set(t_time_[s], tv.data(), 0, ggml_nbytes(t_time_[s]));
    }

    // ── Compute + read output ────────────────────────────────────────────
    const auto        ti0 = clk::now();
    const ggml_status st  = ggml_backend_graph_compute(backend, gf_);
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();
    }
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "lingbot_vla_v2: ggml_backend_graph_compute failed (%d)\n", static_cast<int>(st));
        return {};
    }

    if (!dump_dir_.empty() && !dbg_taps_.empty()) {
        dump_debug(p, n_prefix, n_suf);
    }

    // Read the canonical [max_ad, chunk] result, de-normalise and gather the
    // live dims back into the raw robot order.
    // bounds_99_woclip: ((x + 1) / 2) * (q99 - q01 + 1e-6) + q01.
    {
        std::vector<float> canon(static_cast<size_t>(chunk) * max_ad);
        ggml_backend_tensor_get(x_final_, canon.data(), 0, canon.size() * sizeof(float));
        std::vector<float> out(static_cast<size_t>(chunk) * cfg.real_action_dim);
        for (int64_t t = 0; t < chunk; ++t) {
            const float * row = canon.data() + static_cast<size_t>(t) * max_ad;
            float *       dst = out.data() + static_cast<size_t>(t) * cfg.real_action_dim;
            for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
                const int64_t c     = action_raw_map_[j];
                const float   denom = action_q99[c] - action_q01[c] + 1e-6f;
                dst[j]              = (row[c] + 1.f) * denom * 0.5f + action_q01[c];
            }
        }
        return out;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// dump_debug: write alignment tensors (VLA_LINGBOT_V2_DUMP_DIR)
// ────────────────────────────────────────────────────────────────────────────

void LingBotVlaV2Model::dump_debug(const PreparedInput & p, int64_t n_prefix, int64_t n_suf) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dump_dir_, ec);
    if (ec) {
        std::fprintf(stderr, "lingbot_vla_v2: cannot create dump dir %s\n", dump_dir_.c_str());
        return;
    }

    int n_written = 0;
    auto write = [&](const std::string & name, const std::vector<int64_t> & shape,
                     const std::vector<float> & v) {
        if (lingbot_vla_v2::write_npy_f32(dump_dir_ + "/" + name + ".npy", shape, v.data())) {
            ++n_written;
        } else {
            std::fprintf(stderr, "lingbot_vla_v2: dump write failed: %s\n", name.c_str());
        }
    };
    auto getf = [&](ggml_tensor * t) {
        std::vector<float> v(ggml_nbytes(t) / sizeof(float));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };

    // 1) Graph taps.  A contiguous ggml tensor [ne0, ne1, ne2] has the same
    // memory layout as a numpy (ne2, ne1, ne0) C-order array, so the dump is
    // a plain copy with the dims reversed; KV taps collapse (nkv, hd).
    for (const auto & d : dbg_taps_) {
        std::vector<int64_t> shape;
        if (d.kv_layout) {
            shape = { d.t->ne[2], d.t->ne[1] * d.t->ne[0] };
        } else {
            for (int i = 3; i >= 1; --i) {
                if (d.t->ne[i] > 1) {
                    shape.push_back(d.t->ne[i]);
                }
            }
            shape.push_back(d.t->ne[0]);
        }
        write(d.name, shape, getf(d.t));
    }

    // 2) Graph inputs (re-read from the backend; keep-views preserved them).
    write("prefix.embs", { n_prefix, cfg.hidden }, getf(t_prefix_emb_));
    for (int64_t l = 0; l < kNDeepstack; ++l) {
        write("prefix.deepstack" + std::to_string(l), { n_prefix, cfg.hidden }, getf(t_ds_[l]));
    }
    {
        std::vector<int32_t> pp(ggml_nbytes(t_prefix_pos_) / sizeof(int32_t));
        ggml_backend_tensor_get(t_prefix_pos_, pp.data(), 0, pp.size() * sizeof(int32_t));
        std::vector<float> pos3(3 * static_cast<size_t>(n_prefix));
        for (size_t i = 0; i < pos3.size(); ++i) {
            pos3[i] = static_cast<float>(pp[i]);
        }
        write("prefix.pos", { 3, 1, n_prefix }, pos3);
    }
    {
        std::vector<int32_t> sp(ggml_nbytes(t_suffix_pos_) / sizeof(int32_t));
        ggml_backend_tensor_get(t_suffix_pos_, sp.data(), 0, sp.size() * sizeof(int32_t));
        std::vector<float> pos3(3 * static_cast<size_t>(n_suf));
        for (size_t i = 0; i < pos3.size(); ++i) {
            pos3[i] = static_cast<float>(sp[i]);
        }
        write("s0.pos", { 3, 1, n_suf }, pos3);
    }
    {
        // Additive masks -> attendance indicator (ref dumps bool, True=attend).
        auto cvt = [&](ggml_tensor * t, int64_t n_kv, int64_t n_q, const std::string & name) {
            const std::vector<float> m = getf(t);
            std::vector<float>       a(m.size());
            for (size_t i = 0; i < m.size(); ++i) {
                a[i] = (m[i] > -1e30f) ? 1.f : 0.f;
            }
            write(name, { 1, n_q, n_kv }, a);
        };
        cvt(t_prefix_mask_, n_prefix, n_prefix, "prefix.att2d");
        cvt(t_suffix_mask_, n_prefix + n_suf, n_suf, "s0.att2d");
    }
    for (size_t s = 0; s < t_time_.size(); ++s) {
        write("s" + std::to_string(s) + ".time_emb", { 1, cfg.expert_h }, getf(t_time_[s]));
    }
    if (!t_time_.empty()) {
        write("s0.ada_cond", { 1, cfg.expert_h }, getf(t_time_[0]));
    }
    write("state.canon", { cfg.max_state_dim }, getf(t_state_));
    write("noise", { cfg.n_suffix, cfg.max_action_dim }, getf(t_x0_));
    write("final.x", { cfg.n_suffix, cfg.max_action_dim }, getf(x_final_));
    write("query.current", { kNumTaskTokens, cfg.hidden }, getf(q_cur));
    write("query.future", { kNumTaskTokens, cfg.hidden }, getf(q_fut));

    // 3) Vision / language host staging from prepare().
    {
        const int64_t mm_dim = cfg.hidden * (1 + kNDeepstack);
        if (!p.img_emb_host.empty() && p.img_emb_host.size() % static_cast<size_t>(mm_dim) == 0) {
            const int64_t  n_tok = static_cast<int64_t>(p.img_emb_host.size()) / mm_dim;
            const float *  src   = p.img_emb_host.data();
            std::vector<float> clip(static_cast<size_t>(n_tok) * cfg.hidden);
            std::vector<float> ds(static_cast<size_t>(kNDeepstack) * n_tok * cfg.hidden);
            for (int64_t i = 0; i < n_tok; ++i) {
                std::memcpy(clip.data() + static_cast<size_t>(i) * cfg.hidden,
                            src + static_cast<size_t>(i) * mm_dim, static_cast<size_t>(cfg.hidden) * sizeof(float));
                for (int64_t l = 0; l < kNDeepstack; ++l) {
                    std::memcpy(ds.data() + static_cast<size_t>(l * n_tok + i) * cfg.hidden,
                                src + static_cast<size_t>(i) * mm_dim + static_cast<size_t>(1 + l) * cfg.hidden,
                                static_cast<size_t>(cfg.hidden) * sizeof(float));
                }
            }
            write("img.clip_out", { n_tok, cfg.hidden }, clip);
            for (int64_t l = 0; l < kNDeepstack; ++l) {
                write("img.deepstack" + std::to_string(l), { n_tok, cfg.hidden },
                      std::vector<float>(ds.begin() + static_cast<size_t>(l * n_tok) * cfg.hidden,
                                         ds.begin() + static_cast<size_t>((l + 1) * n_tok) * cfg.hidden));
            }
        }
        if (!p.lang_rows.empty()) {
            write("lang.embs", { p.n_lang, cfg.hidden }, p.lang_rows);
        }
    }

    // 4) prefix.out_vlm: final RMSNorm of the prefix hidden, computed host-side.
    if (vlm_output_norm) {
        const DebugTap * pre = nullptr;
        for (const auto & d : dbg_taps_) {
            if (d.name == "prefix.pre_final_out") {
                pre = &d;
                break;
            }
        }
        if (pre) {
            const std::vector<float> h = getf(pre->t);  // [P, hidden] row-major
            const std::vector<float> w = getf(vlm_output_norm);
            std::vector<float>       out(h.size());
            for (int64_t r = 0; r < n_prefix; ++r) {
                const float * x = h.data() + static_cast<size_t>(r) * cfg.hidden;
                double        ss = 0.0;
                for (int64_t c = 0; c < cfg.hidden; ++c) {
                    ss += static_cast<double>(x[c]) * x[c];
                }
                const float inv = 1.f / std::sqrt(static_cast<float>(ss / cfg.hidden) + cfg.rms_eps);
                for (int64_t c = 0; c < cfg.hidden; ++c) {
                    out[static_cast<size_t>(r) * cfg.hidden + c] = x[c] * inv * w[c];
                }
            }
            write("prefix.out_vlm", { n_prefix, cfg.hidden }, out);
        }
    }

    std::printf("lingbot_vla_v2: dumped %d tensors to %s\n", n_written, dump_dir_.c_str());
}

// ────────────────────────────────────────────────────────────────────────────
// predict: prepare + compute
// ────────────────────────────────────────────────────────────────────────────

std::vector<float> LingBotVlaV2Model::predict(const Inputs & in) {
    using clk     = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();

    // Proprioception is consumed in compute(); stash it from the Inputs.
    last_state_.clear();
    if (in.state) {
        last_state_.assign(in.state, in.state + cfg.real_state_dim);
    }

    PreparedInput p = prepare_inputs(in);
    if (!p.ok) {
        return {};
    }
    std::vector<float> out = compute_actions(p);
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    }
    return out;
}

}  // namespace vla
