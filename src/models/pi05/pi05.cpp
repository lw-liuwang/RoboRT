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
 * @file pi05.cpp
 * @brief pi0.5 architecture implementation: model struct, weight loading,
 *        two-phase inference (prepare_inputs + compute_actions), and factory.
 *
 * The implementation is split across five translation units for readability:
 *   pi05_gguf.h/.cpp     — GGUF reader (metadata + tensor I/O)
 *   pi05_weights.h       — weight tensor structs (GemmaLayerW, AdaGemmaLayerW)
 *   pi05_graph.h/.cpp    — GGML compute-graph builders
 *   pi05_vispruner.h/.cpp — visual token pruning
 *   pi05_utils.h         — sinusoidal time embedding
 *   pi05.cpp (this file) — Pi05Model lifecycle + inference pipeline
 */

#include "clip.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model.h"
#include "pi05_gguf.h"
#include "pi05_graph.h"
#include "pi05_utils.h"
#include "pi05_vispruner.h"
#include "pi05_weights.h"
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
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace vla {

// ────────────────────────────────────────────────────────────────────────────
// Pi05Model: concrete vla::Model implementation for pi0.5.
//
// Lifecycle:
//   pi05_create()     — allocate, load config, load weights, build backend.
//   prepare_inputs()  — vision encoding (CLIP) + language embedding lookup.
//   compute_actions() — GGML graph upload + graph compute + de-normalise.
//   predict()         — prepare_inputs() + compute_actions() in one call.
// ────────────────────────────────────────────────────────────────────────────

struct Pi05Model final : public Model {
    ~Pi05Model() override;

    const Config & config() const override { return cfg; }

    const Stats & last_stats() const override { return stats; }

    std::vector<float> predict(const Inputs & in) override;

    PreparedInput prepare(const Inputs & in) override { return prepare_inputs(in); }

    std::vector<float> compute(const PreparedInput & p) override { return compute_actions(p); }

    PreparedInput      prepare_inputs(const Inputs & in);
    std::vector<float> compute_actions(const PreparedInput & p);

    Stats      stats{};
    Config     cfg{};
    std::mutex stats_mu_;  ///< Protects stats (async server has two threads).

    clip_ctx *            cctx        = nullptr;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    bool                  flash_attn_ = false;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    std::string           ckpt_path_;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    std::vector<pi05::GemmaLayerW>    pl_layers;  ///< Prefix (VLM) Gemma layers.
    std::vector<pi05::AdaGemmaLayerW> ex_layers;  ///< Action-expert AdaGemma layers.
    ggml_tensor *                     ex_final_norm_W = nullptr;
    ggml_tensor *                     ex_final_norm_b = nullptr;

    ggml_tensor *W_ain = nullptr, *b_ain = nullptr;    ///< Action input projection.
    ggml_tensor *W_tm1 = nullptr, *b_tm1 = nullptr;    ///< Time MLP layer 1.
    ggml_tensor *W_tm2 = nullptr, *b_tm2 = nullptr;    ///< Time MLP layer 2.
    ggml_tensor *W_aout = nullptr, *b_aout = nullptr;  ///< Action output projection.

    // ── Cached compute graph ─────────────────────────────────────────────
    // Rebuilt only when input token counts change.  Single-threaded REP mode
    // means one cached graph is always safe; async mode locks externally.
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

    // ── RTC (Real-Time Chunking) graph inputs ─────────────────────────────
    ggml_tensor * t_prev_chunk_  = nullptr;
    ggml_tensor * t_rtc_weights_ = nullptr;
    int64_t       c_rtc_         = 0;
    int64_t       c_rtc_horizon_ = 0;

    // ── RTC configuration (read from env once; default off = bit-exact) ───
    bool               rtc_initialized_  = false;
    bool               rtc_enabled_      = false;
    int                rtc_horizon_      = 10;
    int                rtc_delay_        = 0;
    float              rtc_max_guidance_ = 10.f;
    std::string        rtc_schedule_     = "linear";
    bool               rtc_x0_inpaint_   = false;
    std::vector<float> rtc_step_guidance_;

    void init_rtc_config();

    // ── Normalization statistics (loaded from GGUF) ───────────────────────
    std::string        state_norm_mode  = "QUANTILES";
    std::string        action_norm_mode = "QUANTILES";
    std::vector<float> state_mean, state_std, action_mean, action_std;
    std::vector<float> state_q01, state_q99, action_q01, action_q99;

    // ── Noise RNG ─────────────────────────────────────────────────────────
    // Entropy-seeded by default; VLA_PI05_SEED=N makes noise reproducible.
    std::mt19937 rng{ []() -> std::mt19937::result_type {
        const char * e = std::getenv("VLA_PI05_SEED");
        return e ? static_cast<std::mt19937::result_type>(std::strtoul(e, nullptr, 10)) : std::random_device{}();
    }() };

    int n_threads = 4;
};

// ────────────────────────────────────────────────────────────────────────────
// Anonymous namespace: internal helpers (config / stats loading).
// ────────────────────────────────────────────────────────────────────────────

namespace {

bool ends_with(const std::string & s, const char * sfx) {
    const size_t n = std::strlen(sfx);
    return s.size() >= n && s.compare(s.size() - n, n, sfx) == 0;
}

bool load_config(const pi05::GgufReader & g, Config & cfg) {
    auto need = [&](const char * k) {
        if (!g.has_key(k)) {
            std::fprintf(stderr, "pi05: gguf missing key %s\n", k);
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

    if (const char * ns = std::getenv("VLA_PI05_NUM_STEPS");
        ns != nullptr && std::atoi(ns) >= 1 && std::atoi(ns) <= 10) {
        std::printf("pi05: num_steps overridden by env: %d -> %d\n", cfg.num_steps, std::atoi(ns));
        cfg.num_steps = static_cast<uint32_t>(std::atoi(ns));
    }
    if (const char * cs = std::getenv("VLA_PI05_CHUNK"); cs != nullptr && std::atoi(cs) >= 1 && std::atoi(cs) <= 50) {
        std::printf("pi05: chunk_size overridden by env: %d -> %d\n", static_cast<int>(cfg.n_suffix), std::atoi(cs));
        cfg.n_suffix = static_cast<uint32_t>(std::atoi(cs));
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
    cfg.rope_n_dims       = static_cast<int>(cfg.head_dim);
    cfg.rope_freq_base    = g.has_key("pi05.rope_theta") ? static_cast<float>(g.f64("pi05.rope_theta")) : 10000.f;
    cfg.n_prefix          = 0;
    cfg.n_full            = 0;
    return true;
}

bool load_norm_stats(pi05::GgufReader & g, Pi05Model & m) {
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
            std::printf("pi05: %s missing - using identity\n", name);
            return;
        }
        if (t->ne[0] != static_cast<int64_t>(dst.size())) {
            std::printf("pi05: %s dim mismatch - using identity\n", name);
            return;
        }
        if (!g.read_raw(name, dst.data())) {
            std::printf("pi05: %s read failed - using identity\n", name);
        }
    };

    if (m.state_norm_mode == "MEAN_STD") {
        read1d("state_mean", m.state_mean);
        read1d("state_std", m.state_std);
    } else if (m.state_norm_mode == "QUANTILES") {
        read1d("state_q01", m.state_q01);
        read1d("state_q99", m.state_q99);
    } else {
        std::fprintf(stderr, "pi05: unsupported state_norm_mode '%s'\n", m.state_norm_mode.c_str());
        return false;
    }
    if (m.action_norm_mode == "MEAN_STD") {
        read1d("action_mean", m.action_mean);
        read1d("action_std", m.action_std);
    } else if (m.action_norm_mode == "QUANTILES") {
        read1d("action_q01", m.action_q01);
        read1d("action_q99", m.action_q99);
    } else {
        std::fprintf(stderr, "pi05: unsupported action_norm_mode '%s'\n", m.action_norm_mode.c_str());
        return false;
    }
    std::printf("pi05: normalization: state=%s  action=%s\n", m.state_norm_mode.c_str(), m.action_norm_mode.c_str());
    return true;
}

}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Pi05Model members
// ────────────────────────────────────────────────────────────────────────────

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

void Pi05Model::init_rtc_config() {
    if (rtc_initialized_) {
        return;
    }
    rtc_initialized_ = true;

    if (const char * e = std::getenv("VLA_PI05_RTC"); e && std::atoi(e) > 0) {
        rtc_enabled_ = true;
    }
    if (const char * e = std::getenv("VLA_PI05_RTC_HORIZON"); e && std::atoi(e) >= 1) {
        rtc_horizon_ = std::atoi(e);
    }
    if (const char * e = std::getenv("VLA_PI05_RTC_DELAY"); e && std::atoi(e) >= 0) {
        rtc_delay_ = std::atoi(e);
    }
    if (const char * e = std::getenv("VLA_PI05_RTC_GUIDANCE"); e && std::atof(e) > 0.0) {
        rtc_max_guidance_ = static_cast<float>(std::atof(e));
    }
    if (const char * e = std::getenv("VLA_PI05_RTC_SCHEDULE"); e && e[0]) {
        rtc_schedule_ = e;
    }
    if (const char * e = std::getenv("VLA_PI05_RTC_X0_INPAINT"); e && std::atoi(e) > 0) {
        rtc_x0_inpaint_ = true;
    }

    if (rtc_enabled_) {
        // Per-step guidance weight (mirrors LeRobot modeling_rtc.denoise_step):
        //   tau = s/N;  time = 1 - tau;  c = time/tau;  inv_r2 = (time² + tau²) / time²
        //   guidance = min(c * inv_r2, max_g)
        const int   num_steps = cfg.num_steps;
        const float max_g     = rtc_max_guidance_;
        rtc_step_guidance_.assign(num_steps, max_g);
        for (int s = 0; s < num_steps; ++s) {
            const float time = 1.f - static_cast<float>(s) / static_cast<float>(num_steps);
            const float tau  = static_cast<float>(s) / static_cast<float>(num_steps);
            if (tau > 0.f) {
                const float c      = time / tau;
                const float inv_r2 = (time * time + tau * tau) / (time * time);
                float       g      = c * inv_r2;
                if (!(g > 0.f) || g > max_g) {
                    g = max_g;
                }
                rtc_step_guidance_[s] = g;
            }
        }
        std::printf("pi05: RTC enabled: horizon=%d delay=%d guidance=%.1f schedule=%s inpaint=%d steps=%d\n",
                    rtc_horizon_, rtc_delay_, rtc_max_guidance_, rtc_schedule_.c_str(), rtc_x0_inpaint_ ? 1 : 0,
                    num_steps);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Factory
// ────────────────────────────────────────────────────────────────────────────

std::unique_ptr<Model> pi05_create(const std::string & mmproj_path,
                                   const std::string & ckpt_path,
                                   const std::string & config_path) {
    (void) config_path;

    if (!ends_with(ckpt_path, ".gguf")) {
        std::fprintf(stderr,
                     "pi05: ckpt must be a GGUF produced by convert_pi05_to_gguf.py "
                     "(got '%s')\n",
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
        std::printf("pi05: flash attention enabled\n");
    }

    pi05::GgufReader g;
    if (!g.open(ckpt_path)) {
        return nullptr;
    }
    if (!g.has_key("pi05.architecture") || g.str("pi05.architecture") != "pi05") {
        std::fprintf(stderr, "pi05: '%s' is not a pi0.5 GGUF (pi05.architecture missing/wrong)\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, m->cfg)) {
        return nullptr;
    }
    const Config & cfg     = m->cfg;
    const char *   mm_name = [&]() -> const char * {
        if (m->matmul_type == GGML_TYPE_F32) {
            return "F32";
        }
        if (m->matmul_type == GGML_TYPE_Q8_0) {
            return "Q8_0";
        }
        if (m->matmul_type == GGML_TYPE_Q4_0) {
            return "Q4_0";
        }
        return "BF16";
    }();
    std::printf(
        "pi05: hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_layers=%lld "
        "expert_h=%lld expert_inter=%lld chunk=%lld steps=%d "
        "real_state=%lld real_action=%lld matmul=%s\n",
        static_cast<long long>(cfg.hidden), static_cast<long long>(cfg.intermediate),
        static_cast<long long>(cfg.n_q_heads), static_cast<long long>(cfg.n_kv_heads),
        static_cast<long long>(cfg.head_dim), static_cast<long long>(cfg.n_layers),
        static_cast<long long>(cfg.expert_h), static_cast<long long>(cfg.expert_inter),
        static_cast<long long>(cfg.n_suffix), cfg.num_steps, static_cast<long long>(cfg.real_state_dim),
        static_cast<long long>(cfg.real_action_dim), mm_name);

    // ── Backend selection ─────────────────────────────────────────────────
#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) {
        m->is_cuda = true;
        std::printf("pi05: backend = CUDA (device 0)\n");
    } else {
        std::fprintf(stderr, "pi05: ggml_backend_cuda_init failed; falling back to CPU\n");
    }
#endif
    {
        const unsigned hw = std::thread::hardware_concurrency();
        m->n_threads      = (hw == 0) ? 4 : static_cast<int>(std::min(hw, 8u));
    }
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) {
            std::fprintf(stderr, "pi05: ggml_backend_cpu_init failed\n");
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("pi05: backend = CPU (%d threads)\n", m->n_threads);
    }

    // ── Vision tower (CLIP) ───────────────────────────────────────────────
    {
        clip_context_params cp = {};
        cp.use_gpu             = m->is_cuda;
        cp.flash_attn_type     = m->is_cuda ? CLIP_FLASH_ATTN_TYPE_AUTO : CLIP_FLASH_ATTN_TYPE_DISABLED;
        if (std::getenv("VLA_PI05_VIS_PRUNER_TOKENS")) {
            // VisPruner needs raw attention weights; flash attention hides them.
            cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
            std::printf("pi05: VisPruner enabled, CLIP flash attention forced OFF\n");
        }
        cp.image_min_tokens  = -1;
        cp.image_max_tokens  = -1;
        cp.warmup            = m->is_cuda;
        cp.cb_eval           = nullptr;
        cp.cb_eval_user_data = nullptr;
        clip_init_result r   = clip_init(mmproj_path.c_str(), cp);
        if (!r.ctx_v) {
            std::fprintf(stderr, "pi05: clip_init failed for %s\n", mmproj_path.c_str());
            return nullptr;
        }
        m->cctx           = r.ctx_v;
        const int img_sz  = clip_get_image_size(m->cctx);
        const int mm_embd = clip_n_mmproj_embd(m->cctx);
        if (img_sz != 224 || mm_embd != static_cast<int>(cfg.hidden)) {
            std::fprintf(stderr, "pi05: mmproj mismatch (image_size=%d mmproj_embd=%d; want 224/%lld)\n", img_sz,
                         mm_embd, static_cast<long long>(cfg.hidden));
            return nullptr;
        }
    }

    // ── Weight context ────────────────────────────────────────────────────
    {
        ggml_init_params wp = { static_cast<size_t>(16) * 1024 * 1024, nullptr, true };
        m->ctx_weights      = ggml_init(wp);
        if (!m->ctx_weights) {
            std::fprintf(stderr, "pi05: ggml_init(ctx_weights) failed\n");
            return nullptr;
        }
    }
    ggml_context *             W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;

    // mk: register a tensor (metadata only) in the weight context.
    auto mk = [&](const char * name, ggml_type type, int n_dims, const int64_t * ne) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "pi05: missing tensor %s\n", name);
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
            std::fprintf(stderr, "pi05: missing tensor %s\n", name);
            return nullptr;
        }
        return mk(name, m->matmul_type, GGML_MAX_DIMS, gt->ne);
    };
    auto mk_f32 = [&](const char * name) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "pi05: missing tensor %s\n", name);
            return nullptr;
        }
        return mk(name, GGML_TYPE_F32, GGML_MAX_DIMS, gt->ne);
    };

    auto load_layer = [&](const char * tower, int i, pi05::GemmaLayerW & lw) -> bool {
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

    auto load_adarms_layer = [&](const char * tower, int i, pi05::AdaGemmaLayerW & lw) -> bool {
        char b[256];
        auto suf = [&](const char * s) {
            std::snprintf(b, sizeof(b), "%s.blk.%d.%s", tower, i, s);
            return b;
        };
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
        if (!load_layer("vlm", static_cast<int>(i), m->pl_layers[i])) {
            return nullptr;
        }
        if (!load_adarms_layer("aex", static_cast<int>(i), m->ex_layers[i])) {
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
            std::fprintf(stderr, "pi05: weight tensor creation failed\n");
            return nullptr;
        }
    }
    if (!m->ex_final_norm_W || !m->ex_final_norm_b || !m->W_ain || !m->b_ain || !m->W_tm1 || !m->b_tm1 || !m->W_tm2 ||
        !m->b_tm2 || !m->W_aout || !m->b_aout) {
        std::fprintf(stderr, "pi05: failed to wire projection/norm tensors\n");
        return nullptr;
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) {
        std::fprintf(stderr, "pi05: ggml_backend_alloc_ctx_tensors failed (OOM?)\n");
        return nullptr;
    }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type, pi05::is_gemma_norm(t->name));
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "pi05: upload size mismatch for %s (%zu vs %zu)\n", t->name, bytes.size(),
                         ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("pi05: resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    if (!load_norm_stats(g, *m)) {
        return nullptr;
    }
    std::printf("pi05: model loaded (n_threads=%d)\n", m->n_threads);
    return m;
}

// ────────────────────────────────────────────────────────────────────────────
// prepare_inputs: vision encoding + language embedding lookup
// ────────────────────────────────────────────────────────────────────────────

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

    // ── Vision tower ──────────────────────────────────────────────────────
    if (in.precomputed_img_emb) {
        p.n_img_tokens = static_cast<int64_t>(in.n_img_views) * cfg.n_img;
        p.img_emb_host.assign(in.precomputed_img_emb,
                              in.precomputed_img_emb + static_cast<size_t>(p.n_img_tokens) * hidden_pl);
    } else {
        if (in.n_images < 1 || !in.images) {
            std::fprintf(stderr, "pi05: prepare: no images and no precomputed_img_emb\n");
            p.ok    = false;
            p.error = "no images and no precomputed_img_emb";
            return p;
        }
        const int     img_sz          = clip_get_image_size(cctx);
        const size_t  per_pix         = static_cast<size_t>(3) * img_sz * img_sz;
        const size_t  per_out         = clip_embd_nbytes_by_img(cctx, img_sz, img_sz) / sizeof(float);
        const int64_t per_view_tokens = static_cast<int64_t>(per_out / static_cast<size_t>(hidden_pl));

        // VisPruner config: read once per process (static).
        static int   vp_tokens     = -1;
        static float vp_ratio      = 0.5f;
        static bool  vp_keep_order = false;
        if (vp_tokens < 0) {
            const char * et = std::getenv("VLA_PI05_VIS_PRUNER_TOKENS");
            const char * er = std::getenv("VLA_PI05_VIS_PRUNER_RATIO");
            const char * ek = std::getenv("VLA_PI05_VIS_PRUNER_KEEP_ORDER");
            vp_tokens       = et ? std::atoi(et) : 0;
            vp_ratio        = er ? static_cast<float>(std::atof(er)) : 0.5f;
            vp_keep_order   = ek ? (std::atoi(ek) > 0) : false;
        }
        const int vp_per_view = (vp_tokens > 0) ? std::max(1, vp_tokens / std::max(1, in.n_images)) : 0;

        std::vector<float> hwc(per_pix);
        const auto         tv0 = clk::now();
        p.img_emb_host.clear();
        p.img_emb_host.reserve(per_out * static_cast<size_t>(in.n_images));

        for (int v = 0; v < in.n_images; ++v) {
            const ImageView & view = in.images[v];
            if (view.w != img_sz || view.h != img_sz) {
                std::fprintf(stderr, "pi05: image[%d] is %dx%d; requires %dx%d\n", v, view.w, view.h, img_sz, img_sz);
                p.ok    = false;
                p.error = "bad image size";
                return p;
            }
            // Normalise to [-1, 1] (pi0.5 convention).
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
                std::fprintf(stderr, "pi05: clip_encode_float_image failed (view %d)\n", v);
                p.ok    = false;
                p.error = "clip_encode_float_image failed";
                return p;
            }

            // Optional debug dump of raw embeddings.
            if (const char * dump = std::getenv("VLA_PI05_DUMP_IMG"); dump && std::atoi(dump) > 0) {
                char path[256];
                std::snprintf(path, sizeof(path), "/tmp/pi05_img_emb_view%d.f32", v);
                if (FILE * fp = std::fopen(path, "wb")) {
                    std::fwrite(view_emb.data(), sizeof(float), per_out, fp);
                    std::fclose(fp);
                }
                double s  = 0.;
                double sa = 0.;
                double s2 = 0.;
                for (size_t i = 0; i < per_out; ++i) {
                    s += view_emb[i];
                    sa += std::fabs(view_emb[i]);
                    s2 += static_cast<double>(view_emb[i]) * view_emb[i];
                }
                std::printf(
                    "[pi05-debug] view %d: n=%zu mean=%.6f meanabs=%.6f rms=%.6f "
                    "first=[%.4f %.4f %.4f %.4f]\n",
                    v, per_out, s / static_cast<double>(per_out), sa / static_cast<double>(per_out),
                    std::sqrt(s2 / static_cast<double>(per_out)), view_emb[0], view_emb[1], view_emb[2], view_emb[3]);
            }

            int64_t v_tokens = per_view_tokens;
            if (vp_per_view > 0 && v_tokens > vp_per_view) {
                int           n_patches = 0;
                int           n_heads   = 0;
                const float * attn      = clip_get_last_attn_data(cctx, &n_patches, &n_heads);
                if (attn && n_patches == static_cast<int>(v_tokens)) {
                    pi05::vispruner_prune(view_emb, v_tokens, hidden_pl, attn, n_patches, n_heads, vp_per_view,
                                          vp_ratio, vp_keep_order);
                } else {
                    std::fprintf(stderr, "pi05: VisPruner skipped view %d (attn=%p patches=%d tokens=%lld)\n", v,
                                 static_cast<const void *>(attn), n_patches, static_cast<long long>(v_tokens));
                }
            }

            p.img_emb_host.insert(p.img_emb_host.end(), view_emb.begin(), view_emb.end());
            n_img_tokens += v_tokens;
        }
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - tv0).count();
        }
        p.n_img_tokens = n_img_tokens;
    }

    // ── Language embedding lookup ─────────────────────────────────────────
    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "pi05: prepare: empty lang_tokens\n");
        p.ok    = false;
        p.error = "empty lang_tokens";
        return p;
    }
    const int64_t        n_lang = in.n_lang;
    std::vector<int32_t> lang_ids(in.lang_tokens, in.lang_tokens + n_lang);
    p.lang_rows.resize(static_cast<size_t>(n_lang) * hidden_pl);
    {
        pi05::GgufReader g2;
        if (!g2.open(ckpt_path_)) {
            p.ok    = false;
            p.error = "gguf_reader open failed";
            return p;
        }
        if (!g2.fetch_rows_f32("token_embd.weight", lang_ids, p.lang_rows.data(), hidden_pl)) {
            p.ok    = false;
            p.error = "token_embd.weight lookup failed";
            return p;
        }
    }
    p.n_lang = n_lang;

    if (in.noise) {
        p.noise.assign(in.noise, in.noise + static_cast<size_t>(cfg.max_action_dim) * chunk);
    }
    if (in.prev_chunk && in.n_prev_chunk > 0) {
        p.prev_chunk.assign(in.prev_chunk, in.prev_chunk + static_cast<size_t>(in.n_prev_chunk) * cfg.real_action_dim);
        p.n_prev_chunk = in.n_prev_chunk;
    }
    if (in.attention_mask && in.attention_mask_n > 0) {
        p.attention_mask.assign(in.attention_mask, in.attention_mask + in.attention_mask_n);
    }
    p.timing_detail = in.timing_detail;
    p.ok            = true;
    return p;
}

// ────────────────────────────────────────────────────────────────────────────
// compute_actions: graph build (cached) + upload + compute + de-normalise
// ────────────────────────────────────────────────────────────────────────────

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
    const float    dt        = -1.f / static_cast<float>(num_steps);
    const float    rope_base = cfg.rope_freq_base;

    const int64_t n_img_tokens = p.n_img_tokens;
    const int64_t n_lang       = p.n_lang;
    const int64_t n_prefix     = n_img_tokens + n_lang;
    const int64_t n_total      = n_prefix + n_suf;

    init_rtc_config();

    // ── Cached compute graph ──────────────────────────────────────────────
    static const bool no_cache = []() {
        const char * e = std::getenv("VLA_PI05_NO_CACHE");
        return e && std::atoi(e) > 0;
    }();
    const bool need_rebuild = no_cache || !gf_ || c_n_img_tokens_ != n_img_tokens || c_n_lang_ != n_lang ||
                              c_n_suf_ != n_suf || c_rtc_ != static_cast<int64_t>(rtc_enabled_) ||
                              c_rtc_horizon_ != static_cast<int64_t>(rtc_horizon_);

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
        t_image_emb_ = t_lang_emb_ = t_prefix_pos_ = t_x0_ = nullptr;
        t_suffix_pos_ = t_full_mask_ = nullptr;
        t_prev_chunk_ = t_rtc_weights_ = nullptr;
        t_time_.clear();

        ggml_init_params cp = { static_cast<size_t>(64) * 1024 * 1024, nullptr, true };
        comp_ctx_           = ggml_init(cp);
        if (!comp_ctx_) {
            std::fprintf(stderr, "pi05: ggml_init(ctx_compute) failed\n");
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
            t_prev_chunk_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);
            ggml_set_input(t_prev_chunk_);
            t_rtc_weights_ = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);
            ggml_set_input(t_rtc_weights_);
        }
        t_time_.resize(num_steps);
        for (int s = 0; s < num_steps; ++s) {
            t_time_[s] = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_ex, 1);
            ggml_set_input(t_time_[s]);
        }

        // Prefix (VLM) tower: build and cache KV for all layers.
        const float   lang_scale  = static_cast<float>(std::sqrt(static_cast<double>(hidden_pl)));
        ggml_tensor * prefix_embs = ggml_concat(C, t_image_emb_, ggml_scale(C, t_lang_emb_, lang_scale), 1);

        std::vector<ggml_tensor *> cK(n_layers);
        std::vector<ggml_tensor *> cV(n_layers);
        {
            ggml_tensor * h = prefix_embs;
            for (int64_t i = 0; i < n_layers; ++i) {
                h = pi05::build_gemma_layer(C, pl_layers[i], h, t_prefix_pos_, cfg, n_prefix, rope_base, nullptr,
                                            nullptr, nullptr, flash_attn_, &cK[i], &cV[i]);
            }
            (void) h;
        }

        // Action expert: denoising loop over num_steps flow-matching steps.
        ggml_tensor * x_t = t_x0_;
        for (int step = 0; step < num_steps; ++step) {
            ggml_tensor * adarms_cond = nullptr;
            ggml_tensor * h =
                pi05::build_embed_suffix(C, x_t, t_time_[step], W_ain, b_ain, W_tm1, b_tm1, W_tm2, b_tm2, &adarms_cond);
            for (int64_t i = 0; i < n_layers; ++i) {
                h = pi05::build_adarms_gemma_layer(C, ex_layers[i], h, t_suffix_pos_, adarms_cond, cfg, n_suf,
                                                   rope_base, cK[i], cV[i], t_full_mask_, flash_attn_);
            }
            // Final expert norm (AdaRMS; gate output ignored per OpenPI).
            ggml_tensor * h_final =
                pi05::adarms_norm(C, h, ex_final_norm_W, ex_final_norm_b, adarms_cond, cfg, nullptr);
            ggml_tensor * v_t = ggml_add(C, ggml_mul_mat(C, W_aout, h_final), b_aout);

            if (rtc_enabled_) {
                // RTC prefix guidance (mirrors LeRobot modeling_rtc.denoise_step):
                //   x1_est = x_t - time * v_t
                //   err    = (prev_chunk - x1_est) * weights
                //   v_t   -= step_guidance * err
                const float   time   = 1.f + static_cast<float>(step) * dt;
                ggml_tensor * x1_est = ggml_sub(C, x_t, ggml_scale(C, v_t, time));
                ggml_tensor * diff   = ggml_sub(C, t_prev_chunk_, x1_est);
                ggml_tensor * err    = ggml_mul(C, diff, t_rtc_weights_);
                v_t                  = ggml_sub(C, v_t, ggml_scale(C, err, rtc_step_guidance_[step]));
            }
            x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
        }
        x_final_ = x_t;
        ggml_set_output(x_final_);

        // AdaRMS adds scale/shift/gate nodes per layer, so pi0.5 needs a
        // larger graph arena than the plain pi0 expert.
        gf_ = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf_, x_final_);

        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf_)) {
            std::fprintf(stderr, "pi05: ggml_gallocr_alloc_graph failed (OOM?)\n");
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

    // ── Upload inputs (every request) ────────────────────────────────────
    ggml_backend_tensor_set(t_image_emb_, p.img_emb_host.data(), 0, ggml_nbytes(t_image_emb_));
    ggml_backend_tensor_set(t_lang_emb_, p.lang_rows.data(), 0, ggml_nbytes(t_lang_emb_));

    {
        std::vector<int32_t> pp(n_prefix);
        std::vector<int32_t> sp(n_suf);
        for (int64_t i = 0; i < n_prefix; ++i) {
            pp[i] = static_cast<int32_t>(i);
        }
        for (int64_t i = 0; i < n_suf; ++i) {
            sp[i] = static_cast<int32_t>(n_prefix + i);
        }
        ggml_backend_tensor_set(t_prefix_pos_, pp.data(), 0, ggml_nbytes(t_prefix_pos_));
        ggml_backend_tensor_set(t_suffix_pos_, sp.data(), 0, ggml_nbytes(t_suffix_pos_));
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
        // RTC x0 inpainting: seed the executed prefix into the initial noise.
        if (rtc_x0_inpaint_ && p.n_prev_chunk > 0) {
            const int64_t L = std::min(static_cast<int64_t>(rtc_horizon_), p.n_prev_chunk);
            for (int64_t t = 0; t < L; ++t) {
                for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
                    const float xr = p.prev_chunk[static_cast<size_t>(t) * cfg.real_action_dim + j];
                    if (action_norm_mode == "MEAN_STD") {
                        x0h[static_cast<size_t>(t) * max_ad + j] =
                            (xr - action_mean[j]) / (action_std[j] + cfg.norm_eps);
                    } else {
                        const float denom                        = action_q99[j] - action_q01[j];
                        x0h[static_cast<size_t>(t) * max_ad + j] = 2.f * (xr - action_q01[j]) / denom - 1.f;
                    }
                }
            }
        }
        ggml_backend_tensor_set(t_x0_, x0h.data(), 0, ggml_nbytes(t_x0_));
    }

    // RTC prev_chunk + weights.
    if (rtc_enabled_) {
        std::vector<float> prevh(static_cast<size_t>(max_ad) * chunk, 0.f);
        if (p.n_prev_chunk > 0) {
            const int64_t nt = std::min(p.n_prev_chunk, chunk);
            for (int64_t t = 0; t < nt; ++t) {
                for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
                    const float xr = p.prev_chunk[static_cast<size_t>(t) * cfg.real_action_dim + j];
                    if (action_norm_mode == "MEAN_STD") {
                        prevh[static_cast<size_t>(t) * max_ad + j] =
                            (xr - action_mean[j]) / (action_std[j] + cfg.norm_eps);
                    } else {
                        const float denom                          = action_q99[j] - action_q01[j];
                        prevh[static_cast<size_t>(t) * max_ad + j] = 2.f * (xr - action_q01[j]) / denom - 1.f;
                    }
                }
            }
        }
        ggml_backend_tensor_set(t_prev_chunk_, prevh.data(), 0, ggml_nbytes(t_prev_chunk_));

        std::vector<float> wgh(static_cast<size_t>(max_ad) * chunk, 0.f);
        if (p.n_prev_chunk > 0) {
            const int          L = static_cast<int>(std::min<int64_t>(rtc_horizon_, p.n_prev_chunk));
            const int          D = std::min(rtc_delay_, L);
            std::vector<float> sched(static_cast<size_t>(chunk), 0.f);
            for (int i = 0; i < D; ++i) {
                sched[i] = 1.f;
            }
            if (L > D) {
                const float denom = static_cast<float>(L - D) + 1.f;
                const float em1   = std::exp(1.f) - 1.f;
                for (int i = D; i < L; ++i) {
                    float w = (static_cast<float>(L) - static_cast<float>(i)) / denom;
                    if (rtc_schedule_ == "exp") {
                        w = w * std::expm1(w) / em1;
                    }
                    sched[i] = w;
                }
            }
            for (int64_t t = 0; t < chunk; ++t) {
                for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
                    wgh[static_cast<size_t>(t) * max_ad + j] = sched[t];
                }
            }
        }
        ggml_backend_tensor_set(t_rtc_weights_, wgh.data(), 0, ggml_nbytes(t_rtc_weights_));
    }

    if (!flash_attn_) {
        // Flash attention uses a null mask; only upload the all-zeros mask in non-flash mode.
        std::vector<float> mk(static_cast<size_t>(n_total) * n_suf, 0.f);
        ggml_backend_tensor_set(t_full_mask_, mk.data(), 0, ggml_nbytes(t_full_mask_));
    }

    for (int s = 0; s < num_steps; ++s) {
        const float              timestep = 1.f + static_cast<float>(s) * dt;
        const std::vector<float> tv = pi05::sinusoidal_time_emb(timestep, hidden_ex, cfg.min_period, cfg.max_period);
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
        std::fprintf(stderr, "pi05: ggml_backend_graph_compute failed (%d)\n", static_cast<int>(st));
        return {};
    }

    std::vector<float> out(static_cast<size_t>(chunk) * max_ad);
    ggml_backend_tensor_get(x_final_, out.data(), 0, out.size() * sizeof(float));

    // De-normalise: latent space → real action units.
    for (int64_t t = 0; t < chunk; ++t) {
        float * row = out.data() + static_cast<size_t>(t) * max_ad;
        for (int64_t j = 0; j < cfg.real_action_dim && j < max_ad; ++j) {
            if (action_norm_mode == "MEAN_STD") {
                row[j] = row[j] * (action_std[j] + cfg.norm_eps) + action_mean[j];
            } else {
                const float denom = action_q99[j] - action_q01[j];
                row[j]            = (row[j] + 1.f) * denom * 0.5f + action_q01[j];
            }
        }
    }
    return out;
}

// ────────────────────────────────────────────────────────────────────────────
// predict: convenience wrapper (prepare + compute in one call)
// ────────────────────────────────────────────────────────────────────────────

std::vector<float> Pi05Model::predict(const Inputs & in) {
    using clk        = std::chrono::high_resolution_clock;
    const auto    t0 = clk::now();
    PreparedInput p  = prepare_inputs(in);
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
