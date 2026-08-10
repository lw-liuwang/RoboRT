// Copyright 2026 VinRobotics
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

#include "adapter.h"
#include "model.h"
#include "server_base.h"
#include "vla.pb.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool decode_image(const vla::Image &             img,
                  std::vector<uint8_t> &         u8,
                  std::vector<float> &           f32,
                  embodied::adapter::ImageView & view) {
    if (img.encoding() == vla::Image::JPEG) {
        int             w = 0, h = 0, ch = 0;
        const auto &    data = img.data();
        unsigned char * px   = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(data.data()),
                                                     static_cast<int>(data.size()), &w, &h, &ch, 3);
        if (!px) {
            std::fprintf(stderr, "vla-server: stbi_load_from_memory failed: %s\n", stbi_failure_reason());
            return false;
        }
        u8.assign(px, px + size_t(3) * w * h);
        stbi_image_free(px);
        view.data     = u8.data();
        view.width    = w;
        view.height   = h;
        view.type     = embodied::adapter::ElementType::U8;
        view.layout   = embodied::adapter::ImageLayout::HWC;
        view.color    = embodied::adapter::ImageColor::RGB;
        view.encoding = embodied::adapter::ImageEncoding::RAW;
        return true;
    } else if (img.encoding() == vla::Image::RGB_U8) {
        const size_t expected = size_t(3) * img.width() * img.height();
        if (img.data().size() != expected) {
            std::fprintf(stderr, "vla-server: RGB_U8 size %zu != 3*%u*%u = %zu\n", img.data().size(), img.width(),
                         img.height(), expected);
            return false;
        }
        u8.assign(reinterpret_cast<const uint8_t *>(img.data().data()),
                  reinterpret_cast<const uint8_t *>(img.data().data()) + expected);
        view.data     = u8.data();
        view.width    = int(img.width());
        view.height   = int(img.height());
        view.type     = embodied::adapter::ElementType::U8;
        view.layout   = embodied::adapter::ImageLayout::HWC;
        view.color    = embodied::adapter::ImageColor::RGB;
        view.encoding = embodied::adapter::ImageEncoding::RAW;
        return true;
    } else if (img.encoding() == vla::Image::F32_RGB_01) {
        const size_t pixels   = size_t(3) * img.width() * img.height();
        const size_t expected = pixels * sizeof(float);
        if (img.data().size() != expected) {
            std::fprintf(stderr, "vla-server: F32_RGB_01 size %zu != 4*3*%u*%u = %zu\n", img.data().size(), img.width(),
                         img.height(), expected);
            return false;
        }

        f32.resize(pixels);
        std::memcpy(f32.data(), img.data().data(), expected);
        view.data     = f32.data();
        view.width    = int(img.width());
        view.height   = int(img.height());
        view.type     = embodied::adapter::ElementType::F32;
        view.layout   = embodied::adapter::ImageLayout::HWC;
        view.color    = embodied::adapter::ImageColor::RGB;
        view.encoding = embodied::adapter::ImageEncoding::RAW;
        return true;
    } else {
        std::fprintf(stderr, "vla-server: unknown image encoding %d\n", int(img.encoding()));
        return false;
    }
}

std::string make_error_response(uint64_t request_id, const std::string & msg) {
    vla::PredictResponse resp;
    resp.set_request_id(request_id);
    resp.set_error(msg);
    return resp.SerializeAsString();
}

int find_non_finite(const float * data, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) {
            return i;
        }
    }
    return -1;
}

void usage(const char * prog) {
    std::fprintf(stderr,
                 "usage: %s [--bind ADDR] [--timing-detail none|phase] [--config PATH] "
                 "[<mmproj.gguf>] <ckpt>\n"
                 "  <mmproj.gguf>           pi0.5 vision-tower mmproj GGUF. Omit for HY-VLA\n"
                 "                          and LingBot-VA combined GGUF checkpoints.\n"
                 "  <ckpt>                  pi0.5, HY-VLA, or LingBot-VA GGUF checkpoint; the\n"
                 "                          architecture is auto-detected from metadata.\n"
                 "  --bind ADDR             ZMQ bind address (default: tcp://*:5555)\n"
                 "  --timing-detail LEVEL   per-request timing breakdown (default: none)\n"
                 "                          'none'  : single ms_inference\n"
                 "                          'phase' : ms_prefill + ms_denoise broken out\n"
                 "  --config PATH           Reserved for compatibility; GGUF checkpoints carry\n"
                 "                          the required runtime metadata.\n",
                 prog);
}

}  // anonymous namespace

// ── VlaServer ─────────────────────────────────────────────────────────

class VlaServer : public vla::serving::ZmqServerBase {
  public:
    VlaServer(const std::string & bind_addr,
              vla::Model *        model,
              const vla::Config & cfg,
              vla::TimingDetail   timing_detail) :
        ZmqServerBase(bind_addr),
        model_(model),
        cfg_(cfg),
        timing_detail_(timing_detail),
        input_adapter_(embodied::adapter::AdapterConfig{
            cfg_.max_state_dim,
            cfg_.max_action_dim,
            cfg_.n_suffix,
            true,
            true,
        }) {}

    ~VlaServer() override { vla::model_free(model_); }

  private:
    std::string handle_request(const uint8_t * data, size_t size) override {
        const uint64_t rid = [&]() -> uint64_t {
            vla::PredictRequest req;
            if (!req.ParseFromArray(data, static_cast<int>(size))) {
                std::fprintf(stderr, "vla-server: PredictRequest parse failed (size=%zu)\n", size);
                return 0;
            }
            return req.request_id();
        }();

        vla::PredictRequest req;
        if (!req.ParseFromArray(data, static_cast<int>(size))) {
            return make_error_response(0, "request parse failed");
        }

        if (cfg_.n_img > 0 && req.images_size() < 1 && req.precomputed_img_emb_size() == 0) {
            return make_error_response(req.request_id(), "PredictRequest must contain images or precomputed_img_emb");
        }
        if (req.lang_tokens_size() < 1 || req.lang_tokens_size() > int(cfg_.n_lang)) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "lang_tokens length %d out of range [1, %lld]", req.lang_tokens_size(),
                          (long long) cfg_.n_lang);
            return make_error_response(req.request_id(), buf);
        }
        for (int t = 0; t < req.lang_tokens_size(); ++t) {
            if (req.lang_tokens(t) < 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "lang_tokens[%d] = %d is negative", t, req.lang_tokens(t));
                return make_error_response(req.request_id(), buf);
            }
        }
        if (req.state_size() != int(cfg_.max_state_dim)) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "state length %d != expected %lld", req.state_size(),
                          (long long) cfg_.max_state_dim);
            return make_error_response(req.request_id(), buf);
        }
        const int expected_noise_n = int(cfg_.n_suffix * cfg_.max_action_dim);
        if (req.noise_size() != 0 && req.noise_size() != expected_noise_n) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "noise length %d != 0 or %d (chunk_size * action_dim)", req.noise_size(),
                          expected_noise_n);
            return make_error_response(req.request_id(), buf);
        }

        const bool         use_precomputed = req.precomputed_img_emb_size() > 0;
        std::vector<float> precomputed_emb;
        int                precomputed_n_views = 0;

        const int                                 n_views = req.images_size();
        std::vector<std::vector<uint8_t>>         u8_bufs(n_views);
        std::vector<std::vector<float>>           f32_bufs(n_views);
        std::vector<embodied::adapter::ImageView> img_views(n_views);

        if (use_precomputed) {
            if (cfg_.n_img > 0) {
                precomputed_n_views = static_cast<int>(req.precomputed_img_emb_n_views());
                const int per_view  = int(cfg_.n_img * cfg_.hidden);
                const int expected  = per_view * precomputed_n_views;
                if (precomputed_n_views < 1 || req.precomputed_img_emb_size() != expected) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf),
                                  "precomputed_img_emb size %d != %d (n_views=%d * n_img_per_view=%lld * hidden=%lld)",
                                  req.precomputed_img_emb_size(), expected, precomputed_n_views, (long long) cfg_.n_img,
                                  (long long) cfg_.hidden);
                    return make_error_response(req.request_id(), buf);
                }
            } else {
                if (cfg_.hidden <= 0 || req.precomputed_img_emb_size() % int(cfg_.hidden) != 0) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf),
                                  "precomputed_img_emb size %d is not divisible by hidden=%lld for token-style "
                                  "precomputed embeddings",
                                  req.precomputed_img_emb_size(), (long long) cfg_.hidden);
                    return make_error_response(req.request_id(), buf);
                }
                precomputed_n_views = req.precomputed_img_emb_size() / int(cfg_.hidden);
                if (precomputed_n_views < 1) {
                    return make_error_response(req.request_id(), "precomputed_img_emb token count must be >= 1");
                }
            }
            precomputed_emb.assign(req.precomputed_img_emb().begin(), req.precomputed_img_emb().end());
            const int bad = find_non_finite(precomputed_emb.data(), static_cast<int>(precomputed_emb.size()));
            if (bad >= 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "precomputed_img_emb[%d] = %g is not finite (NaN/Inf)", bad,
                              precomputed_emb[bad]);
                return make_error_response(req.request_id(), buf);
            }
        } else {
            bool decode_ok = true;
            for (int v = 0; v < n_views; ++v) {
                if (!decode_image(req.images(v), u8_bufs[v], f32_bufs[v], img_views[v])) {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "image[%d] decode failed", v);
                    return make_error_response(req.request_id(), buf);
                }
            }
        }

        embodied::adapter::Observation observation;
        observation.language_tokens.assign(req.lang_tokens().begin(), req.lang_tokens().end());
        observation.proprioception.assign(req.state().begin(), req.state().end());
        observation.images = std::move(img_views);
        {
            const int bad =
                find_non_finite(observation.proprioception.data(), static_cast<int>(observation.proprioception.size()));
            if (bad >= 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "state[%d] = %g is not finite (NaN/Inf)", bad,
                              observation.proprioception[bad]);
                return make_error_response(req.request_id(), buf);
            }
        }
        if (req.noise_size() == expected_noise_n) {
            observation.noise.assign(req.noise().begin(), req.noise().end());
            const int bad = find_non_finite(observation.noise.data(), static_cast<int>(observation.noise.size()));
            if (bad >= 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "noise[%d] = %g is not finite (NaN/Inf)", bad, observation.noise[bad]);
                return make_error_response(req.request_id(), buf);
            }
        }

        embodied::adapter::ModelInputStorage model_input;
        embodied::adapter::AdapterStatus     adapter_status = input_adapter_.build(observation, &model_input);
        if (!adapter_status.ok) {
            return make_error_response(req.request_id(), adapter_status.message);
        }

        if (use_precomputed) {
            model_input.inputs.precomputed_img_emb = precomputed_emb.data();
            model_input.inputs.n_img_views         = precomputed_n_views;
            model_input.inputs.images              = nullptr;
            model_input.inputs.n_images            = 0;
        } else {
            model_input.inputs.precomputed_img_emb = nullptr;
            model_input.inputs.n_img_views         = 0;
        }
        std::vector<int32_t> attn_mask_vec;
        if (req.attention_mask_size() > 0) {
            attn_mask_vec.assign(req.attention_mask().begin(), req.attention_mask().end());
        }
        model_input.inputs.attention_mask   = attn_mask_vec.empty() ? nullptr : attn_mask_vec.data();
        model_input.inputs.attention_mask_n = static_cast<int>(attn_mask_vec.size());
        model_input.inputs.timing_detail    = timing_detail_;

        std::vector<float> action_chunk = vla::predict(model_, model_input.inputs);
        const auto &       st           = vla::last_stats(model_);

        if (action_chunk.empty()) {
            return make_error_response(req.request_id(), "predict failed");
        }

        vla::PredictResponse resp;
        resp.set_request_id(req.request_id());
        resp.mutable_action_chunk()->Reserve(static_cast<int>(action_chunk.size()));
        for (float v : action_chunk) {
            resp.add_action_chunk(v);
        }
        resp.set_chunk_size(static_cast<uint32_t>(cfg_.n_suffix));
        resp.set_action_dim(static_cast<uint32_t>(cfg_.max_action_dim));
        resp.set_latency_ms_total(st.ms_total);
        resp.set_latency_ms_vision(st.ms_vision);
        resp.set_latency_ms_inference(st.ms_inference);
        resp.set_latency_ms_prefill(st.ms_prefill);
        resp.set_latency_ms_denoise(st.ms_denoise);

        if (served() % 10 == 0) {
            const float ms_other = std::max(0.f, st.ms_total - st.ms_vision - st.ms_inference);
            if (timing_detail_ == vla::TimingDetail::PHASE) {
                std::printf(
                    "vla-server: rid=%llu  served=%llu  total=%.1f ms  "
                    "vision=%.1f  inf=%.1f (prefill=%.1f + denoise=%.1f)  other=%.1f\n",
                    (unsigned long long) req.request_id(), (unsigned long long) served(), st.ms_total, st.ms_vision,
                    st.ms_inference, st.ms_prefill, st.ms_denoise, ms_other);
            } else {
                std::printf(
                    "vla-server: rid=%llu  served=%llu  total=%.1f ms  "
                    "vision=%.1f  inf=%.1f  other=%.1f\n",
                    (unsigned long long) req.request_id(), (unsigned long long) served(), st.ms_total, st.ms_vision,
                    st.ms_inference, ms_other);
            }
            std::fflush(stdout);
        }

        return resp.SerializeAsString();
    }

    vla::Model *                            model_ = nullptr;
    const vla::Config &                     cfg_;
    vla::TimingDetail                       timing_detail_;
    embodied::adapter::VlaModelInputAdapter input_adapter_;
};

// ── main ──────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string       bind_addr = "tcp://*:5555";
    std::string       mmproj_path;
    std::string       ckpt_path;
    std::string       config_path;
    vla::TimingDetail timing_detail = vla::TimingDetail::NONE;

    std::vector<std::string> positionals;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--timing-detail" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "none") {
                timing_detail = vla::TimingDetail::NONE;
            } else if (v == "phase") {
                timing_detail = vla::TimingDetail::PHASE;
            } else {
                std::fprintf(stderr, "vla-server: bad --timing-detail value '%s'\n", v.c_str());
                usage(argv[0]);
                return 1;
            }
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            positionals.push_back(std::move(a));
        }
    }
    if (positionals.size() == 1) {
        ckpt_path = positionals[0];
    } else if (positionals.size() == 2) {
        mmproj_path = positionals[0];
        ckpt_path   = positionals[1];
    } else {
        std::fprintf(stderr,
                     "vla-server: expected 1 or 2 positional args "
                     "(<mmproj.gguf> <ckpt> for pi0.5, or just <ckpt> "
                     "for HY-VLA/LingBot-VA), got %zu\n",
                     positionals.size());
        usage(argv[0]);
        return 1;
    }

    std::printf("vla-server: loading model ...\n");
    if (!mmproj_path.empty()) {
        std::printf("  mmproj: %s\n", mmproj_path.c_str());
    }
    std::printf("  ckpt:   %s\n", ckpt_path.c_str());
    if (!config_path.empty()) {
        std::printf("  config: %s\n", config_path.c_str());
    }
    vla::Model * model = vla::model_load(mmproj_path, ckpt_path, config_path);
    if (!model) {
        std::fprintf(stderr, "vla-server: model_load failed\n");
        return 1;
    }
    const auto & cfg = vla::model_config(model);
    std::printf(
        "vla-server: loaded. chunk_size=%lld  action_dim=%lld  "
        "n_lang=%lld  hidden=%lld  expert_h=%lld  timing_detail=%s\n",
        (long long) cfg.n_suffix, (long long) cfg.max_action_dim, (long long) cfg.n_lang, (long long) cfg.hidden,
        (long long) cfg.expert_h, timing_detail == vla::TimingDetail::PHASE ? "phase" : "none");

    VlaServer server(bind_addr, model, cfg, timing_detail);
    server.run();

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
