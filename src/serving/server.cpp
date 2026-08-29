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

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
                 "usage: %s [--bind ADDR] [--async] [--timing-detail none|phase] [--config PATH] "
                 "[<mmproj.gguf>] <ckpt>\n"
                 "  <mmproj.gguf>           pi0.5 vision-tower mmproj GGUF. Omit for HY-VLA,\n"
                 "                          FasterWAM, and other single-GGUF architectures.\n"
                 "  <ckpt>                  GGUF checkpoint; the architecture is auto-detected\n"
                 "                          from the GGUF metadata (pi05/hy_vla/fasterwam).\n"
                 "  --bind ADDR             ZMQ bind address (default: tcp://*:5555)\n"
                 "  --async                 dual-thread ROUTER pipeline: vision encoding of\n"
                 "                          request N+1 overlaps graph compute of request N\n"
                 "                          (pi0.5 only; off = single-threaded REP, bit-exact)\n"
                 "  --timing-detail LEVEL   per-request timing breakdown (default: none)\n"
                 "                          'none'  : single ms_inference\n"
                 "                          'phase' : ms_prefill + ms_denoise broken out\n"
                 "  --config PATH           Reserved for compatibility; GGUF checkpoints carry\n"
                 "                          the required runtime metadata.\n",
                 prog);
}

}  // anonymous namespace

// ── VlaServer ─────────────────────────────────────────────────────────

namespace {

// Parsed + validated request, before any model execution.  Holds every buffer
// the adapter-built vla::Inputs borrows from; only needs to stay alive for the
// duration of the parse call (vision encoding happens inside that window in
// async mode, so after vla::prepare the owning vla::PreparedInput can take over).
struct ParsedRequest {
    uint64_t request_id = 0;
    bool ok = true;
    std::string error;   // serialised error response when ok == false

    std::vector<std::vector<uint8_t>> u8_bufs;
    std::vector<std::vector<float>>   f32_bufs;
    std::vector<embodied::adapter::ImageView> img_views;
    std::vector<float> precomputed_emb;
    int  precomputed_n_views = 0;
    bool use_precomputed = false;
    std::vector<int32_t> attn_mask_vec;

    embodied::adapter::Observation observation;
    embodied::adapter::ModelInputStorage model_input;
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // anonymous namespace

class VlaServer : public vla::serving::ZmqServerBase {
  public:
    VlaServer(const std::string & bind_addr,
              bool                async,
              vla::Model *        model,
              const vla::Config & cfg,
              vla::TimingDetail   timing_detail) :
        ZmqServerBase(bind_addr, async ? zmq::socket_type::router : zmq::socket_type::rep),
        async_(async),
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

    void run() override {
        if (async_) {
            run_async_router();
        } else {
            // Sync mode: the original single-threaded REP loop, untouched.
            ZmqServerBase::run();
        }
    }

  private:
    // ── Shared: parse + validate + decode + adapter build (no model exec) ──
    bool parse_request(const uint8_t * data, size_t size, ParsedRequest * out) {
        vla::PredictRequest req;
        if (!req.ParseFromArray(data, static_cast<int>(size))) {
            out->ok    = false;
            out->error = make_error_response(0, "request parse failed");
            return false;
        }
        out->request_id = req.request_id();

        if (cfg_.n_img > 0 && req.images_size() < 1 && req.precomputed_img_emb_size() == 0) {
            out->ok    = false;
            out->error = make_error_response(req.request_id(),
                                             "PredictRequest must contain images or precomputed_img_emb");
            return false;
        }
        if (req.lang_tokens_size() < 1 || req.lang_tokens_size() > int(cfg_.n_lang)) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "lang_tokens length %d out of range [1, %lld]",
                          req.lang_tokens_size(), (long long) cfg_.n_lang);
            out->ok = false;
            out->error = make_error_response(req.request_id(), buf);
            return false;
        }
        for (int t = 0; t < req.lang_tokens_size(); ++t) {
            if (req.lang_tokens(t) < 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "lang_tokens[%d] = %d is negative", t, req.lang_tokens(t));
                out->ok = false;
                out->error = make_error_response(req.request_id(), buf);
                return false;
            }
        }
        if (req.state_size() != int(cfg_.max_state_dim)) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "state length %d != expected %lld", req.state_size(),
                          (long long) cfg_.max_state_dim);
            out->ok = false;
            out->error = make_error_response(req.request_id(), buf);
            return false;
        }
        const int expected_noise_n = int(cfg_.n_suffix * cfg_.max_action_dim);
        if (req.noise_size() != 0 && req.noise_size() != expected_noise_n) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "noise length %d != 0 or %d (chunk_size * action_dim)",
                          req.noise_size(), expected_noise_n);
            out->ok = false;
            out->error = make_error_response(req.request_id(), buf);
            return false;
        }
        if (req.prev_chunk_size() > 0) {
            if (cfg_.real_action_dim <= 0 || req.prev_chunk_size() % int(cfg_.real_action_dim) != 0) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "prev_chunk size %d is not a multiple of real_action_dim=%lld",
                              req.prev_chunk_size(), (long long) cfg_.real_action_dim);
                out->ok = false;
                out->error = make_error_response(req.request_id(), buf);
                return false;
            }
        }

        out->use_precomputed = req.precomputed_img_emb_size() > 0;
        const int n_views    = req.images_size();
        out->u8_bufs.assign(n_views, {});
        out->f32_bufs.assign(n_views, {});
        out->img_views.resize(n_views);

        if (out->use_precomputed) {
            if (cfg_.n_img > 0) {
                out->precomputed_n_views = static_cast<int>(req.precomputed_img_emb_n_views());
                const int per_view       = int(cfg_.n_img * cfg_.hidden);
                const int expected       = per_view * out->precomputed_n_views;
                if (out->precomputed_n_views < 1 || req.precomputed_img_emb_size() != expected) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf),
                                  "precomputed_img_emb size %d != %d (n_views=%d * n_img_per_view=%lld * hidden=%lld)",
                                  req.precomputed_img_emb_size(), expected, out->precomputed_n_views,
                                  (long long) cfg_.n_img, (long long) cfg_.hidden);
                    out->ok = false;
                    out->error = make_error_response(req.request_id(), buf);
                    return false;
                }
            } else {
                if (cfg_.hidden <= 0 || req.precomputed_img_emb_size() % int(cfg_.hidden) != 0) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf),
                                  "precomputed_img_emb size %d is not divisible by hidden=%lld for token-style "
                                  "precomputed embeddings",
                                  req.precomputed_img_emb_size(), (long long) cfg_.hidden);
                    out->ok = false;
                    out->error = make_error_response(req.request_id(), buf);
                    return false;
                }
                out->precomputed_n_views = req.precomputed_img_emb_size() / int(cfg_.hidden);
                if (out->precomputed_n_views < 1) {
                    out->ok    = false;
                    out->error = make_error_response(req.request_id(), "precomputed_img_emb token count must be >= 1");
                    return false;
                }
            }
            out->precomputed_emb.assign(req.precomputed_img_emb().begin(), req.precomputed_img_emb().end());
            const int bad = find_non_finite(out->precomputed_emb.data(),
                                            static_cast<int>(out->precomputed_emb.size()));
            if (bad >= 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "precomputed_img_emb[%d] = %g is not finite (NaN/Inf)", bad,
                              out->precomputed_emb[bad]);
                out->ok = false;
                out->error = make_error_response(req.request_id(), buf);
                return false;
            }
        } else {
            for (int v = 0; v < n_views; ++v) {
                if (!decode_image(req.images(v), out->u8_bufs[v], out->f32_bufs[v], out->img_views[v])) {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "image[%d] decode failed", v);
                    out->ok = false;
                    out->error = make_error_response(req.request_id(), buf);
                    return false;
                }
            }
        }

        out->observation.language_tokens.assign(req.lang_tokens().begin(), req.lang_tokens().end());
        out->observation.proprioception.assign(req.state().begin(), req.state().end());
        out->observation.images = std::move(out->img_views);
        {
            const int bad = find_non_finite(out->observation.proprioception.data(),
                                            static_cast<int>(out->observation.proprioception.size()));
            if (bad >= 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "state[%d] = %g is not finite (NaN/Inf)", bad,
                              out->observation.proprioception[bad]);
                out->ok = false;
                out->error = make_error_response(req.request_id(), buf);
                return false;
            }
        }
        if (req.noise_size() == expected_noise_n) {
            out->observation.noise.assign(req.noise().begin(), req.noise().end());
            const int bad = find_non_finite(out->observation.noise.data(),
                                            static_cast<int>(out->observation.noise.size()));
            if (bad >= 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "noise[%d] = %g is not finite (NaN/Inf)", bad,
                              out->observation.noise[bad]);
                out->ok = false;
                out->error = make_error_response(req.request_id(), buf);
                return false;
            }
        }
        if (req.prev_chunk_size() > 0) {
            out->observation.prev_chunk.assign(req.prev_chunk().begin(), req.prev_chunk().end());
            const int bad = find_non_finite(out->observation.prev_chunk.data(),
                                            static_cast<int>(out->observation.prev_chunk.size()));
            if (bad >= 0) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "prev_chunk[%d] = %g is not finite (NaN/Inf)", bad,
                              out->observation.prev_chunk[bad]);
                out->ok = false;
                out->error = make_error_response(req.request_id(), buf);
                return false;
            }
        }

        embodied::adapter::AdapterStatus adapter_status = input_adapter_.build(out->observation, &out->model_input);
        if (!adapter_status.ok) {
            out->ok    = false;
            out->error = make_error_response(req.request_id(), adapter_status.message);
            return false;
        }

        if (out->use_precomputed) {
            out->model_input.inputs.precomputed_img_emb = out->precomputed_emb.data();
            out->model_input.inputs.n_img_views         = out->precomputed_n_views;
            out->model_input.inputs.images              = nullptr;
            out->model_input.inputs.n_images            = 0;
        } else {
            out->model_input.inputs.precomputed_img_emb = nullptr;
            out->model_input.inputs.n_img_views         = 0;
        }
        if (req.attention_mask_size() > 0) {
            out->attn_mask_vec.assign(req.attention_mask().begin(), req.attention_mask().end());
        }
        out->model_input.inputs.attention_mask   = out->attn_mask_vec.empty() ? nullptr : out->attn_mask_vec.data();
        out->model_input.inputs.attention_mask_n = static_cast<int>(out->attn_mask_vec.size());
        out->model_input.inputs.timing_detail    = timing_detail_;
        return true;
    }

    // ── Sync path (bit-exact baseline): parse + predict + response ──
    std::string handle_request(const uint8_t * data, size_t size) override {
        ParsedRequest p;
        if (!parse_request(data, size, &p)) {
            return p.error;
        }

        std::vector<float> action_chunk = vla::predict(model_, p.model_input.inputs);
        const auto &       st           = vla::last_stats(model_);
        if (action_chunk.empty()) {
            return make_error_response(p.request_id, "predict failed");
        }

        vla::PredictResponse resp;
        resp.set_request_id(p.request_id);
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
                    (unsigned long long) p.request_id, (unsigned long long) served(), st.ms_total, st.ms_vision,
                    st.ms_inference, st.ms_prefill, st.ms_denoise, ms_other);
            } else {
                std::printf(
                    "vla-server: rid=%llu  served=%llu  total=%.1f ms  "
                    "vision=%.1f  inf=%.1f  other=%.1f\n",
                    (unsigned long long) p.request_id, (unsigned long long) served(), st.ms_total, st.ms_vision,
                    st.ms_inference, ms_other);
            }
            std::fflush(stdout);
        }
        return resp.SerializeAsString();
    }

    // ── Async compute half: PreparedInput -> response ──
    std::string compute_and_build_response(uint64_t                  request_id,
                                           const vla::PreparedInput & prepared,
                                           double                    recv_ts_ms) {
        const double        ti0 = now_ms();
        std::vector<float>  action_chunk = vla::compute(model_, prepared);
        const double        ti1 = now_ms();
        if (action_chunk.empty()) {
            return make_error_response(request_id, "compute failed");
        }
        vla::PredictResponse resp;
        resp.set_request_id(request_id);
        resp.mutable_action_chunk()->Reserve(static_cast<int>(action_chunk.size()));
        for (float v : action_chunk) {
            resp.add_action_chunk(v);
        }
        resp.set_chunk_size(static_cast<uint32_t>(cfg_.n_suffix));
        resp.set_action_dim(static_cast<uint32_t>(cfg_.max_action_dim));
        resp.set_latency_ms_total(float(ti1 - recv_ts_ms));
        resp.set_latency_ms_inference(float(ti1 - ti0));
        return resp.SerializeAsString();
    }

    // ── Async mode: ROUTER dual-thread pipeline ──
    // Receive thread: recv -> parse -> vla::prepare (vision + lang lookup).
    // Compute thread:  vla::compute.  Request N+1's pre-processing overlaps
    // request N's graph compute (true pipelining; a REP socket's lock-step
    // semantics would serialise these, hence the ROUTER).
    //
    // The receive thread is the SOLE owner of the ROUTER socket: it polls,
    // receives, and drains+sends pending replies.  The compute thread never
    // touches the socket, so no socket mutex is needed.  (Holding a mutex
    // across poll and immediately re-acquiring it in a tight loop starves a
    // concurrent sender via futex self-handoff; the reply queue + drain here
    // avoids that entirely.)
    void run_async_router() {
        std::signal(SIGINT, ZmqServerBase::signal_shutdown);
        std::signal(SIGTERM, ZmqServerBase::signal_shutdown);

        struct Job {
            zmq::message_t identity;   // ROUTER frame 0
            uint64_t       request_id = 0;
            vla::PreparedInput prepared;
            double recv_ts_ms = 0;
        };
        struct Reply {
            zmq::message_t identity;
            std::string    data;
        };

        std::mutex              q_mu;  // guards jobs and replies
        std::condition_variable q_cv;
        std::deque<Job>         jobs;
        std::deque<Reply>       replies;
        std::atomic<bool>       done{false};

        auto enqueue_reply = [&](zmq::message_t identity, std::string data) {
            std::lock_guard<std::mutex> lk(q_mu);
            replies.push_back(Reply{std::move(identity), std::move(data)});
        };

        // Compute thread: pop job -> graph compute + de-normalise -> reply.
        std::thread compute_thread([&] {
            for (;;) {
                Job job;
                {
                    std::unique_lock<std::mutex> lk(q_mu);
                    q_cv.wait(lk, [&] { return !jobs.empty() || done.load(); });
                    if (jobs.empty() && done.load()) {
                        break;
                    }
                    job = std::move(jobs.front());
                    jobs.pop_front();
                }
                const double c0 = now_ms();
                const std::string resp = compute_and_build_response(job.request_id, job.prepared, job.recv_ts_ms);
                const double c1 = now_ms();
                enqueue_reply(std::move(job.identity), resp);
                ++served_;
                std::printf("vla-server[async]: rid=%llu  served=%llu  e2e=%.1f ms  compute=%.1f ms  prep=%.1f ms\n",
                            (unsigned long long) job.request_id, (unsigned long long) served_, c1 - job.recv_ts_ms,
                            c1 - c0, c0 - job.recv_ts_ms);
                std::fflush(stdout);
            }
        });

        std::printf("vla-server[async]: ROUTER dual-thread pipeline bound to %s. ready.\n", bind_addr_.c_str());

        zmq::pollitem_t poll[] = {
            { static_cast<void *>(sock_), 0, ZMQ_POLLIN, 0 }
        };

        // Receive thread (this one): sole owner of the socket.  Each cycle
        // first drains pending replies (the compute thread never sends), then
        // polls for new requests.  A short poll keeps reply latency ~10 ms.
        while (!s_shutdown_.load(std::memory_order_relaxed)) {
            {
                std::lock_guard<std::mutex> lk(q_mu);
                while (!replies.empty()) {
                    Reply r = std::move(replies.front());
                    replies.pop_front();
                    sock_.send(std::move(r.identity), zmq::send_flags::sndmore);
                    sock_.send(zmq::str_buffer(""), zmq::send_flags::sndmore);
                    sock_.send(zmq::buffer(r.data), zmq::send_flags::none);
                }
            }

            try {
                zmq::poll(poll, 1, std::chrono::milliseconds(10));
            } catch (const zmq::error_t & e) {
                if (e.num() == EINTR) {
                    continue;
                }
                throw;
            }
            if (!(poll[0].revents & ZMQ_POLLIN)) {
                continue;
            }

            // REQ -> ROUTER arrives as [identity, empty-delimiter, body]
            zmq::message_t identity, empty, body;
            try {
                if (!sock_.recv(identity, zmq::recv_flags::none)) continue;
                if (!sock_.recv(empty, zmq::recv_flags::none)) continue;
                if (!sock_.recv(body, zmq::recv_flags::none)) continue;
            } catch (const zmq::error_t & e) {
                if (e.num() == EINTR) {
                    continue;
                }
                throw;
            }

            const double recv_ts = now_ms();

            Job job;
            job.identity = std::move(identity);
            ParsedRequest parsed;
            if (!parse_request(static_cast<const uint8_t *>(body.data()), body.size(), &parsed)) {
                enqueue_reply(std::move(job.identity), parsed.error);
                continue;
            }
            job.request_id = parsed.request_id;
            job.recv_ts_ms = recv_ts;
            job.prepared   = vla::prepare(model_, parsed.model_input.inputs);
            if (!job.prepared.ok) {
                enqueue_reply(std::move(job.identity), make_error_response(job.request_id, job.prepared.error));
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(q_mu);
                jobs.push_back(std::move(job));
            }
            q_cv.notify_one();
        }

        done.store(true);
        q_cv.notify_all();
        compute_thread.join();  // drain in-flight jobs, then exit
        std::printf("vla-server[async]: shutting down (served %llu requests)\n", (unsigned long long) served_);
    }

    bool                                     async_ = false;
    vla::Model *                             model_ = nullptr;
    const vla::Config &                      cfg_;
    vla::TimingDetail                        timing_detail_;
    embodied::adapter::VlaModelInputAdapter  input_adapter_;
};

// ── main ──────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string       bind_addr   = "tcp://*:5555";
    std::string       mmproj_path;
    std::string       ckpt_path;
    std::string       config_path;
    vla::TimingDetail timing_detail = vla::TimingDetail::NONE;
    bool              async         = false;

    std::vector<std::string> positionals;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--async") {
            async = true;
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

    std::printf("robort-server: loading model ...\n");
    if (!mmproj_path.empty()) {
        std::printf("  mmproj: %s\n", mmproj_path.c_str());
    }
    std::printf("  ckpt:   %s\n", ckpt_path.c_str());
    if (!config_path.empty()) {
        std::printf("  config: %s\n", config_path.c_str());
    }
    if (async) {
        std::printf("  mode:   async (ROUTER dual-thread pipeline)\n");
    }
    vla::Model * model = vla::model_load(mmproj_path, ckpt_path, config_path);
    if (!model) {
        std::fprintf(stderr, "robort-server: model_load failed\n");
        return 1;
    }
    const auto & cfg = vla::model_config(model);
    std::printf("robort-server: loaded. chunk_size=%lld  action_dim=%lld  "
                "n_lang=%lld  hidden=%lld  expert_h=%lld  timing_detail=%s\n",
                (long long) cfg.n_suffix, (long long) cfg.max_action_dim, (long long) cfg.n_lang, (long long) cfg.hidden,
                (long long) cfg.expert_h, timing_detail == vla::TimingDetail::PHASE ? "phase" : "none");

    VlaServer server(bind_addr, async, model, cfg, timing_detail);
    server.run();

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
