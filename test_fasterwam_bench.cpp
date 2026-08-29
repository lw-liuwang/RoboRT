// Benchmark for FasterWAM cached inference path.
// Measures per-phase timing in both the first call (cold) and subsequent calls (warm).
//
// Build and run (in robort container):
//   g++ -std=c++17 -O2 test_fasterwam_bench.cpp \
//       -I src/api \
//       -L build/bin -lrobort \
//       -Wl,-rpath,$(pwd)/build/bin \
//       -o /tmp/test_fasterwam_bench
//
//   CUDA_VISIBLE_DEVICES=3 /tmp/test_fasterwam_bench \
//       models/FasterWAM/robotwin/fasterwam_robotwin.gguf [N_WARMUP] [N_BENCH]

#include "policy.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using Clock = std::chrono::steady_clock;
static float ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

int main(int argc, char ** argv) {
    const char * ckpt     = argc > 1 ? argv[1] : "models/FasterWAM/robotwin/fasterwam_robotwin.gguf";
    const int    n_warmup = argc > 2 ? std::atoi(argv[2]) : 3;
    const int    n_bench  = argc > 3 ? std::atoi(argv[3]) : 10;

    std::printf("Loading %s...\n", ckpt);
    robo::Policy * m = robo::policy_load("", ckpt, "");
    if (!m) { std::fprintf(stderr, "FAIL: policy_load\n"); return 1; }

    const robo::PolicyConfig & cfg = robo::policy_config(m);
    const int state_dim  = (int) cfg.real_state_dim;
    const int action_dim = (int) cfg.real_action_dim;
    const int horizon    = (int) cfg.n_suffix;

    std::vector<float> state(state_dim, 0.f);
    std::vector<float> noise((size_t) horizon * (size_t) action_dim, 0.1f);

    robo::PolicyInput in{};
    in.images      = nullptr;
    in.n_images    = 0;
    in.lang_tokens = nullptr;
    in.n_lang      = 0;
    in.state       = state.data();
    in.noise       = noise.data();

    // -----------------------------------------------------------------------
    // Warm-up: first call cold, remaining calls warm (const_buf cached)
    // -----------------------------------------------------------------------
    std::printf("\nWarm-up calls (%d):\n", n_warmup);
    for (int i = 0; i < n_warmup; ++i) {
        // Vary state slightly each call to simulate real usage
        state[0] = (float) i * 0.001f;
        auto t0 = Clock::now();
        auto actions = robo::step(m, in);
        auto t1 = Clock::now();
        const auto & st = robo::last_stats(m);
        std::printf("  call %d: total=%.1f ms  prefill=%.1f ms  denoise=%.1f ms  actions[0]=%.4f\n",
                    i, ms(t0, t1), st.ms_prefill, st.ms_denoise, actions.empty() ? 0.f : actions[0]);
    }

    // -----------------------------------------------------------------------
    // Benchmark: cached path (text unchanged, state varies)
    // -----------------------------------------------------------------------
    std::printf("\nBenchmark calls (%d, cached text):\n", n_bench);
    std::vector<float> total_ms(n_bench), prefill_ms(n_bench), denoise_ms(n_bench), wall_ms(n_bench);
    for (int i = 0; i < n_bench; ++i) {
        state[0] = (float)(n_warmup + i) * 0.001f;
        auto t0 = Clock::now();
        auto actions = robo::step(m, in);
        auto t1 = Clock::now();
        (void) actions;
        const auto & st = robo::last_stats(m);
        total_ms[i]   = st.ms_total;
        prefill_ms[i] = st.ms_prefill;
        denoise_ms[i] = st.ms_denoise;
        wall_ms[i]    = ms(t0, t1);
    }

    auto avg = [](const std::vector<float> & v) {
        return std::accumulate(v.begin(), v.end(), 0.f) / (float) v.size();
    };
    auto med = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v[v.size()/2];
    };
    auto p95 = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v[(size_t)(v.size() * 0.95f)];
    };

    std::printf("\n--- Results (n=%d, cached) ---\n", n_bench);
    std::printf("  Wall:    avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(wall_ms),    med(wall_ms),    p95(wall_ms));
    std::printf("  Total:   avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(total_ms),   med(total_ms),   p95(total_ms));
    std::printf("  Prefill: avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(prefill_ms), med(prefill_ms), p95(prefill_ms));
    std::printf("  Denoise: avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(denoise_ms), med(denoise_ms), p95(denoise_ms));

    // -----------------------------------------------------------------------
    // Benchmark 2: text changes every call (measures prefill re-trigger cost)
    // -----------------------------------------------------------------------
    std::printf("\nBenchmark calls (%d, text changed every call):\n", n_bench);
    std::vector<float> text_emb((size_t) horizon * 4096, 0.f);  // fake T5 embeddings, text_seq=horizon
    in.precomputed_img_emb = nullptr;
    in.n_img_views = 0;

    std::vector<float> total2(n_bench), prefill2(n_bench), denoise2(n_bench), wall2(n_bench);
    for (int i = 0; i < n_bench; ++i) {
        // Change text every call
        text_emb[0] = (float)(i + 1) * 0.01f;
        in.precomputed_img_emb = text_emb.data();
        in.n_img_views = 1;  // 1 text token for speed
        state[0] = (float)(n_warmup + n_bench + i) * 0.001f;
        auto t0 = Clock::now();
        auto actions = robo::step(m, in);
        auto t1 = Clock::now();
        (void) actions;
        const auto & st = robo::last_stats(m);
        total2[i]   = st.ms_total;
        prefill2[i] = st.ms_prefill;
        denoise2[i] = st.ms_denoise;
        wall2[i]    = ms(t0, t1);
        std::printf("  call %d: wall=%.1f ms  prefill=%.1f ms  denoise=%.1f ms\n",
                    i, wall2[i], prefill2[i], denoise2[i]);
    }
    std::printf("\n--- Results (n=%d, text-changed) ---\n", n_bench);
    std::printf("  Wall:    avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(wall2),    med(wall2),    p95(wall2));
    std::printf("  Total:   avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(total2),   med(total2),   p95(total2));
    std::printf("  Prefill: avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(prefill2), med(prefill2), p95(prefill2));
    std::printf("  Denoise: avg=%.2f  med=%.2f  p95=%.2f ms\n",  avg(denoise2), med(denoise2), p95(denoise2));

    robo::policy_free(m);
    return 0;
}
