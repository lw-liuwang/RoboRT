// Quick smoke test for FasterWAM GGUF inference.
// Loads the GGUF, calls predict() with zero inputs, and checks we get a
// non-empty action chunk back.
//
// Build and run:
//   g++ -std=c++17 test_fasterwam.cpp \
//       -I src/api \
//       -L build/bin -lrobort \
//       -Wl,-rpath,build/bin \
//       -o /tmp/test_fasterwam && \
//   CUDA_VISIBLE_DEVICES=2,3 /tmp/test_fasterwam \
//       models/FasterWAM/robotwin/fasterwam_robotwin.gguf

#include "policy.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const char * ckpt = argc > 1 ? argv[1]
                                 : "models/FasterWAM/robotwin/fasterwam_robotwin.gguf";

    std::printf("fasterwam smoke-test: loading %s\n", ckpt);

    robo::Policy * m = robo::policy_load("", ckpt, "");
    if (!m) {
        std::fprintf(stderr, "FAIL: policy_load returned nullptr\n");
        return 1;
    }

    const robo::PolicyConfig & cfg = robo::policy_config(m);
    std::printf("loaded: action_horizon=%lld  action_dim=%lld  steps=%d\n",
                (long long) cfg.n_suffix,
                (long long) cfg.real_action_dim,
                cfg.num_steps);

    // Dummy inputs – zero state, no images, no language
    const int state_dim  = (int) cfg.real_state_dim;
    const int action_dim = (int) cfg.real_action_dim;
    const int horizon    = (int) cfg.n_suffix;

    std::vector<float> state(state_dim, 0.f);
    std::vector<float> noise((size_t) horizon * (size_t) action_dim, 0.1f);

    robo::PolicyInput in{};
    in.images   = nullptr;
    in.n_images = 0;
    in.lang_tokens = nullptr;
    in.n_lang      = 0;
    in.state       = state.data();
    in.noise       = noise.data();

    std::printf("running predict()...\n");
    std::vector<float> actions = robo::step(m, in);

    if (actions.empty()) {
        std::fprintf(stderr, "FAIL: predict() returned empty vector\n");
        robo::policy_free(m);
        return 1;
    }

    std::printf("PASS: got action chunk of size %zu  (expected %d)\n",
                actions.size(), horizon * action_dim);
    std::printf("  actions[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
                actions[0], actions[1], actions[2], actions[3], actions[4]);

    const robo::PolicyStats & st = robo::last_stats(m);
    std::printf("  timing: total=%.1f ms  prefill=%.1f ms  denoise=%.1f ms\n",
                st.ms_total, st.ms_prefill, st.ms_denoise);

    robo::policy_free(m);
    return 0;
}
