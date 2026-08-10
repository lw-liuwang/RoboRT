// Self-check for the vla-pi05 engine: loads a pi05 mmproj + checkpoint,
// runs one predict per request with a fixed image/token input, and dumps the
// action chunk to a file.
//
// Run twice (e.g. the RoboRT wrapper engine vs this fork module) with
// VLA_PI05_SEED=42 and byte-compare the dumps: the outputs must match
// exactly. Random noise is regenerated per predict from the seeded RNG, so
// the dump captures the LAST iteration's action vector.
//
// Build: the vla-pi05-selfcheck CMake target. Manual equivalent:
//   g++ -O2 -std=c++17 selfcheck.cpp -I .. -I <ggml/include> -L <build>
//       -lvla-pi05 -lmtmd -lllama -lggml -lggml-base -lggml-cuda -lggml-cpu
//       -Wl,-rpath,<build> -o vla-pi05-selfcheck
//
// usage: vla-pi05-selfcheck mmproj.gguf pi05.gguf out.txt [n_lang]
#include "model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s mmproj.gguf pi05.gguf out.txt [n_lang]\n", argv[0]);
        return 1;
    }
    const char * mmproj = argv[1];
    const char * ckpt   = argv[2];
    const char * out    = argv[3];
    const int    n_lang = argc > 4 ? atoi(argv[4]) : 47;

    vla::Model * m = vla::model_load(mmproj, ckpt, "");
    if (!m) {
        fprintf(stderr, "model_load failed\n");
        return 1;
    }
    vla::model_config(m);  // compile-check the config accessor

    // 224x224 RGB views (fixed content so both runs see identical inputs)
    const int            img_sz  = 224;
    const size_t         per_pix = (size_t) img_sz * img_sz * 3;
    std::vector<uint8_t> img(per_pix);
    srand(12345);
    for (size_t i = 0; i < per_pix; ++i) {
        img[i] = (uint8_t) (rand() % 256);
    }
    std::vector<vla::ImageView> views(2);
    views[0] = { img.data(), img_sz, img_sz, vla::PixelFormat::U8 };
    views[1] = { img.data(), img_sz, img_sz, vla::PixelFormat::U8 };

    // real Gemma language tokens (libero_object task_0 + state digits)
    static const int32_t lang_ids[] = { 2,      7071,   235292, 4788,   908,    573,    2656,   14581,  578,    2040,
                                        665,    611,    573,    8811,   1173,   3040,   235292, 235248, 235274, 235321,
                                        235310, 235248, 235315, 235321, 235248, 235276, 235248, 235276, 235248, 235274,
                                        235308, 235315, 235248, 235274, 235324, 235324, 235248, 235276, 235248, 235284,
                                        235308, 235308, 235289, 108,    4022,   235292, 235248 };
    const int            lang_n     = n_lang <= 0 ? 47 : std::min(n_lang, 47);

    vla::Inputs in{};
    in.images        = views.data();
    in.n_images      = 2;
    in.lang_tokens   = lang_ids;
    in.n_lang        = lang_n;
    in.noise         = nullptr;
    in.timing_detail = vla::TimingDetail::PHASE;

    // warmup (first call builds the graph cache)
    vla::predict(m, in);
    vla::predict(m, in);

    double             t_tot = 0, t_inf = 0;
    std::vector<float> act;
    for (int it = 0; it < 3; ++it) {
        double t0       = now_ms();
        act             = vla::predict(m, in);
        double       t1 = now_ms();
        const auto & st = vla::last_stats(m);
        t_tot += t1 - t0;
        t_inf += st.ms_inference;
        printf("iter %d: wall=%.1f inf=%.1f vis=%.1f tot=%.1f a[0]=%g a[%zu]=%g\n", it, t1 - t0, st.ms_inference,
               st.ms_vision, st.ms_total, act[0], act.size() - 1, act.back());
    }
    printf("avg: wall=%.2f ms  inf=%.2f ms  n_act=%zu\n", t_tot / 3, t_inf / 3, act.size());

    FILE * f = fopen(out, "w");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", out);
        return 1;
    }
    for (float v : act) {
        fprintf(f, "%.9e\n", v);
    }
    fclose(f);
    printf("wrote %zu floats to %s\n", act.size(), out);

    vla::model_free(m);
    return 0;
}
