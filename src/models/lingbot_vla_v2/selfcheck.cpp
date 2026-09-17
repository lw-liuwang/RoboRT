// Alignment driver for the lingbot_vla_v2 engine (numerical check against
// the HuggingFace reference).
//
// Reads the deterministic inputs produced from the reference export
// (export_lingbot_vla_v2_reference.py → run_lingbot_vla_v2_align.py):
//   indir/img_emb.npy      f32 [192, 10240]  merger+deepstack clip output
//                                           (bypasses the vision tower via
//                                           vla::Inputs::precomputed_img_emb)
//   indir/lang_tokens.npy  i32 [n_lang]
//   indir/state.npy        f32 [14]          raw robot proprioception
//   indir/noise.npy        f32 [50, 55]      initial denoiser noise
//
// One predict() run; with VLA_LINGBOT_V2_DUMP_DIR set (this binary points it
// at outdir before model_load) the engine also writes every tapped
// intermediate as .npy for key-by-key comparison, and the raw action chunk is
// saved as outdir/final.actions_raw.npy.
//
// usage: robort-lingbot-v2-selfcheck mmproj.gguf lingbot.gguf indir outdir
#include "model.h"
#include "lingbot_vla_v2_npy.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s mmproj.gguf lingbot.gguf indir outdir\n", argv[0]);
        return 1;
    }
    const char * mmproj = argv[1];
    const char * ckpt   = argv[2];
    const char * indir  = argv[3];
    const char * outdir = argv[4];

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outdir, ec);
    // The engine reads this env at model_load time.
    setenv("VLA_LINGBOT_V2_DUMP_DIR", outdir, 1);

    lingbot_vla_v2::NpyArray img, lang, state, noise;
    if (!lingbot_vla_v2::read_npy(std::string(indir) + "/img_emb.npy", img) ||
        !lingbot_vla_v2::read_npy(std::string(indir) + "/lang_tokens.npy", lang) ||
        !lingbot_vla_v2::read_npy(std::string(indir) + "/state.npy", state) ||
        !lingbot_vla_v2::read_npy(std::string(indir) + "/noise.npy", noise)) {
        fprintf(stderr, "cannot read input .npy files from %s\n", indir);
        return 1;
    }
    const std::vector<float>    img_v  = img.as_f32();
    const std::vector<int32_t>  lang_v = lang.as_i32();
    const std::vector<float>    st_v   = state.as_f32();
    const std::vector<float>    nz_v   = noise.as_f32();

    vla::Model * m = vla::model_load(mmproj, ckpt, "");
    if (!m) {
        fprintf(stderr, "model_load failed\n");
        return 1;
    }
    const vla::Config & cfg = vla::model_config(m);

    if (img_v.size() != static_cast<size_t>(3) * 64 * cfg.hidden * 4) {
        fprintf(stderr, "img_emb size %zu != %lld\n", img_v.size(),
                static_cast<long long>(3 * 64 * cfg.hidden * 4));
        return 1;
    }
    if (lang_v.empty() || st_v.size() != static_cast<size_t>(cfg.real_state_dim)) {
        fprintf(stderr, "lang/state size mismatch (%zu lang, %zu state)\n", lang_v.size(), st_v.size());
        return 1;
    }
    if (nz_v.size() != static_cast<size_t>(cfg.max_action_dim) * cfg.n_suffix) {
        fprintf(stderr, "noise size %zu != %lld\n", nz_v.size(),
                static_cast<long long>(cfg.max_action_dim * cfg.n_suffix));
        return 1;
    }

    vla::Inputs in{};
    in.precomputed_img_emb = img_v.data();
    in.n_img_views         = 3;
    in.lang_tokens         = lang_v.data();
    in.n_lang              = static_cast<int>(lang_v.size());
    in.state               = st_v.data();
    in.noise               = nz_v.data();
    in.timing_detail       = vla::TimingDetail::PHASE;

    const std::vector<float> act = vla::predict(m, in);
    const auto &              st  = vla::last_stats(m);
    printf("predict: vis=%.1f inf=%.1f tot=%.1f ms  n_act=%zu\n", st.ms_vision, st.ms_inference, st.ms_total,
           act.size());
    if (act.size() != static_cast<size_t>(cfg.n_suffix) * cfg.real_action_dim) {
        fprintf(stderr, "action size %zu != %lld\n", act.size(),
                static_cast<long long>(cfg.n_suffix * cfg.real_action_dim));
        return 1;
    }

    // Raw action chunk [n_suffix, real_action_dim] (row-major, matches the
    // reference final.action_* ordering).
    if (!lingbot_vla_v2::write_npy_f32(std::string(outdir) + "/final.actions_raw.npy",
                                       { cfg.n_suffix, cfg.real_action_dim }, act.data())) {
        fprintf(stderr, "cannot write final.actions_raw.npy\n");
        return 1;
    }
    printf("wrote %zu floats to %s/final.actions_raw.npy\n", act.size(), outdir);

    vla::model_free(m);
    return 0;
}
