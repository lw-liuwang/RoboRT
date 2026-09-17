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
 * @file model.cpp
 * @brief Implementation of the public API: architecture detection +
 *        dispatch in @ref vla::model_load, plus the free-function wrappers
 *        that delegate through the @ref vla::Model virtual interface.
 *
 * New code should use the @ref robo:: aliases declared in policy.h.  The
 * vla:: symbols remain for backward compatibility.
 *
 * Adding a new architecture requires touching exactly three places, all marked
 * with an "ARCH:" comment below, plus one row in the build registry
 * (@c src/models/architectures.cmake, which also selects what gets compiled
 * via the ROBORT_MODELS option):
 *   1. declare the @c *_create factory (guarded by its @c VLA_HAS_* macro),
 *   2. extend @ref Arch and the detection fingerprint in @ref detect_arch_gguf,
 *   3. add the dispatch case in @ref vla::model_load.
 * The factory itself lives in the model's own TU under @c src/models/<arch>/.
 */

#include "model.h"

#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace vla {

// ---------------------------------------------------------------------------
// ARCH: per-architecture factory declarations.  Implemented in
// src/models/<arch>/; compiled only when selected via ROBORT_MODELS
// (see src/models/architectures.cmake).  Keep in sync with the enum and
// the dispatch switch below.
// ---------------------------------------------------------------------------
#ifdef VLA_HAS_PI05
std::unique_ptr<Model> pi05_create(const std::string & mmproj_path,
                                   const std::string & ckpt_path,
                                   const std::string & config_path);
#endif
#ifdef VLA_HAS_HY_VLA
std::unique_ptr<Model> hy_vla_create(const std::string & mmproj_path,
                                     const std::string & ckpt_path,
                                     const std::string & config_path);
#endif
// ARCH: FasterWAM factory declaration.
#ifdef VLA_HAS_FASTERWAM
std::unique_ptr<Model> fasterwam_create(const std::string & mmproj_path,
                                        const std::string & ckpt_path,
                                        const std::string & config_path);
#endif
// ARCH: LingBot-VLA-v2 factory declaration.
#ifdef VLA_HAS_LINGBOT_VLA_V2
std::unique_ptr<Model> lingbot_vla_v2_create(const std::string & mmproj_path,
                                             const std::string & ckpt_path,
                                             const std::string & config_path);
#endif

namespace {

enum class Arch {
    PI05,       ///< Physical Intelligence pi0.5 policy (mmproj + ckpt).
    HY_VLA,     ///< Tencent Hy-Embodied-0.5-VLA dual-tower flow policy (single GGUF).
    FASTERWAM,  ///< HUST FasterWAM World Action Model (Wan2.2 + SparseActionDiT, single GGUF).
    LINGBOT_VLA_V2,  ///< Robbyant LingBot-VLA-v2-6B flow-matching VLA (mmproj + ckpt).
};

bool ends_with_gguf(const std::string & p) {
    if (p.size() < 5) {
        return false;
    }
    return std::strcmp(p.c_str() + p.size() - 5, ".gguf") == 0;
}

bool detect_arch_gguf(const std::string & path, Arch * out) {
    gguf_init_params p{};
    p.no_alloc          = true;
    p.ctx               = nullptr;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), p);
    if (!gctx) {
        return false;
    }

    auto try_str = [&](const char * key, std::string & val) -> bool {
        const int64_t kid = gguf_find_key(gctx, key);
        if (kid < 0) {
            return false;
        }
        const char * s = gguf_get_val_str(gctx, kid);
        if (!s) {
            return false;
        }
        val = s;
        return true;
    };

    // ARCH: detection fingerprint.  Checkpoints declare their architecture
    // under "general.architecture" (llama.cpp convention) or, for
    // project-specific converters, "<arch>.architecture".  Recognise every
    // registered arch string here.
    bool        ok = false;
    std::string arch_str;
    if (try_str("general.architecture", arch_str) || try_str("pi05.architecture", arch_str) ||
        try_str("hy_vla.architecture", arch_str) || try_str("fasterwam.architecture", arch_str) ||
        try_str("lingbot_vla_v2.architecture", arch_str)) {
        if (arch_str == "pi05") {
            *out = Arch::PI05;
            ok   = true;
        } else if (arch_str == "hy_vla") {
            *out = Arch::HY_VLA;
            ok   = true;
        } else if (arch_str == "fasterwam") {
            *out = Arch::FASTERWAM;
            ok   = true;
        } else if (arch_str == "lingbot_vla_v2") {
            *out = Arch::LINGBOT_VLA_V2;
            ok   = true;
        }
    }

    gguf_free(gctx);
    return ok;
}

bool detect_arch_from_ckpt(const std::string & ckpt_path, Arch * out) {
    if (!out) {
        return false;
    }
    if (ends_with_gguf(ckpt_path)) {
        return detect_arch_gguf(ckpt_path, out);
    }
    return false;
}

}  // namespace

Model * model_load(const std::string & mmproj_path, const std::string & ckpt_path, const std::string & config_path) {
    Arch arch;
    if (!detect_arch_from_ckpt(ckpt_path, &arch)) {
        std::fprintf(stderr,
                     "robort: cannot detect architecture from %s "
                     "(unrecognised GGUF KV; expected pi05/hy_vla/fasterwam .architecture)\n",
                     ckpt_path.c_str());
        return nullptr;
    }

    std::unique_ptr<Model> impl;
    switch (arch) {
        // ARCH: dispatch.  Each case calls the factory when the architecture
        // is compiled in, and reports a clear error otherwise.
        case Arch::PI05:
#ifdef VLA_HAS_PI05
            std::printf("robort: arch = pi05\n");
            impl = pi05_create(mmproj_path, ckpt_path, config_path);
#else
            std::fprintf(stderr,
                         "robort: pi05 architecture not built "
                         "(reconfigure with ROBORT_MODELS=\"pi05 hy_vla\")\n");
#endif
            break;
        case Arch::HY_VLA:
#ifdef VLA_HAS_HY_VLA
            std::printf("robort: arch = hy_vla\n");
            impl = hy_vla_create(mmproj_path, ckpt_path, config_path);
#else
            std::fprintf(stderr,
                         "robort: hy_vla architecture not built "
                         "(reconfigure with ROBORT_MODELS=\"pi05 hy_vla\")\n");
#endif
            break;
        case Arch::FASTERWAM:
#ifdef VLA_HAS_FASTERWAM
            std::printf("robort: arch = fasterwam\n");
            impl = fasterwam_create(mmproj_path, ckpt_path, config_path);
#else
            std::fprintf(stderr,
                         "robort: fasterwam architecture not built "
                         "(reconfigure with ROBORT_MODELS=\"pi05 hy_vla fasterwam\")\n");
#endif
            break;
        case Arch::LINGBOT_VLA_V2:
#ifdef VLA_HAS_LINGBOT_VLA_V2
            std::printf("robort: arch = lingbot_vla_v2\n");
            impl = lingbot_vla_v2_create(mmproj_path, ckpt_path, config_path);
#else
            std::fprintf(stderr,
                         "robort: lingbot_vla_v2 architecture not built "
                         "(reconfigure with ROBORT_MODELS=\"pi05 hy_vla lingbot_vla_v2\")\n");
#endif
            break;
    }
    return impl.release();
}

void model_free(Model * m) {
    delete m;
}

const Config & model_config(const Model * m) {
    return m->config();
}

const Stats & last_stats(const Model * m) {
    return m->last_stats();
}

std::vector<float> predict(Model * m, const Inputs & in) {
    return m->predict(in);
}

PreparedInput prepare(Model * m, const Inputs & in) {
    return m->prepare(in);
}

std::vector<float> compute(Model * m, const PreparedInput & p) {
    return m->compute(p);
}

}  // namespace vla
