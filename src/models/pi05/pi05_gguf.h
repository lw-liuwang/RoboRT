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
 * @file pi05_gguf.h
 * @brief Lightweight GGUF reader for pi0.5: metadata queries and on-demand
 *        tensor reads / row-level random access.
 *
 * Holds a parsed GGUF context and an open FILE* for raw tensor reads.
 * Non-copyable; close automatically on destruction.
 */

#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace pi05 {

struct GgufReader {
    gguf_context * gctx     = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE *         fp       = nullptr;
    size_t         data_off = 0;

    GgufReader()                               = default;
    GgufReader(const GgufReader &)             = delete;
    GgufReader & operator=(const GgufReader &) = delete;

    ~GgufReader() {
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

    bool open(const std::string & path) {
        gguf_init_params p{};
        p.no_alloc = true;
        p.ctx      = &meta_ctx;
        gctx       = gguf_init_from_file(path.c_str(), p);
        if (!gctx) {
            std::fprintf(stderr, "pi05: gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "pi05: fopen failed for %s\n", path.c_str());
            return false;
        }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }

    bool has_key(const char * k) const { return gguf_find_key(gctx, k) >= 0; }

    uint32_t u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }

    float f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }

    double f64(const char * k) const { return gguf_get_val_f64(gctx, gguf_find_key(gctx, k)); }

    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    /// Read raw bytes for tensor `name` into `buf` (caller-allocated, must be
    /// exactly gguf_get_tensor_size bytes).
    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) {
            std::fprintf(stderr, "pi05: missing tensor %s\n", name);
            return false;
        }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, static_cast<long>(off), SEEK_SET) != 0) {
            return false;
        }
        return std::fread(buf, 1, nb, fp) == nb;
    }

    /// Read tensor `name`, convert from its stored type (F32 or BF16) to
    /// `target`, optionally applying the Gemma-norm +1 bias.
    std::vector<uint8_t> read_convert(const char * name, ggml_type target, bool gemma_norm);

    /// Read specific rows of a 2-D F32/BF16 matrix tensor.
    /// `dst` must have space for `row_ids.size() * cols` floats.
    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids, float * dst, int64_t cols) const;
};

}  // namespace pi05
