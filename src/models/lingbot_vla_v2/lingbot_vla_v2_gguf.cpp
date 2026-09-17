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

#include "lingbot_vla_v2_gguf.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace lingbot_vla_v2 {

std::vector<uint8_t> GgufReader::read_convert(const char * name, ggml_type target) {
    const ggml_tensor * t = meta(name);
    if (!t) {
        std::fprintf(stderr, "lingbot_vla_v2: missing tensor %s\n", name);
        return {};
    }
    const int64_t n = ggml_nelements(t);

    std::vector<float> f32(n);
    if (t->type == GGML_TYPE_F32) {
        if (!read_raw(name, f32.data())) {
            return {};
        }
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> tmp(n);
        if (!read_raw(name, tmp.data())) {
            return {};
        }
        ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
    } else {
        std::fprintf(stderr, "lingbot_vla_v2: tensor %s has unsupported type %d\n", name, static_cast<int>(t->type));
        return {};
    }

    if (target == GGML_TYPE_F32) {
        std::vector<uint8_t> out(static_cast<size_t>(n) * sizeof(float));
        std::memcpy(out.data(), f32.data(), out.size());
        return out;
    }
    if (target == GGML_TYPE_BF16) {
        std::vector<uint8_t> out(static_cast<size_t>(n) * sizeof(ggml_bf16_t));
        ggml_fp32_to_bf16_row(f32.data(), reinterpret_cast<ggml_bf16_t *>(out.data()), n);
        return out;
    }
    if (ggml_is_quantized(target)) {
        const int64_t n_per_row = t->ne[0];
        const int64_t nrows     = n / n_per_row;
        const int64_t blck      = ggml_blck_size(target);
        if (n_per_row % blck != 0) {
            std::fprintf(stderr, "lingbot_vla_v2: cannot quantize %s to %s: ne0=%lld not divisible by block=%lld\n", name,
                         ggml_type_name(target), static_cast<long long>(n_per_row), static_cast<long long>(blck));
            return {};
        }
        const size_t qbytes =
            static_cast<size_t>(nrows) * static_cast<size_t>(n_per_row / blck) * ggml_type_size(target);
        std::vector<uint8_t> out(qbytes);
        const size_t written = ggml_quantize_chunk(target, f32.data(), out.data(), 0, nrows, n_per_row, nullptr);
        if (written != qbytes) {
            std::fprintf(stderr, "lingbot_vla_v2: quantized byte mismatch for %s (%zu vs %zu)\n", name, written, qbytes);
            return {};
        }
        return out;
    }
    std::fprintf(stderr, "lingbot_vla_v2: unsupported resident type %d for %s\n", static_cast<int>(target), name);
    return {};
}

bool GgufReader::fetch_rows_f32(const char *                 name,
                                const std::vector<int32_t> & row_ids,
                                float *                      dst,
                                int64_t                      cols) const {
    const ggml_tensor * t = meta(name);
    if (!t) {
        std::fprintf(stderr, "lingbot_vla_v2: missing tensor %s\n", name);
        return false;
    }
    if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
        std::fprintf(stderr, "lingbot_vla_v2: %s shape unfit for row-fetch\n", name);
        return false;
    }
    const int64_t rows = t->ne[1];
    const int64_t id   = gguf_find_tensor(gctx, name);
    const size_t  base = data_off + gguf_get_tensor_offset(gctx, id);
    const size_t  elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
    const size_t  rb   = static_cast<size_t>(cols) * elsz;

    std::vector<uint8_t> row(rb);
    for (size_t k = 0; k < row_ids.size(); ++k) {
        const int32_t r = row_ids[k];
        if (r < 0 || r >= rows) {
            std::fprintf(stderr, "lingbot_vla_v2: row %d out of range for %s\n", r, name);
            return false;
        }
        if (std::fseek(fp, static_cast<long>(base + static_cast<size_t>(r) * rb), SEEK_SET) != 0) {
            return false;
        }
        if (std::fread(row.data(), 1, rb, fp) != rb) {
            return false;
        }
        if (elsz == 4) {
            std::memcpy(dst + k * cols, row.data(), rb);
        } else {
            ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
    }
    return true;
}

}  // namespace lingbot_vla_v2
