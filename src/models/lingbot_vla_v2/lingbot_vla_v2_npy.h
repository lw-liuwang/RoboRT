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
 * @file lingbot_vla_v2_npy.h
 * @brief Minimal NumPy .npy writer (v1.0, little-endian '<f4', C order).
 *
 * Only what the numerical-alignment dumps need: one 4-byte-float array per
 * file, written row-major.  The reference side (export_lingbot_vla_v2_
 * reference.py) saves .npz with '/'-separated keys; this writer uses the
 * same keys with '.' separators as file names (compare_lingbot_vla_v2_ref.py
 * normalises both).
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace lingbot_vla_v2 {

/// One .npy file read into memory: shape + dtype + raw little-endian data.
struct NpyArray {
    std::vector<int64_t> shape;
    char                 dtype = 0;  ///< 'f' f32, 'd' f64, 'i' i32, 'I' i64, 'u' u8.
    size_t               itemsize = 0;
    std::vector<uint8_t> data;

    int64_t count() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            n *= d;
        }
        return n;
    }
    std::vector<float> as_f32() const;
    std::vector<int32_t> as_i32() const;
};

/// Read a little-endian C-order .npy written by numpy or write_npy_f32().
inline bool read_npy(const std::string & path, NpyArray & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    bool ok = false;
    do {
        unsigned char magic[8];
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            break;
        }
        size_t hlen = 0;
        if (magic[6] == 1) {
            unsigned char h[2];
            if (std::fread(h, 1, 2, f) != 2) {
                break;
            }
            hlen = static_cast<size_t>(h[0]) | (static_cast<size_t>(h[1]) << 8);
        } else if (magic[6] == 2) {
            unsigned char h[4];
            if (std::fread(h, 1, 4, f) != 4) {
                break;
            }
            hlen = static_cast<size_t>(h[0]) | (static_cast<size_t>(h[1]) << 8) | (static_cast<size_t>(h[2]) << 16) |
                   (static_cast<size_t>(h[3]) << 24);
        } else {
            break;
        }
        std::string hdr(hlen, '\0');
        if (hlen == 0 || std::fread(&hdr[0], 1, hlen, f) != hlen) {
            break;
        }

        // crude dict parse: 'descr': '<f4', 'shape': (a, b),
        const size_t dp = hdr.find("'descr'");
        const size_t q1 = dp == std::string::npos ? std::string::npos : hdr.find('\'', dp + 7);
        const size_t q2 = q1 == std::string::npos ? std::string::npos : hdr.find('\'', q1 + 1);
        const size_t sp = hdr.find("'shape'");
        const size_t p1 = sp == std::string::npos ? std::string::npos : hdr.find('(', sp + 7);
        const size_t p2 = p1 == std::string::npos ? std::string::npos : hdr.find(')', p1);
        if (q2 == std::string::npos || p2 == std::string::npos) {
            break;
        }
        const std::string descr = hdr.substr(q1 + 1, q2 - q1 - 1);
        if (descr == "<f4" || descr == "f4") {
            out.dtype = 'f';
            out.itemsize = 4;
        } else if (descr == "<f8" || descr == "f8") {
            out.dtype = 'd';
            out.itemsize = 8;
        } else if (descr == "<i4" || descr == "i4") {
            out.dtype = 'i';
            out.itemsize = 4;
        } else if (descr == "<i8" || descr == "i8") {
            out.dtype = 'I';
            out.itemsize = 8;
        } else if (descr == "|u1" || descr == "u1") {
            out.dtype = 'u';
            out.itemsize = 1;
        } else {
            break;  // unsupported dtype (or big-endian)
        }

        out.shape.clear();
        std::string dims = hdr.substr(p1 + 1, p2 - p1 - 1);
        size_t           pos = 0;
        while (pos < dims.size()) {
            const size_t comma = dims.find(',', pos);
            const std::string tok = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!tok.empty()) {
                out.shape.push_back(std::atoll(tok.c_str()));
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
        if (out.shape.empty()) {
            break;
        }

        const int64_t n = out.count();
        out.data.resize(static_cast<size_t>(n) * out.itemsize);
        if (n > 0 && std::fread(out.data.data(), out.itemsize, static_cast<size_t>(n), f) !=
                        static_cast<size_t>(n)) {
            break;
        }
        ok = true;
    } while (false);
    std::fclose(f);
    return ok;
}

inline std::vector<float> NpyArray::as_f32() const {
    std::vector<float> v(static_cast<size_t>(count()));
    for (int64_t i = 0; i < count(); ++i) {
        const uint8_t * p = data.data() + static_cast<size_t>(i) * itemsize;
        if (dtype == 'f') {
            std::memcpy(&v[static_cast<size_t>(i)], p, 4);
        } else if (dtype == 'd') {
            double d = 0;
            std::memcpy(&d, p, 8);
            v[static_cast<size_t>(i)] = static_cast<float>(d);
        } else if (dtype == 'i') {
            int32_t x = 0;
            std::memcpy(&x, p, 4);
            v[static_cast<size_t>(i)] = static_cast<float>(x);
        } else if (dtype == 'I') {
            int64_t x = 0;
            std::memcpy(&x, p, 8);
            v[static_cast<size_t>(i)] = static_cast<float>(x);
        } else {
            v[static_cast<size_t>(i)] = static_cast<float>(p[0]);
        }
    }
    return v;
}

inline std::vector<int32_t> NpyArray::as_i32() const {
    std::vector<int32_t> v(static_cast<size_t>(count()));
    for (int64_t i = 0; i < count(); ++i) {
        const uint8_t * p = data.data() + static_cast<size_t>(i) * itemsize;
        if (dtype == 'i') {
            std::memcpy(&v[static_cast<size_t>(i)], p, 4);
        } else if (dtype == 'I') {
            int64_t x = 0;
            std::memcpy(&x, p, 8);
            v[static_cast<size_t>(i)] = static_cast<int32_t>(x);
        } else if (dtype == 'u') {
            v[static_cast<size_t>(i)] = p[0];
        } else {
            float x = 0;
            std::memcpy(&x, p, 4);
            v[static_cast<size_t>(i)] = static_cast<int32_t>(x);
        }
    }
    return v;
}

/// Write `count = prod(shape)` float32 values to `path` as a C-order .npy.
inline bool write_npy_f32(const std::string & path, const std::vector<int64_t> & shape, const float * data) {
    size_t count = 1;
    for (int64_t d : shape) {
        if (d < 0) {
            return false;
        }
        count *= static_cast<size_t>(d);
    }

    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            hdr += ", ";
        }
        hdr += std::to_string(shape[i]);
    }
    if (shape.size() == 1) {
        hdr += ",";
    }
    hdr += "), }";

    // Pad so that magic(6) + version(2) + hlen(2) + header (incl. '\n') is a
    // multiple of 64 bytes (numpy only requires the hlen field to match).
    const size_t base = 10;
    size_t       total = base + hdr.size() + 1;
    total               = ((total + 63) / 64) * 64;
    hdr.resize(total - base - 1, ' ');
    hdr.push_back('\n');

    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const unsigned char magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    const uint16_t       hlen16 = static_cast<uint16_t>(hdr.size());
    bool                 ok      = std::fwrite(magic, 1, 8, f) == 8 && std::fwrite(&hlen16, 2, 1, f) == 1 &&
                     std::fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size();
    if (ok && count > 0) {
        ok = std::fwrite(data, sizeof(float), count, f) == count;
    }
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

}  // namespace lingbot_vla_v2
