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
 * @file pi05_utils.h
 * @brief Small math / encoding utilities shared across pi05 translation units.
 */

#pragma once

#include <cmath>
#include <vector>

namespace pi05 {

/// Sinusoidal time embedding used by the flow-matching schedule.
/// dim must be even; the embedding is [sin_0..sin_(dim/2-1), cos_0..cos_(dim/2-1)].
inline std::vector<float> sinusoidal_time_emb(double t, int64_t dim, double min_p, double max_p) {
    const int64_t      half = dim / 2;
    std::vector<float> out(dim);
    for (int64_t i = 0; i < half; ++i) {
        const double frac   = (half == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(half - 1);
        const double period = min_p * std::pow(max_p / min_p, frac);
        const double s      = (2.0 * M_PI / period) * t;
        out[i]              = static_cast<float>(std::sin(s));
        out[half + i]       = static_cast<float>(std::cos(s));
    }
    return out;
}

}  // namespace pi05
