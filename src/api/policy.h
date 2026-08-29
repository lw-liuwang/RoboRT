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
 * @file policy.h
 * @brief Public C++ API of the RoboRT robot-brain inference engine.
 *
 * The `robo::` namespace is the forward-facing API for the engine.  It is a
 * thin type-alias layer over the existing `vla::` implementation so that new
 * callers and new architecture implementations can work entirely in `robo::`
 * while the legacy `vla::` symbols remain valid for backward compatibility.
 *
 * Architecture authors should implement `robo::Policy` (= `vla::Model`) and
 * register factories in `src/api/policy.cpp` / `src/models/architectures.cmake`.
 *
 * Mapping:
 *   robo::Policy       = vla::Model
 *   robo::PolicyConfig = vla::Config
 *   robo::PolicyInput  = vla::Inputs
 *   robo::PolicyStats  = vla::Stats
 *   robo::PreparedStep = vla::PreparedInput
 *
 * Free functions `robo::policy_load`, `robo::policy_free`, `robo::step`,
 * `robo::prepare`, `robo::compute`, `robo::last_stats` mirror the `vla::`
 * equivalents.
 */

#pragma once

#include "model.h"  // provides vla:: types that robo:: aliases

namespace robo {

// ---------------------------------------------------------------------------
// Type aliases — robo:: names map 1-to-1 onto the existing vla:: structs.
// The aliases keep header size minimal while giving new code a clean namespace.
// ---------------------------------------------------------------------------

/// Resolved hyper-parameters of a loaded policy (see vla::Config).
using PolicyConfig = vla::Config;

/// Lightweight view over a single image in caller-owned memory.
using ImageView = vla::ImageView;

/// Pixel format of an ImageView.
using PixelFormat = vla::PixelFormat;

/// Granularity of timing data collected during a step.
using TimingDetail = vla::TimingDetail;

/// Per-phase timing from the most recent step.
using PolicyStats = vla::Stats;

/// Inputs for one robo::step call (images + language + state + optional noise).
using PolicyInput = vla::Inputs;

/// Host-side prepared inputs for the two-phase async pipeline.
using PreparedStep = vla::PreparedInput;

/// Model-specific input extension; cast with dynamic_cast in the implementation.
using PolicyInputExtension = vla::ModelInputExtension;

/// Abstract handle for a loaded robot-brain policy.
/// Architectures implement this interface (= vla::Model).
using Policy = vla::Model;

// ---------------------------------------------------------------------------
// Free-function API — robo:: wrappers delegating to vla:: implementations.
// ---------------------------------------------------------------------------

/**
 * @brief Load a policy, dispatching to the architecture detected from the
 *        checkpoint's GGUF metadata.
 *
 * @param mmproj_path  Vision-tower GGUF path; empty for single-file architectures.
 * @param ckpt_path    Main checkpoint GGUF path.
 * @param config_path  Optional JSON override; empty = use bundled config.
 * @return Owning handle.  Free with policy_free().
 */
inline Policy * policy_load(const std::string & mmproj_path,
                             const std::string & ckpt_path,
                             const std::string & config_path = "") {
    return vla::model_load(mmproj_path, ckpt_path, config_path);
}

/// Release a policy handle returned by policy_load().
inline void policy_free(Policy * p) {
    vla::model_free(p);
}

/// Resolved configuration of a loaded policy.
inline const PolicyConfig & policy_config(const Policy * p) {
    return vla::model_config(p);
}

/**
 * @brief Run one forward step: vision → language → denoise → action chunk.
 * @return Action chunk [num_steps * real_action_dim], row-major, de-normalised.
 */
inline std::vector<float> step(Policy * p, const PolicyInput & in) {
    return vla::predict(p, in);
}

/**
 * @brief Phase 1 of the two-phase async pipeline (vision + language embedding).
 * @return PreparedStep to pass to compute().  ok=false on error.
 */
inline PreparedStep prepare(Policy * p, const PolicyInput & in) {
    return vla::prepare(p, in);
}

/**
 * @brief Phase 2 of the two-phase async pipeline (graph compute + de-norm).
 * @return Action chunk identical to step().  Empty on error.
 */
inline std::vector<float> compute(Policy * p, const PreparedStep & ps) {
    return vla::compute(p, ps);
}

/// Stats from the most recent step() / compute() call.
inline const PolicyStats & last_stats(const Policy * p) {
    return vla::last_stats(p);
}

}  // namespace robo
