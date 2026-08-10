// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

/**
 * @file model_input_extension.h
 * @brief Extension interface for model-specific @ref Inputs fields.
 *
 * Consumers of the public @ref vla::predict API populate the generic
 * @ref vla::Inputs struct (images, language, state).  Models that require
 * additional typed inputs (e.g. video latents, action conditions, cache
 * controls) define a subclass of @ref ModelInputExtension and pass it
 * through @ref Inputs::model_specific.
 *
 * Each architecture defines its own extension type and the predict() call
 * casts @ref Inputs::model_specific back to the known type with
 * @c dynamic_cast.
 */

#pragma once

namespace vla {

/**
 * @brief Base class for model-specific input extensions.
 *
 * Every supported VLA architecture may define a subclass carrying fields
 * that do not belong in the common @ref Inputs API.  The predict()
 * implementation uses @c dynamic_cast to recover the concrete type.
 */
struct ModelInputExtension {
    virtual ~ModelInputExtension() = default;
};

}  // namespace vla
