/**************************************************************************/
/* ultralytics_model.hpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file ultralytics_model.hpp
 * @brief Declares the translation of an Ultralytics model description into a model file.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>
#include <SushiAI/vision/importers/ultralytics_tensors.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the input side a model takes when its checkpoint states none. */
        inline constexpr std::int64_t K_ULTRALYTICS_IMAGE_SIDE = 640;

        /** @brief Holds a translated description: the model document and the tensor rules. */
        struct UltralyticsTranslation
        {
            /** @brief Holds the model document: rows, outputs and task section. */
            Config::Value document;

            /** @brief Holds how the checkpoint's tensor names become the model's record names. */
            TensorRules tensor_rules;
        };

        /**
         * @brief Translates @p architecture, the "yaml" of an Ultralytics or YOLOv5 model.
         * @param image_side The side of the square input, as the checkpoint trained it.
         * @param modules The pickled module classes; a module none names is built as
         *                Ultralytics' own source builds it.
         * @return Unsupported listing every module no translator reads, with its entries.
         */
        [[nodiscard]] Result<UltralyticsTranslation> describe_ultralytics_model(
            const Config::Value& architecture, std::int64_t image_side,
            SushiRuntime::span<const UltralyticsModuleClass> modules,
            std::string_view origin) noexcept;

        /** @brief Returns the document of describe_ultralytics_model parsed by parse_model. */
        [[nodiscard]] Result<Config::ModelSpec> translate_ultralytics_model(
            const Config::Value& architecture, std::int64_t image_side,
            SushiRuntime::span<const UltralyticsModuleClass> modules,
            std::string_view origin) noexcept;
    } // namespace Vision::Importers
} // namespace SushiAI
