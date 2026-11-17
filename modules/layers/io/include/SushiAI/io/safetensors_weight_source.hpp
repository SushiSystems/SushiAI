/**************************************************************************/
/* safetensors_weight_source.hpp                                          */
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
 * @file safetensors_weight_source.hpp
 * @brief Declares the weight source of a safetensors file.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/weight_source.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Returns whether @p bytes open with the 8-byte little-endian length of a header
         *        that fits the file and starts with '{'.
         */
        [[nodiscard]] bool is_safetensors_file(SushiRuntime::span<const std::byte> bytes) noexcept;

        /**
         * @brief Opens a safetensors file.
         * @param bytes The whole file; the source keeps them.
         * @param options Its state_key is ignored: the file has no nesting.
         * @return The source holding the tensors in the order of their bytes, or the failure of
         *         SafetensorsReader::read.
         */
        [[nodiscard]] Result<std::unique_ptr<WeightSource>> open_safetensors_source(
            std::vector<std::byte> bytes, const WeightSourceOptions& options) noexcept;

        /** @brief Holds the row of the safetensors container. */
        inline constexpr WeightSourceEntry K_SAFETENSORS_WEIGHT_SOURCE_ROW{
            "safetensors", &is_safetensors_file, &open_safetensors_source};

    } // namespace IO
} // namespace SushiAI
