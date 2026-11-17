/**************************************************************************/
/* torch_weight_source.hpp                                                */
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
 * @file torch_weight_source.hpp
 * @brief Declares the weight source of a PyTorch file: the tensors of its state dictionary.
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
         * @brief Returns whether @p bytes begin like a zip archive, the container of `.pth`,
         *        `.pt` and `.pth.tar`.
         */
        [[nodiscard]] bool is_torch_file(SushiRuntime::span<const std::byte> bytes) noexcept;

        /**
         * @brief Opens the state dictionary of a PyTorch file kept as @p bytes.
         * @param options Its state_key names the dictionary; empty takes the file's root.
         * @return The source holding one tensor per entry, in pickle order. InvalidFile when
         *         the key is absent, an entry is no tensor, or the archive is no PyTorch file;
         *         Unsupported for a byte order or a storage the reader lacks.
         */
        [[nodiscard]] Result<std::unique_ptr<WeightSource>> open_torch_source(
            std::vector<std::byte> bytes, const WeightSourceOptions& options) noexcept;

        /** @brief Holds the row of the PyTorch container. */
        inline constexpr WeightSourceEntry K_TORCH_WEIGHT_SOURCE_ROW{"torch", &is_torch_file,
                                                                     &open_torch_source};

    } // namespace IO
} // namespace SushiAI
