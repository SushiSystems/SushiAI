/**************************************************************************/
/* weight_import.hpp                                                      */
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
 * @file weight_import.hpp
 * @brief Declares the strict import of a weight source against the state records of a model.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Matches the tensors of @p source to @p records and reads them, permuting a
         *        convolution weight with copy_kernel_last and a depthwise one with
         *        copy_depthwise_kernel_last.
         * @param map The names to drop and the renames to apply to the rest.
         * @return One ImportedTensor per record. Incompatible, in one message, naming each
         *         missing, unmatched or misshapen tensor; Unsupported for another element type.
         */
        [[nodiscard]] Result<std::vector<ImportedTensor>> import_state_weights(
            SushiRuntime::span<const NN::StateEntry> records, const WeightSource& source,
            const NameMap& map) noexcept;

    } // namespace IO
} // namespace SushiAI
