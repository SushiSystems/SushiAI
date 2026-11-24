/**************************************************************************/
/* weight_export.hpp                                                      */
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
 * @file weight_export.hpp
 * @brief Declares the export of state records to foreign weight tensors, inverse of import.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <string>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Holds the tensors a weight export writes and what it could not write. */
        struct ExportedWeights
        {
            /** @brief Holds the tensors in record order, named and laid out for the file. */
            std::vector<ImportedTensor> tensors;

            /** @brief Lists what the file states that the export leaves out. */
            std::vector<std::string> notes;
        };

        /**
         * @brief Moves each record of @p records to the layout and the name of the file.
         * KernelLast [out,kh,kw,in] goes to [out,in,kh,kw], DepthwiseKernelLast [kh,kw,C] to
         * [C,1,kh,kw], Plain stays; a name takes back the `from` prefix of the first @p map
         * rename whose `to` it starts with. Drops and PyTorch's batch-norm counter are notes.
         * @return The tensors; Incompatible, in one message, for records with no entry or
         *         another shape; Unsupported for a kernel that is not float32.
         */
        [[nodiscard]] Result<ExportedWeights> export_state_weights(
            SushiRuntime::span<const ImportedTensor> records,
            SushiRuntime::span<const NN::StateEntry> entries, const NameMap& map) noexcept;

    } // namespace IO
} // namespace SushiAI
