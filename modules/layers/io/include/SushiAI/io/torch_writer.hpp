/**************************************************************************/
/* torch_writer.hpp                                                       */
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
 * @file torch_writer.hpp
 * @brief Declares the PyTorch state dictionary writer, read by TorchArchive and `torch.load`.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Writes @p tensors as a PyTorch state dictionary zip under @p folder.
         * Entries: `data.pkl`, a protocol 2 `OrderedDict` of `_rebuild_tensor_v2` over ids
         * `('storage', <type>, '<k>', 'cpu', <elements>)`; `data/<k>`, 64-byte aligned;
         * `version` `3`; `byteorder` `little`. A non-empty @p state_key nests the dictionary.
         * @return The file; Unsupported for a tensor not float32, float64 or half;
         *         InvalidDescription for bytes unlike a record, or a @p folder empty or with '/'.
         */
        [[nodiscard]] Result<std::vector<std::byte>> write_torch_state_dict(
            SushiRuntime::span<const ImportedTensor> tensors, std::string_view folder,
            std::string_view state_key) noexcept;
    } // namespace IO
} // namespace SushiAI
