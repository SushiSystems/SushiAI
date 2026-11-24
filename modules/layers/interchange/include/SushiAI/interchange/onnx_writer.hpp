/**************************************************************************/
/* onnx_writer.hpp                                                        */
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
 * @file onnx_writer.hpp
 * @brief Declares the writer of an ONNX ModelProto, the inverse of read_onnx_graph.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        /**
         * @brief Writes @p graph as the bytes of an ONNX ModelProto.
         * Model fields come from the graph's ir_version, producer and opsets; tensors go as
         * raw_data, value infos with type and shape, attribute subgraphs as GraphProto.
         * @return The file; InvalidDescription naming a tensor whose bytes do not fill its
         *         extents, a String tensor or an attribute of Undefined type.
         */
        [[nodiscard]] Result<std::vector<std::byte>> write_onnx_graph(
            const OnnxGraph& graph) noexcept;
    } // namespace Interchange
} // namespace SushiAI
