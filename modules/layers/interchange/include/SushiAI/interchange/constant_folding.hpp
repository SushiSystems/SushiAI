/**************************************************************************/
/* constant_folding.hpp                                                   */
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
 * @file constant_folding.hpp
 * @brief Declares the evaluation of the shape arithmetic an ONNX export leaves in its nodes.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        /** @brief Names which vector of a ConstantValue holds its elements. */
        enum class ConstantKind : std::uint8_t
        {
            Integer, /**< The elements are in integers; int32 and int64 both land here. */
            Real,    /**< The elements are in reals; float32. */
        };

        /** @brief Holds a small constant tensor the folder computed. */
        struct ConstantValue
        {
            ConstantKind kind = ConstantKind::Integer; /**< Which vector holds the elements. */
            std::vector<std::int64_t> dimensions;      /**< The extents; empty for a scalar. */
            std::vector<std::int64_t> integers;        /**< The elements of an Integer value. */
            std::vector<float> reals;                  /**< The elements of a Real value. */
        };

        /** @brief Names a computed constant by the tensor it replaces. */
        struct NamedConstant
        {
            std::string name;     /**< The name a kept node reads it under. */
            ConstantValue value;  /**< The computed tensor. */
        };

        /** @brief Holds a graph's nodes after the folding, and the constants they now read. */
        struct FoldedGraph
        {
            /** @brief Holds the nodes that were not folded, in their original order. */
            std::vector<OnnxNode> nodes;

            /** @brief Holds the computed tensors that kept nodes read, by name. */
            std::vector<NamedConstant> constants;

            /** @brief Holds how many nodes the folding removed. */
            std::size_t folded_count = 0;
        };

        /**
         * @brief Folds Shape, Gather, Unsqueeze, Concat, Cast and Constant nodes that feed a
         *        Reshape, a Resize, an Expand or each other.
         *
         * Inputs must be constants: small int32, int64 or float32 initializers, or the Shape
         * of a graph input with stated extents. A node whose result a graph output or another
         * operator reads is kept.
         */
        [[nodiscard]] Result<FoldedGraph> fold_constants(const OnnxGraph& graph);
    } // namespace Interchange
} // namespace SushiAI
