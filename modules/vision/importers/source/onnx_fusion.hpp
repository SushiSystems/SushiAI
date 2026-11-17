/**************************************************************************/
/* onnx_fusion.hpp                                                        */
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
 * @file onnx_fusion.hpp
 * @brief Declares the plan that finds the SiLU pairs of an ONNX graph before it is translated.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <SushiAI/interchange/onnx_graph.hpp>
#include "onnx_translation.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            /** @brief Holds what each node of a graph is for the translation, by node position. */
            struct OnnxNodePlan
            {
                /** @brief Holds the role of each node. */
                std::vector<OnnxNodeRole> roles;

                /** @brief Holds, for a Silu node, the tensor its Sigmoid reads; else empty. */
                std::vector<std::string> gate_sources;
            };

            /**
             * @brief Finds each Sigmoid read only by a Mul of the Sigmoid and its own input, the
             *        pair an export writes for SiLU: the Sigmoid becomes a Gate and the Mul a
             *        Silu. Any other Sigmoid stays a plain sigmoid row.
             * @param nodes The nodes in file order, as fold_constants returns them.
             * @param outputs The graph outputs, whose names count as readers.
             */
            [[nodiscard]] OnnxNodePlan plan_onnx_nodes(
                const std::vector<Interchange::OnnxNode>& nodes,
                const std::vector<Interchange::OnnxValueInfo>& outputs);
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
