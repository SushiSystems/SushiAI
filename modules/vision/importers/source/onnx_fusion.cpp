/**************************************************************************/
/* onnx_fusion.cpp                                                        */
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
 * @file onnx_fusion.cpp
 * @brief Defines the plan that finds the SiLU pairs of an ONNX graph.
 * @author Mustafa Garip
 */

#include "onnx_fusion.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include <SushiAI/interchange/onnx_graph.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Names the operator of the gate of a SiLU. */
                constexpr const char* K_GATE_OPERATOR = "Sigmoid";

                /** @brief Names the operator that multiplies the gate. */
                constexpr const char* K_PRODUCT_OPERATOR = "Mul";

                /** @brief Counts how many nodes and graph outputs read each tensor name. */
                std::unordered_map<std::string, std::size_t> count_readers(
                    const std::vector<Interchange::OnnxNode>& nodes,
                    const std::vector<Interchange::OnnxValueInfo>& outputs)
                {
                    std::unordered_map<std::string, std::size_t> readers;
                    for (const Interchange::OnnxNode& node : nodes)
                    {
                        for (const std::string& name : node.inputs)
                            ++readers[name];
                    }
                    for (const Interchange::OnnxValueInfo& output : outputs)
                        ++readers[output.name];
                    return readers;
                }

                /** @brief Returns whether @p product multiplies @p gate by @p source. */
                bool multiplies(const Interchange::OnnxNode& product, const std::string& gate,
                                const std::string& source) noexcept
                {
                    return product.op_type == K_PRODUCT_OPERATOR && product.inputs.size() == 2 &&
                           ((product.inputs[0] == source && product.inputs[1] == gate) ||
                            (product.inputs[0] == gate && product.inputs[1] == source));
                }
            } // namespace

            OnnxNodePlan plan_onnx_nodes(const std::vector<Interchange::OnnxNode>& nodes,
                                         const std::vector<Interchange::OnnxValueInfo>& outputs)
            {
                OnnxNodePlan plan;
                plan.roles.assign(nodes.size(), OnnxNodeRole::Plain);
                plan.gate_sources.assign(nodes.size(), std::string());
                const std::unordered_map<std::string, std::size_t> readers =
                    count_readers(nodes, outputs);

                for (std::size_t gate = 0; gate < nodes.size(); ++gate)
                {
                    const Interchange::OnnxNode& sigmoid = nodes[gate];
                    if (sigmoid.op_type != K_GATE_OPERATOR || sigmoid.inputs.size() != 1 ||
                        sigmoid.outputs.size() != 1)
                        continue;
                    const std::string& source = sigmoid.inputs[0];
                    const std::string& gated = sigmoid.outputs[0];
                    const auto read = readers.find(gated);
                    if (read == readers.end() || read->second != 1 || source == gated)
                        continue;

                    for (std::size_t at = gate + 1; at < nodes.size(); ++at)
                    {
                        if (plan.roles[at] != OnnxNodeRole::Plain ||
                            !multiplies(nodes[at], gated, source))
                            continue;
                        plan.roles[gate] = OnnxNodeRole::Gate;
                        plan.roles[at] = OnnxNodeRole::Silu;
                        plan.gate_sources[at] = source;
                        break;
                    }
                }
                return plan;
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
