/**************************************************************************/
/* objective_context.cpp                                                  */
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
 * @file objective_context.cpp
 * @brief Implements the listing of the extents a traced model published.
 * @author Mustafa Garip
 */

#include <SushiAI/train/objective_context.hpp>

#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<std::vector<OutputShape>> list_output_shapes(const Graph::Graph& graph,
                                                            const NN::ModelOutputs& outputs) noexcept
        {
            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<OutputShape> listed;
                listed.reserve(outputs.all().size());
                for (const NN::ModelOutputs::Entry& output : outputs.all())
                    listed.push_back(OutputShape{output.first, graph.value(output.second).shape});

                return listed;
            });
        }

        const OutputShape* find_output_shape(const std::vector<OutputShape>& outputs,
                                             std::string_view name) noexcept
        {
            for (const OutputShape& output : outputs)
            {
                if (output.name == name)
                    return &output;
            }

            return nullptr;
        }

    } // namespace Train
} // namespace SushiAI
