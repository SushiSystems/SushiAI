/**************************************************************************/
/* built_rows.cpp                                                         */
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
 * @file built_rows.cpp
 * @brief Implements BuiltRows over the nodes of a model graph.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/built_rows.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/nn/model_graph.hpp>

namespace SushiAI
{
    namespace NN
    {
        BuiltRows::BuiltRows(const ModelGraph& graph) noexcept : graph_(&graph)
        {

        }

        std::size_t BuiltRows::size() const noexcept
        {
            return graph_ == nullptr ? 0 : graph_->size();
        }

        Node& BuiltRows::node(std::size_t index) const noexcept
        {
            SA_FATAL_IF(index >= size(), "BuiltRows::node was asked for row {}; {} rows are above",
                        index, size());

            return graph_->node(index);
        }

    } // namespace NN
} // namespace SushiAI
