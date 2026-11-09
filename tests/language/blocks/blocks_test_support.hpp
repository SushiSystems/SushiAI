/**************************************************************************/
/* blocks_test_support.hpp                                                */
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
 * @file blocks_test_support.hpp
 * @brief Declares the values and the state writes the blocks tests share for host evaluation.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Returns @p count eighths that cycle through -5/8 to 5/8, starting at @p offset.
         * @return Small dyadic values, so a sum or a product of a few of them is exact.
         */
        inline std::vector<double> eighths(int64_t count, int64_t offset)
        {
            std::vector<double> values;
            values.reserve(static_cast<std::size_t>(count));
            for (int64_t i = 0; i < count; ++i)
                values.push_back(static_cast<double>((i * 7 + offset) % 11 - 5) / 8.0);

            return values;
        }

        /**
         * @brief Writes eighths() into every parameter and buffer of @p module, in walk order,
         *        each from an offset one above the last.
         * @return A success, or the first failure HostEvaluator::write() returns.
         * @pre @p module was traced into @p graph.
         */
        template<typename Module>
        Status write_state(Graph::HostEvaluator& evaluator, const Graph::Graph& graph,
                           Module& module)
        {
            Status outcome;
            int64_t offset = 0;
            const auto write = [&](Graph::ValueID id, const Shape& shape)
            {
                const Status written =
                    evaluator.write(id, eighths(shape.num_elements(), offset++));
                if (outcome.is_ok())
                    outcome = written;
            };
            const auto on_parameter = [&](const std::string&, NN::Parameter& parameter)
            {
                write(parameter.id_in(graph), parameter.shape());
            };
            const auto on_buffer = [&](const std::string&, NN::Buffer& buffer)
            {
                write(buffer.id_in(graph), buffer.shape());
            };
            const NN::StateVisitor visitor(on_parameter, on_buffer);
            NN::walk_state(module, "", visitor);
            return outcome;
        }

        /** @brief Returns row @p row of @p width elements of the row-major @p values. */
        inline std::vector<double> row_of(const std::vector<double>& values, std::size_t row,
                                          std::size_t width)
        {
            return std::vector<double>(values.begin() + static_cast<std::ptrdiff_t>(row * width),
                                       values.begin()
                                           + static_cast<std::ptrdiff_t>((row + 1) * width));
        }
    } // namespace Test
} // namespace SushiAI
