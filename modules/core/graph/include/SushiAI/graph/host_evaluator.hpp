/**************************************************************************/
/* host_evaluator.hpp                                                     */
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
 * @file host_evaluator.hpp
 * @brief Declares HostEvaluator, which computes the values of a Graph on host buffers.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Reports whether HostEvaluator has a rule for @p op.
         * @return True for LEAF and for each operation the evaluator computes.
         */
        [[nodiscard]] bool host_evaluates(OpID op) noexcept;

        /**
         * @class HostEvaluator
         * @brief Computes every value of a Graph in double, node by node, on host buffers.
         *
         * A leaf takes its elements from write(); run() then evaluates each node in insertion
         * order. Every value is held as double whatever its element type, so the result is the
         * host reference of the graph, not the rounding of a device run.
         */
        class HostEvaluator final
        {
            public:
                /**
                 * @brief Prepares one empty buffer per value of @p graph.
                 * @param graph The graph to evaluate; must outlive this evaluator and keep its
                 *        values and nodes while the evaluator reads them.
                 */
                explicit HostEvaluator(const Graph& graph) noexcept;

                HostEvaluator(const HostEvaluator&) = delete;
                HostEvaluator& operator=(const HostEvaluator&) = delete;
                HostEvaluator(HostEvaluator&&) = delete;
                HostEvaluator& operator=(HostEvaluator&&) = delete;
                ~HostEvaluator() = default;

                /**
                 * @brief Gives leaf @p value its elements, in row-major order.
                 * @return A success, or InvalidDescription when @p value is not a value of the
                 *         graph, a node other than LEAF writes it, or @p elements does not hold
                 *         its element count.
                 */
                [[nodiscard]] Status write(ValueID value, std::vector<double> elements) noexcept;

                /**
                 * @brief Evaluates every node of the graph in insertion order.
                 * @return A success, Unsupported naming the first operation host_evaluates()
                 *         refuses, or InvalidDescription for a node that reads a value holding
                 *         no elements or an index outside its table.
                 */
                [[nodiscard]] Status run() noexcept;

                /**
                 * @brief Returns the elements of @p value, in row-major order.
                 * @return The elements, or InvalidDescription when @p value is not a value of
                 *         the graph or holds no elements yet.
                 */
                [[nodiscard]] Result<std::vector<double>> read(ValueID value) const noexcept;

            private:
                /// @brief The graph whose values the buffers hold.
                const Graph& graph_;

                /// @brief Holds the elements of each value, indexed by its ValueID.
                std::vector<std::vector<double>> buffers_;

                /// @brief Holds whether each value has received its elements.
                std::vector<bool> written_;
        };
    } // namespace Graph
} // namespace SushiAI
