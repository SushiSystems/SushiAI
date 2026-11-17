/**************************************************************************/
/* host_row_values.hpp                                                    */
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
 * @file host_row_values.hpp
 * @brief Declares evaluate_on_host, which runs a traced graph on the host and reads one value.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/host_table.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Names a leaf of a traced graph and the elements it holds in a host run. */
        struct HostLeaf
        {
            Graph::ValueID value;         /**< The leaf. */
            std::vector<double> elements; /**< Its elements, in row-major order. */
        };

        /**
         * @brief Evaluates @p graph on the host, in double, and returns the elements of @p result.
         * @param leaves The elements of every leaf the graph reads that is not a host table.
         * @param tables The host tables the trace declared; each one is written as its own leaf.
         * @return The elements of @p result, or the failure of a write, of the run or of the read.
         */
        [[nodiscard]] inline Result<std::vector<double>> evaluate_on_host(
            const Graph::Graph& graph, std::vector<HostLeaf> leaves,
            const NN::HostTableList& tables, Graph::ValueID result)
        {
            Graph::HostEvaluator evaluator(graph);
            for (HostLeaf& leaf : leaves)
                SA_TRY(evaluator.write(leaf.value, std::move(leaf.elements)));

            for (const NN::HostTable& table : tables.tables())
                SA_TRY(evaluator.write(table.leaf, table.values));

            SA_TRY(evaluator.run());
            return evaluator.read(result);
        }

        /**
         * @brief Evaluates @p graph on the host, in double, for a trace that declared no table.
         * @param leaves The elements of every leaf the graph reads.
         * @return The elements of @p result, or the failure of a write, of the run or of the read.
         */
        [[nodiscard]] inline Result<std::vector<double>> evaluate_on_host(
            const Graph::Graph& graph, std::vector<HostLeaf> leaves, Graph::ValueID result)
        {
            return evaluate_on_host(graph, std::move(leaves), NN::HostTableList(), result);
        }
    } // namespace Test
} // namespace SushiAI
