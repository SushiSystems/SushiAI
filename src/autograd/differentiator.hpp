/**************************************************************************/
/* differentiator.hpp                                                    */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

// Differentiator's definition, private to the two translation units that need
// it: differentiator.cpp, which owns differentiate() and constructs one per run,
// and
// tape.cpp, whose GradientTape methods call back into it through the owner_
// pointer. It is deliberately not reachable from tape.hpp or rule.hpp - a rule
// sees only GradientTape, never this class - so this header lives in src/,
// outside the public include tree, included by exactly those two .cpp files.

#include <unordered_map>
#include <vector>
#include <SushiAI/autograd/tape.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @class Differentiator
         * @brief One differentiation run's bookkeeping.
         *
         * Holds the contributions per primal value, the memoised sums, and the
         * set of nodes a fusion has already absorbed. Lives only for the length
         * of a @c differentiate() call.
         */
        class Differentiator
        {
            public:
                Differentiator(Graph::Graph& graph, Graph::ValueId loss,
                               Graph::ValueId seed_factor)
                    : graph_(graph),
                      loss_(loss),
                      builder_(graph),
                      forward_node_count_(graph.node_count()),
                      skipped_(graph.node_count(), false),
                      tape_(Graph::GraphBuilder(graph), *this, seed_factor)
                {
                }

                /** @brief Runs the reverse sweep over the forward nodes. */
                void run();

                /** @brief The summed gradient of @p primal, materialising the sum on first read. */
                Graph::ValueId resolve(Graph::ValueId primal);

                /** @brief Records one gradient contribution to @p primal. */
                void contribute(Graph::ValueId primal, Graph::ValueId gradient);

                /** @brief Marks @p node as already differentiated by a fusion. */
                void skip(Graph::NodeId node);

                /** @brief Whether @p value is the scalar being differentiated. */
                bool is_root(Graph::ValueId value) const noexcept { return value == loss_; }

                /**
                 * @brief The node the tape currently points at.
                 *
                 * A snapshot, not a reference into the graph. Every rule emits
                 * while it is running, and emitting grows the node vector,
                 * which reallocates and would leave a rule holding a dangling
                 * reference to the very node it is differentiating. Copying one
                 * node per differentiated node is unmeasurable; the alternative
                 * is a use-after-free that only appears once a graph outgrows
                 * its initial capacity.
                 */
                const Graph::Node& current_node() const noexcept { return current_node_; }

                /** @brief The graph being extended. */
                Graph::Graph& graph() noexcept { return graph_; }

            private:
                Graph::Graph& graph_;
                Graph::ValueId loss_;
                Graph::GraphBuilder builder_;
                std::size_t forward_node_count_;
                std::vector<bool> skipped_;
                std::unordered_map<Graph::ValueId, std::vector<Graph::ValueId>> contributions_;
                std::unordered_map<Graph::ValueId, Graph::ValueId> resolved_;
                GradientTape tape_;
                Graph::Node current_node_{};
        };

    } // namespace Autograd
} // namespace SushiAI
