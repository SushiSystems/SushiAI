/**************************************************************************/
/* tape.hpp                                                               */
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

// The only interface a backward rule sees.
//
// A rule does not name Differentiator: the dependency is inverted, so the
// traversal can change without touching a single rule, and a rule cannot reach
// into the traversal's bookkeeping even by accident. That was already true when
// this class lived in backward.hpp; giving it its own header is what makes it
// visible, and what lets a rules/*.cpp include the interface without also
// including differentiate()'s declaration.

#include <cstddef>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief The differentiation run's private bookkeeping; see src/autograd/differentiator.cpp. */
        class Differentiator;

        /**
         * @class GradientTape
         * @brief The state a backward rule reads and writes.
         *
         * One instance exists per @c differentiate() call and is re-pointed at
         * each node in turn. A rule sees the node it is differentiating, the
         * gradients already known for that node's results, and one way to
         * publish a gradient for anything upstream.
         */
        class GradientTape
        {
            public:
                /** @brief The graph being extended. Rules emit through builder(). */
                Graph::Graph& graph() const noexcept { return builder_.graph(); }

                /** @brief The emitter every rule uses; it shape-checks what a rule writes. */
                const Graph::GraphBuilder& builder() const noexcept { return builder_; }

                /** @brief The node currently being differentiated. */
                const Graph::Node& node() const;

                /** @brief The id of the node currently being differentiated. */
                Graph::NodeId node_id() const noexcept { return node_; }

                /**
                 * @brief The gradient of the node's @p index-th result.
                 *
                 * @param index Which result.
                 * @return The gradient's id, or @c Graph::NO_VALUE when that
                 *         result has no gradient - which includes the
                 *         differentiation root, whose gradient is the implicit
                 *         scalar one and has no tensor to name.
                 */
                Graph::ValueId output_grad(std::size_t index) const;

                /** @brief Whether the node's @p index-th result is the differentiation root. */
                bool output_is_root(std::size_t index) const;

                /**
                 * @brief Whether a gradient should be produced for the node's @p index-th operand.
                 * @param index Which operand.
                 * @return True when that operand is on a path from something that requires one.
                 */
                bool input_requires_grad(std::size_t index) const;

                /**
                 * @brief Registers @p gradient as a contribution to @p primal's gradient.
                 *
                 * Several rules may contribute to the same primal; they are
                 * summed later, in the order they arrived.
                 *
                 * @param primal The forward value whose gradient this is.
                 * @param gradient The contribution.
                 * @throws Error If @p primal's gradient has already been read,
                 *         which would mean a contribution arrived too late to
                 *         be counted - an ordering bug, not a user error.
                 */
                void contribute(Graph::ValueId primal, Graph::ValueId gradient) const;

                /** @brief Convenience: contribute to the current node's @p index-th operand. */
                void contribute_input(std::size_t index, Graph::ValueId gradient) const;

                /**
                 * @brief Declares another node already handled by this rule.
                 *
                 * Used by fusions: the cross-entropy rule differentiates
                 * @c LOG_SOFTMAX and @c NLL_LOSS as one, then skips the
                 * log-softmax node it absorbed.
                 *
                 * @param node The node the reverse walk should not visit.
                 */
                void skip(Graph::NodeId node) const;

                /**
                 * @brief The device-resident factor the seeded gradient is multiplied by.
                 *
                 * @c Graph::NO_VALUE unless the caller passed one to
                 * @c differentiate(). A rule that seeds the root gradient - only
                 * @c NLL_LOSS today - must apply it; every other rule can ignore
                 * it, because a factor applied once at the seed reaches the whole
                 * backward pass by linearity.
                 *
                 * @return The 1-element value's id, or @c Graph::NO_VALUE.
                 */
                Graph::ValueId seed_factor() const noexcept { return seed_factor_; }

                /**
                 * @brief Applies @ref seed_factor to @p gradient, or returns it unchanged.
                 *
                 * The one line a seeding rule writes instead of branching on
                 * @c seed_factor() itself, so that "no factor" and "a factor of
                 * one" cannot diverge by a rule forgetting one of the two cases.
                 *
                 * @param gradient The seeded gradient.
                 * @return @p gradient, or a @c DYNAMIC_SCALE of it.
                 */
                Graph::ValueId apply_seed_factor(Graph::ValueId gradient) const;

                /** @brief Internal: constructs a tape over a differentiation run. */
                GradientTape(Graph::GraphBuilder builder, Differentiator& owner,
                             Graph::ValueId seed_factor = Graph::NO_VALUE) noexcept
                    : builder_(builder), owner_(&owner), seed_factor_(seed_factor) {}

                /** @brief Internal: points the tape at the next node. */
                void retarget(Graph::NodeId node) noexcept { node_ = node; }

            private:
                Graph::GraphBuilder builder_;
                Differentiator* owner_;
                Graph::NodeId node_ = Graph::NO_NODE;
                Graph::ValueId seed_factor_ = Graph::NO_VALUE;
        };

    } // namespace Autograd
} // namespace SushiAI
