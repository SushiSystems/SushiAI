/**************************************************************************/
/* backward.hpp                                                           */
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

/**
 * @file backward.hpp
 * @brief Reverse-mode differentiation as an IR-to-IR transform.
 *
 * @c differentiate() reads a graph and writes more of the same graph. It
 * touches no device, allocates no tensor and compiles nothing - see ir.hpp's
 * file comment for why that is the load-bearing decision of the whole library
 * rather than a stylistic one.
 *
 * @section registry Why the rule table is a plain array
 *
 * The obvious shape for "one rule per operation" is a static map that each
 * rule's translation unit registers itself into from a namespace-scope
 * initializer. That shape is wrong here, and it fails silently. @c sushiai is
 * a @b static library: the linker pulls in only those objects that resolve a
 * referenced symbol, and a translation unit whose only purpose is to run a
 * constructor resolves nothing. It is dropped, the registration never runs,
 * and the failure surfaces as "no backward rule for sushiai.matmul" in a
 * consumer's build but not in this repo's own tests, which reference
 * everything anyway.
 *
 * A file-scope @c constexpr array of {OpID, function pointer} plus a linear
 * search cannot be dropped, needs no initialization order, and searches two
 * dozen entries in less time than one graph-node allocation.
 *
 * @section accumulation Fan-out, tiered
 *
 * A value read by several consumers receives one gradient contribution per
 * consumer, and they must be summed. The sum is tiered so the common cases
 * cost nothing:
 *
 * - 0 contributions: no gradient value at all. A symbolic zero, not a
 *   materialised buffer of zeros - a frozen layer must cost nothing, not one
 *   wasted kernel and one wasted allocation per parameter.
 * - 1 contribution: the gradient @e is that value. Nothing emitted.
 * - 2 or more: a chain of @c ACCUMULATE nodes, in contributor order.
 *
 * The chain is ordered by the graph's own edges. No atomics: an atomic sum
 * reorders run to run, and SushiRuntime's @c sushiruntime_deterministic_fp
 * contract - which SushiAI inherits rather than re-states - promises
 * byte-identical replay. Non-associative float addition in a
 * nondeterministic order would break that quietly, showing up as a training
 * run that cannot be reproduced rather than as a test failure.
 *
 * Host lane: no SushiBLAS, no SYCL.
 */

#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief The differentiation run's private bookkeeping; see backward.cpp. */
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

        /** @brief What every backward rule looks like. */
        using BackwardRule = void (*)(const GradientTape&);

        /**
         * @brief Whether an operation can be differentiated.
         * @param op The operation to look up.
         * @return True when a backward rule exists for it.
         */
        bool has_backward_rule(OpID op) noexcept;

        /**
         * @brief Extends @p graph with the gradient of @p loss.
         *
         * Walks the forward nodes in reverse - insertion order is topological,
         * so reversing it visits every consumer of a value before its producer,
         * which is exactly the condition under which a value's gradient is
         * complete by the time its producer asks for it.
         *
         * @c propagate_requires_grad() runs first, and any node none of whose
         * operands requires a gradient is skipped whole. A frozen layer
         * therefore contributes no nodes at all rather than nodes that compute
         * a gradient nobody reads.
         *
         * The gradient of @p loss with respect to itself is the implicit scalar
         * one; no tensor is created for it, so @p loss must be produced by an
         * operation whose rule knows that - today, @c NLL_LOSS. Anything else
         * is rejected rather than silently differentiated against a gradient
         * that does not exist.
         *
         * @param graph The graph to differentiate and extend, in place.
         * @param loss The scalar being differentiated; must be a 1-element value.
         * @param wrt The values whose gradients the caller wants.
         * @param seed_factor A 1-element value the seeded gradient is multiplied
         *        by, or @c Graph::NO_VALUE for none. This is where gradient loss
         *        scaling enters the graph, and it is here rather than in a pass
         *        of its own for one reason: the seed is *built* here, so
         *        multiplying it costs one node emitted in topological position,
         *        where a later pass would have to re-route every consumer of the
         *        seed and rebuild the node list to stay in insertion order. The
         *        factor is a value and not a number because it changes between
         *        replays; see @c Ops::DYNAMIC_SCALE and @c optim/loss_scaler.hpp.
         * @return One id per entry of @p wrt, in the same order;
         *         @c Graph::NO_VALUE where no gradient flows to that value.
         * @throws Error If @p loss is not scalar, is not produced by a rule
         *         that seeds itself, or the graph contains a differentiable
         *         operation with no backward rule.
         */
        std::vector<Graph::ValueId> differentiate(Graph::Graph& graph,
                                                  Graph::ValueId loss,
                                                  SushiRuntime::span<const Graph::ValueId> wrt,
                                                  Graph::ValueId seed_factor = Graph::NO_VALUE);

    } // namespace Autograd
} // namespace SushiAI
