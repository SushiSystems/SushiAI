/****************************************************************/
/* tape.hpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file tape.hpp
 * @brief Declares the gradient tape interface exposed to backward rules.
 * @author Mustafa Garip
 */

#pragma once

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
         * @brief Holds traversal state and emitter interfaces for backward rules.
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
                 * @brief Returns the gradient ValueId of the node's result.
                 * @param index Output index on the current node.
                 * @return Gradient ValueId, or Graph::NO_VALUE if unseeded or inactive.
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
                 * @brief Registers a gradient contribution to a forward primal value.
                 * @param primal Forward value receiving the gradient.
                 * @param gradient Contribution value to accumulate.
                 */
                void contribute(Graph::ValueId primal, Graph::ValueId gradient) const;

                /** @brief Convenience: contribute to the current node's @p index-th operand. */
                void contribute_input(std::size_t index, Graph::ValueId gradient) const;

                /**
                 * @brief Marks an upstream node as already handled by the current rule.
                 * @param node Node identifier to skip during reverse traversal.
                 */
                void skip(Graph::NodeId node) const;

                /**
                 * @brief Returns the optional device-resident scaling factor for the root gradient.
                 * @return Single-element ValueId, or Graph::NO_VALUE if unscaled.
                 */
                Graph::ValueId seed_factor() const noexcept { return seed_factor_; }

                /**
                 * @brief Multiplies gradient by seed_factor if present, returning the result.
                 * @param gradient Root gradient to scale.
                 * @return Scaled gradient ValueId, or gradient unchanged if unscaled.
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
