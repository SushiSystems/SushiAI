/**************************************************************************/
/* tape.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file tape.hpp
 * @brief Declares the gradient tape interface exposed to backward rules.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief The differentiation run's private bookkeeping; see
         * modules/core/autograd/source/differentiator.cpp.
         */
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
                Graph::NodeID node_id() const noexcept { return node_; }

                /**
                 * @brief Returns the gradient ValueID of the node's result.
                 * @param index Output index on the current node.
                 * @return Gradient ValueID, or Graph::NO_VALUE if unseeded or inactive.
                 */
                Graph::ValueID output_grad(std::size_t index) const;

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
                void contribute(Graph::ValueID primal, Graph::ValueID gradient) const;

                /** @brief Convenience: contribute to the current node's @p index-th operand. */
                void contribute_input(std::size_t index, Graph::ValueID gradient) const;

                /**
                 * @brief Marks an upstream node as already handled by the current rule.
                 * @param node Node identifier to skip during reverse traversal.
                 */
                void skip(Graph::NodeID node) const;

                /**
                 * @brief Returns the optional device-resident scaling factor for the root gradient.
                 * @return Single-element ValueID, or Graph::NO_VALUE if unscaled.
                 */
                Graph::ValueID seed_factor() const noexcept { return seed_factor_; }

                /**
                 * @brief Multiplies gradient by seed_factor if present, returning the result.
                 * @param gradient Root gradient to scale.
                 * @return Scaled gradient ValueID, or gradient unchanged if unscaled.
                 */
                Graph::ValueID apply_seed_factor(Graph::ValueID gradient) const;

                /** @brief Internal: constructs a tape over a differentiation run. */
                GradientTape(Graph::GraphBuilder builder, Differentiator& owner,
                             Graph::ValueID seed_factor = Graph::K_NO_VALUE) noexcept
                    : builder_(builder), owner_(&owner), seed_factor_(seed_factor) {}

                /** @brief Internal: points the tape at the next node. */
                void retarget(Graph::NodeID node) noexcept { node_ = node; }

            private:
                Graph::GraphBuilder builder_;
                Differentiator* owner_;
                Graph::NodeID node_ = Graph::K_NO_NODE;
                Graph::ValueID seed_factor_ = Graph::K_NO_VALUE;
        };

    } // namespace Autograd
} // namespace SushiAI
