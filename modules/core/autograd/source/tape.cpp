/**************************************************************************/
/* tape.cpp                                                               */
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
 * @file tape.cpp
 * @brief Implements gradient tape operations querying and extending backward graphs.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/tape.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include "differentiator.hpp"

namespace SushiAI
{
    namespace Autograd
    {
        namespace
        {
            /**
             * @brief Hands @p gradient to @p owner as a contribution to @p primal.
             * @return A success, or the failure @p gradient holds, which contributes nothing.
             */
            Status forward_contribution(Differentiator& owner, Graph::ValueID primal,
                                        const Graph::Traced& gradient)
            {
                SA_TRY_ASSIGN(const Graph::ValueID id, gradient);
                owner.contribute(primal, id);
                return Status();
            }
        } // namespace

        const Graph::Node& GradientTape::node() const noexcept
        {
            return owner_->current_node();
        }

        Graph::Traced GradientTape::apply_seed_factor(const Graph::Traced& gradient) const noexcept
        {
            if (seed_factor_ == Graph::K_NO_VALUE) return gradient;
            return builder_.dynamic_scale(gradient, seed_factor_, "loss_scaled_grad");
        }

        Graph::ValueID GradientTape::input(std::size_t index) const noexcept
        {
            const Graph::Node& n = node();
            SA_FATAL_IF(index >= n.inputs.size(),
                        "{} has no operand {}", Ops::name_of(n.op), index);
            return n.inputs[index];
        }

        Graph::ValueID GradientTape::output(std::size_t index) const noexcept
        {
            const Graph::Node& n = node();
            SA_FATAL_IF(index >= n.outputs.size(),
                        "{} has no result {}", Ops::name_of(n.op), index);
            return n.outputs[index];
        }

        Graph::ValueID GradientTape::output_grad(std::size_t index) const noexcept
        {
            const Graph::Node& n = node();
            SA_FATAL_IF(index >= n.outputs.size(),
                        "{} has no result {}", Ops::name_of(n.op), index);
            return owner_->resolve(n.outputs[index]);
        }

        bool GradientTape::output_is_root(std::size_t index) const noexcept
        {
            const Graph::Node& n = node();
            SA_FATAL_IF(index >= n.outputs.size(),
                        "{} has no result {}", Ops::name_of(n.op), index);
            return owner_->is_root(n.outputs[index]);
        }

        bool GradientTape::input_requires_grad(std::size_t index) const noexcept
        {
            const Graph::Node& n = node();
            SA_FATAL_IF(index >= n.inputs.size(),
                        "{} has no operand {}", Ops::name_of(n.op), index);
            return owner_->graph().value(n.inputs[index]).requires_grad;
        }

        void GradientTape::contribute(Graph::ValueID primal, const Graph::Traced& gradient) const noexcept
        {
            observe(forward_contribution(*owner_, primal, gradient));
        }

        void GradientTape::contribute_input(std::size_t index,
                                            const Graph::Traced& gradient) const noexcept
        {
            const Graph::Node& n = node();
            SA_FATAL_IF(index >= n.inputs.size(),
                        "{} has no operand {}", Ops::name_of(n.op), index);
            observe(forward_contribution(*owner_, n.inputs[index], gradient));
        }

        void GradientTape::observe(const Status& outcome) const noexcept
        {
            if (outcome.is_ok())
                return;

            // The root seed is emitted before any node is current, and its message stands alone.
            if (node_ == Graph::K_NO_NODE)
            {
                owner_->observe(outcome);
                return;
            }

            owner_->observe(SA_ANNOTATE(outcome, "the backward rule of {} (node {})",
                                        Ops::name_of(node().op), node_));
        }

        void GradientTape::skip(Graph::NodeID node_to_skip) const noexcept
        {
            owner_->skip(node_to_skip);
        }

    } // namespace Autograd
} // namespace SushiAI
