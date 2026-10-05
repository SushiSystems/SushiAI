/****************************************************************/
/* tape.cpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file tape.cpp
 * @brief Implements gradient tape operations querying and extending backward graphs.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/tape.hpp>
#include "differentiator.hpp"

namespace SushiAI
{
    namespace Autograd
    {
        const Graph::Node& GradientTape::node() const
        {
            return owner_->current_node();
        }

        Graph::ValueId GradientTape::apply_seed_factor(Graph::ValueId gradient) const
        {
            if (seed_factor_ == Graph::NO_VALUE) return gradient;
            return builder_.dynamic_scale(gradient, seed_factor_, "loss_scaled_grad");
        }

        Graph::ValueId GradientTape::output_grad(std::size_t index) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.outputs.size(),
                        "{} has no result {}", Ops::name_of(n.op), index);
            return owner_->resolve(n.outputs[index]);
        }

        bool GradientTape::output_is_root(std::size_t index) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.outputs.size(),
                        "{} has no result {}", Ops::name_of(n.op), index);
            return owner_->is_root(n.outputs[index]);
        }

        bool GradientTape::input_requires_grad(std::size_t index) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.inputs.size(),
                        "{} has no operand {}", Ops::name_of(n.op), index);
            return owner_->graph().value(n.inputs[index]).requires_grad;
        }

        void GradientTape::contribute(Graph::ValueId primal, Graph::ValueId gradient) const
        {
            owner_->contribute(primal, gradient);
        }

        void GradientTape::contribute_input(std::size_t index, Graph::ValueId gradient) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.inputs.size(),
                        "{} has no operand {}", Ops::name_of(n.op), index);
            owner_->contribute(n.inputs[index], gradient);
        }

        void GradientTape::skip(Graph::NodeId node_to_skip) const
        {
            owner_->skip(node_to_skip);
        }

    } // namespace Autograd
} // namespace SushiAI
