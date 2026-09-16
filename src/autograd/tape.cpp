/**************************************************************************/
/* tape.cpp                                                               */
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
