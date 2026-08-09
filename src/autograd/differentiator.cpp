/**************************************************************************/
/* differentiator.cpp                                                    */
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

#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rule.hpp>
#include <SushiAI/autograd/rules/all.hpp>
#include <SushiAI/autograd/rules/loss.hpp>
#include "differentiator.hpp"

namespace SushiAI
{
    namespace Autograd
    {
        namespace
        {
            /** @brief The extra entry that only exists to produce a good error. */
            constexpr RuleEntry LOG_SOFTMAX_RULE{Ops::LOG_SOFTMAX.id, &log_softmax_backward};

            /** @brief Finds a rule, or null when the operation has none. */
            constexpr BackwardRule find_rule(OpID op) noexcept
            {
                for (const RuleEntry& entry : RULES)
                    if (entry.op == op) return entry.rule;
                if (op == LOG_SOFTMAX_RULE.op) return LOG_SOFTMAX_RULE.rule;
                return nullptr;
            }
        }

        bool has_backward_rule(OpID op) noexcept
        {
            return find_rule(op) != nullptr;
        }

        // =====================================================================
        // Differentiator
        // =====================================================================

        void Differentiator::contribute(Graph::ValueId primal, Graph::ValueId gradient)
        {
            SA_THROW_IF(resolved_.find(primal) != resolved_.end(),
                        "A gradient contribution to value {} arrived after its gradient was "
                        "already read. The reverse walk visits every consumer before the "
                        "producer, so this is an ordering bug in a backward rule, not a "
                        "caller error.",
                        primal);

            const Graph::Value& a = graph_.value(primal);
            const Graph::Value& b = graph_.value(gradient);
            SA_THROW_IF(a.shape != b.shape,
                        "A gradient of shape {} was contributed to value {} of shape {}",
                        b.shape.to_string(), primal, a.shape.to_string());

            contributions_[primal].push_back(gradient);
        }

        void Differentiator::skip(Graph::NodeId node)
        {
            SA_THROW_IF(node >= skipped_.size(),
                        "Cannot skip node {}; the forward graph has {} nodes",
                        node, skipped_.size());
            skipped_[node] = true;
        }

        Graph::ValueId Differentiator::resolve(Graph::ValueId primal)
        {
            // The differentiation root's own gradient is the implicit scalar
            // one; there is no tensor for it and the rule that seeds the sweep
            // knows not to ask for one.
            if (primal == loss_) return Graph::NO_VALUE;

            const auto cached = resolved_.find(primal);
            if (cached != resolved_.end()) return cached->second;

            Graph::ValueId result = Graph::NO_VALUE;
            const auto found = contributions_.find(primal);

            if (found != contributions_.end() && !found->second.empty())
            {
                const std::vector<Graph::ValueId>& parts = found->second;
                result = parts.front();
                for (std::size_t i = 1; i < parts.size(); ++i)
                    result = builder_.accumulate(result, parts[i]);
            }

            resolved_.emplace(primal, result);
            return result;
        }

        void Differentiator::run()
        {
            graph_.finalise();

            SA_THROW_IF(graph_.value(loss_).shape.num_elements() != 1,
                        "differentiate() needs a scalar to differentiate; value {} has shape {}",
                        loss_, graph_.value(loss_).shape.to_string());

            // Being one element is necessary and not sufficient. The root's own
            // gradient is the implicit scalar one, so there is no tensor for
            // resolve() to hand back, and every rule but nll_loss_backward
            // reads that NO_VALUE as "nothing flows here" and returns. A [1, 1]
            // MATMUL therefore differentiates to a complete set of empty
            // gradients and no error - the caller only finds out one layer
            // later, from bind_parameters blaming the model for a loss-side
            // mistake. Reject it here, where the actual cause is still known.
            const Graph::NodeId root_producer = graph_.value(loss_).producer;
            SA_THROW_IF(root_producer == Graph::NO_NODE,
                        "differentiate() cannot seed a gradient for value {}: nothing produces "
                        "it. The loss must be the result of NLL_LOSS - see NN::CrossEntropyLoss.",
                        loss_);
            SA_THROW_IF(graph_.node(root_producer).op != Ops::NLL_LOSS.id,
                        "differentiate() seeds the loss gradient as the implicit scalar one, and "
                        "only NLL_LOSS knows to start from it. Value {} is produced by {}, whose "
                        "rule declines an absent output gradient and returns, so every requested "
                        "gradient would come back empty with nothing reported. Produce the loss "
                        "with NLL_LOSS - see NN::CrossEntropyLoss.",
                        loss_, Ops::name_of(graph_.node(root_producer).op));

            for (std::size_t offset = forward_node_count_; offset > 0; --offset)
            {
                const Graph::NodeId id = static_cast<Graph::NodeId>(offset - 1);
                if (skipped_[id]) continue;

                // Snapshot before anything below can grow the graph.
                current_node_ = graph_.node(id);

                bool any_input_needs_grad = false;
                for (Graph::ValueId in : current_node_.inputs)
                    if (graph_.value(in).requires_grad) { any_input_needs_grad = true; break; }

                // A leaf has no operands, so this also skips every leaf, and a
                // frozen subgraph costs nothing at all.
                if (!any_input_needs_grad) continue;

                bool reachable = false;
                for (Graph::ValueId out : current_node_.outputs)
                    if (is_root(out) || resolve(out) != Graph::NO_VALUE) { reachable = true; break; }

                if (!reachable) continue;

                const BackwardRule rule = find_rule(current_node_.op);
                SA_THROW_IF(rule == nullptr,
                            "No backward rule for {} (node {}), but its operands require a "
                            "gradient. Add one to the appropriate domain under "
                            "autograd/rules/, or mark the subgraph as not requiring gradients.",
                            Ops::name_of(current_node_.op), id);

                tape_.retarget(id);
                rule(tape_);
            }
        }

        std::vector<Graph::ValueId> differentiate(Graph::Graph& graph,
                                                  Graph::ValueId loss,
                                                  SushiRuntime::span<const Graph::ValueId> wrt,
                                                  Graph::ValueId seed_factor)
        {
            Differentiator differentiator(graph, loss, seed_factor);
            differentiator.run();

            std::vector<Graph::ValueId> gradients;
            gradients.reserve(wrt.size());

            for (Graph::ValueId value : wrt)
            {
                const Graph::ValueId gradient = differentiator.resolve(value);
                gradients.push_back(gradient);

                // A parameter's gradient outlives the step exactly as the
                // parameter does, so it belongs in the gradient arena rather
                // than being packed against the activations.
                if (gradient != Graph::NO_VALUE
                    && graph.value(value).lifetime == Graph::Lifetime::PARAMETER)
                    graph.value(gradient).lifetime = Graph::Lifetime::GRADIENT;
            }

            return gradients;
        }

    } // namespace Autograd
} // namespace SushiAI
