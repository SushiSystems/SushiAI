/**************************************************************************/
/* differentiator.cpp                                                     */
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
 * @file differentiator.cpp
 * @brief Implements reverse-mode automatic differentiation over computational graphs.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/autograd/backward.hpp>
#include <utility>
#include <vector>
#include <SushiAI/autograd/rule.hpp>
#include <SushiAI/autograd/rules/all.hpp>
#include <SushiAI/autograd/rules/loss.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>
#include "differentiator.hpp"

namespace SushiAI
{
    namespace Autograd
    {
        namespace
        {
            /** @brief Holds d(loss)/d(loss), the value an explicit root seed is filled with. */
            constexpr double K_ROOT_SEED = 1.0;

            /** @brief Finds a rule, or null when the operation has none. */
            constexpr BackwardRule find_rule(OpID op) noexcept
            {
                for (const RuleEntry& entry : K_RULES)
                    if (entry.op == op) return entry.rule;
                return nullptr;
            }
        }

        bool has_backward_rule(OpID op) noexcept
        {
            return find_rule(op) != nullptr;
        }

        void Differentiator::contribute(Graph::ValueID primal, Graph::ValueID gradient)
        {
            SA_FATAL_IF(resolved_.find(primal) != resolved_.end(),
                        "A gradient contribution to value {} arrived after its gradient was "
                        "already read. The reverse walk visits every consumer before the "
                        "producer, so this is an ordering bug in a backward rule, not a "
                        "caller error.",
                        primal);

            const Graph::Value& a = graph_.value(primal);
            const Graph::Value& b = graph_.value(gradient);
            SA_FATAL_IF(a.shape != b.shape,
                        "A gradient of shape {} was contributed to value {} of shape {}",
                        b.shape.to_string(), primal, a.shape.to_string());

            contributions_[primal].push_back(gradient);
            ++contribution_counts_[gradient];
        }

        void Differentiator::skip(Graph::NodeID node)
        {
            SA_FATAL_IF(node >= skipped_.size(),
                        "Cannot skip node {}; the forward graph has {} nodes",
                        node, skipped_.size());
            skipped_[node] = true;
        }

        void Differentiator::observe(const Status& outcome) noexcept
        {
            if (outcome.is_ok())
                return;

            static_cast<void>(builder_.observe(outcome));
            if (failure_.is_ok())
                failure_ = outcome;
        }

        Graph::ValueID Differentiator::resolve(Graph::ValueID primal)
        {
            const auto cached = resolved_.find(primal);
            if (cached != resolved_.end()) return cached->second;

            Graph::ValueID result = Graph::K_NO_VALUE;
            const auto found = contributions_.find(primal);

            if (found != contributions_.end() && !found->second.empty())
            {
                const Result<Graph::ValueID> sum = sum_contributions(found->second);
                if (sum.has_value())
                    result = sum.value();
                else
                    observe(sum.status());
            }

            resolved_.emplace(primal, result);
            if (result != Graph::K_NO_VALUE)
                resolved_gradients_.insert(result);
            return result;
        }

        Result<Graph::ValueID> Differentiator::sum_contributions(
            const std::vector<Graph::ValueID>& parts)
        {
            Graph::ValueID running = parts.front();
            std::size_t next = 1;
            // ACCUMULATE writes into its first operand; a shared one is first added afresh.
            if (parts.size() > 1 && !owns_buffer(running))
            {
                SA_TRY_ASSIGN(running, builder_.add(running, parts[next]));
                ++next;
            }
            for (; next < parts.size(); ++next)
            {
                SA_TRY_ASSIGN(running, builder_.accumulate(running, parts[next]));
            }

            return running;
        }

        bool Differentiator::owns_buffer(Graph::ValueID gradient) const
        {
            const auto count = contribution_counts_.find(gradient);
            if (count == contribution_counts_.end() || count->second != 1)
                return false;
            if (resolved_gradients_.count(gradient) != 0)
                return false;
            const Graph::NodeID producer = graph_.value(gradient).producer;
            return producer != Graph::K_NO_NODE &&
                   static_cast<std::size_t>(producer) >= forward_node_count_;
        }

        Result<Graph::ValueID> Differentiator::resolve_requested(Graph::ValueID value)
        {
            const std::size_t nodes_before = graph_.node_count();
            const Graph::ValueID gradient = resolve(value);
            const Graph::NodeID producer = graph_.value(value).producer;

            SA_FAIL_IF(provenance_ != nullptr && graph_.node_count() > nodes_before
                           && producer == Graph::K_NO_NODE,
                       StatusCode::InvalidDescription,
                       "Cannot record provenance for the gradient of value {}: summing its "
                       "contributions appended a node, and no node produces the value. A "
                       "requested value needs a producer node to carry provenance.",
                       value);

            stamp_appended(producer);
            return gradient;
        }

        void Differentiator::stamp_appended(Graph::NodeID origin)
        {
            if (provenance_ == nullptr)
                return;

            // origin holds one entry per appended node, so its size marks the last stamped node.
            provenance_->origin.resize(graph_.node_count() - forward_node_count_, origin);
        }

        Status Differentiator::require_requestable(Graph::ValueID value) const
        {
            const Graph::Value& requested = graph_.value(value);

            SA_FAIL_IF(requested.lifetime == Graph::Lifetime::Buffer,
                       StatusCode::InvalidDescription,
                       "differentiate() cannot return a gradient for value {} ('{}'): it is a "
                       "buffer, model state that takes no gradient and is written beside the "
                       "graph. Request parameters and inputs only.",
                       value, requested.debug_name);
            return Status();
        }

        Status Differentiator::run()
        {
            SA_TRY(graph_.finalise());

            SA_FAIL_IF(graph_.value(loss_).shape.num_elements() != 1,
                       StatusCode::InvalidDescription,
                       "differentiate() needs a scalar to differentiate; value {} has shape {}",
                       loss_, graph_.value(loss_).shape.to_string());

            const Graph::NodeID root_producer = graph_.value(loss_).producer;
            SA_FAIL_IF(root_producer == Graph::K_NO_NODE, StatusCode::InvalidDescription,
                       "differentiate() cannot seed a gradient for value {}: nothing produces "
                       "it. The loss must be the result of an operation that has a backward "
                       "rule, for example NLL_LOSS (see NN::CrossEntropyLoss) or an ADD of "
                       "loss terms.",
                       loss_);

            // A rule that seeds itself reads no output gradient, so it gets no explicit seed.
            if (!starts_from_implicit_seed(graph_.node(root_producer).op))
            {
                SA_TRY(require_seedable_root(root_producer));
                seed_root(root_producer);
            }

            for (std::size_t offset = forward_node_count_; offset > 0; --offset)
            {
                const Graph::NodeID id = static_cast<Graph::NodeID>(offset - 1);
                SA_TRY(differentiate_node(id));
                stamp_appended(id);
            }

            return Status();
        }

        Status Differentiator::require_seedable_root(Graph::NodeID producer) const
        {
            const OpID op = graph_.node(producer).op;
            const DType dtype = graph_.value(loss_).dtype;

            SA_FAIL_IF(op == Ops::K_LEAF.id, StatusCode::InvalidDescription,
                       "differentiate() cannot differentiate value {}: it is a leaf, produced "
                       "by {}, so no operation computes it from a parameter. The loss must be "
                       "the result of an operation that has a backward rule, for example "
                       "NLL_LOSS (see NN::CrossEntropyLoss) or an ADD of loss terms.",
                       loss_, Ops::name_of(op));
            SA_FAIL_IF(!graph_.value(loss_).requires_grad, StatusCode::InvalidDescription,
                       "differentiate() has nothing to differentiate: value {}, produced by {}, "
                       "requires no gradient. No value that requires a gradient reaches it, or "
                       "every path to it passes through an operation that lets no gradient "
                       "through, such as STOP_GRADIENT.",
                       loss_, Ops::name_of(op));
            SA_FAIL_IF(graph_.node(producer).outputs.front() != loss_,
                       StatusCode::InvalidDescription,
                       "differentiate() cannot differentiate value {}: it is not the first "
                       "result of {}, and a backward rule reads the gradient of the first "
                       "result only. Pass the first result, or a value computed from this one.",
                       loss_, Ops::name_of(op));
            SA_FAIL_IF(!is_real_float(dtype), StatusCode::InvalidDescription,
                       "differentiate() cannot differentiate value {}, produced by {}: its "
                       "element type is {}, and a gradient exists only for half, float32 and "
                       "float64.",
                       loss_, Ops::name_of(op), dtype_name(dtype));
            SA_FAIL_IF(find_rule(op) == nullptr, StatusCode::InvalidDescription,
                       "differentiate() cannot differentiate value {}: {} produces it and has "
                       "no backward rule. Produce the loss with a differentiable operation, or "
                       "add a rule to the appropriate domain under autograd/rules/.",
                       loss_, Ops::name_of(op));
            return Status();
        }

        void Differentiator::seed_root(Graph::NodeID producer)
        {
            tape_.contribute(loss_,
                             tape_.apply_seed_factor(builder_.constant_like(loss_, K_ROOT_SEED)));
            stamp_appended(producer);
        }

        Status Differentiator::differentiate_node(Graph::NodeID id)
        {
            if (skipped_[id])
                return Status();

            // Snapshot before anything below can grow the graph.
            current_node_ = graph_.node(id);

            if (Ops::is_gradient_barrier(current_node_.op))
                return Status();

            bool any_input_needs_grad = false;
            for (Graph::ValueID in : current_node_.inputs)
            {
                if (graph_.value(in).requires_grad)
                {
                    any_input_needs_grad = true;
                    break;
                }
            }

            // Skip leaves and frozen subgraphs whose operands require no gradient.
            if (!any_input_needs_grad)
                return Status();

            bool reachable = false;
            for (Graph::ValueID out : current_node_.outputs)
            {
                if (is_root(out) || resolve(out) != Graph::K_NO_VALUE)
                {
                    reachable = true;
                    break;
                }
            }

            if (!reachable)
                return Status();

            const BackwardRule rule = find_rule(current_node_.op);
            SA_FAIL_IF(rule == nullptr, StatusCode::InvalidDescription,
                       "No backward rule for {} (node {}), but its operands require a "
                       "gradient. Add one to the appropriate domain under "
                       "autograd/rules/, or mark the subgraph as not requiring gradients.",
                       Ops::name_of(current_node_.op), id);

            tape_.retarget(id);
            rule(tape_);
            return Status();
        }

        Result<std::vector<Graph::ValueID>> differentiate(
            Graph::Graph& graph, Graph::ValueID loss,
            SushiRuntime::span<const Graph::ValueID> wrt, Graph::ValueID seed_factor,
            Provenance* provenance)
        {
            SA_FAIL_IF(graph.has_refusal(), StatusCode::InvalidDescription,
                       "differentiate() was given a graph in which an emitter refused part of "
                       "the description, and the emitter's status names the cause. Discard "
                       "the graph and trace it again.");

            Differentiator differentiator(graph, loss, seed_factor, provenance);

            // Every request is checked before the sweep, so a refused one appends no node.
            for (Graph::ValueID value : wrt)
                SA_TRY(differentiator.require_requestable(value));

            SA_TRY(differentiator.run());

            std::vector<Graph::ValueID> gradients;
            gradients.reserve(wrt.size());

            for (Graph::ValueID value : wrt)
            {
                SA_TRY_ASSIGN(const Graph::ValueID gradient,
                              differentiator.resolve_requested(value));
                gradients.push_back(gradient);

                // Parameter gradients persist across execution and receive Lifetime::GRADIENT.
                if (gradient != Graph::K_NO_VALUE
                    && graph.value(value).lifetime == Graph::Lifetime::Parameter)
                    graph.value(gradient).lifetime = Graph::Lifetime::Gradient;
            }

            // A rule reports a refused emission only through observe(), so it is read here.
            SA_TRY(differentiator.failure());
            SA_FAIL_IF(graph.has_refusal(), StatusCode::InvalidDescription,
                       "differentiate() appended a gradient that an emitter refused");

            return std::move(gradients);
        }

    } // namespace Autograd
} // namespace SushiAI
