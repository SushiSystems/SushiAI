/**************************************************************************/
/* differentiator.hpp                                                     */
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
 * @file differentiator.hpp
 * @brief Declares internal bookkeeping for reverse-mode differentiation runs.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/autograd/tape.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

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
        class Differentiator final
        {
            public:
                /**
                 * @brief Binds one run to @p graph and resets @p provenance when it is given.
                 * @param provenance Receives the origin of every appended node; may be null.
                 */
                Differentiator(Graph::Graph& graph, Graph::ValueID loss,
                               Graph::ValueID seed_factor, Provenance* provenance)
                    : graph_(graph),
                      loss_(loss),
                      builder_(graph),
                      forward_node_count_(graph.node_count()),
                      skipped_(graph.node_count(), false),
                      tape_(Graph::GraphBuilder(graph), *this, seed_factor),
                      provenance_(provenance)
                {
                    if (provenance_ == nullptr)
                        return;

                    provenance_->forward_node_count = forward_node_count_;
                    provenance_->origin.clear();
                }

                /**
                 * @brief Checks that a gradient may be requested for @p value.
                 * @param value A value of the graph.
                 * @return A success, or InvalidDescription for a BUFFER value.
                 */
                [[nodiscard]] Status require_requestable(Graph::ValueID value) const;

                /**
                 * @brief Runs the reverse sweep over the forward nodes.
                 * @return A success, or InvalidDescription for a graph whose layouts or root
                 *         cannot be differentiated, or for a node without a backward rule.
                 */
                [[nodiscard]] Status run();

                /**
                 * @brief The summed gradient of @p primal, materialising the sum on first read.
                 * @return The sum, or Graph::K_NO_VALUE when nothing was contributed or the sum
                 *         was refused; a refusal is marked on the graph.
                 */
                Graph::ValueID resolve(Graph::ValueID primal);

                /**
                 * @brief Emits the sum of @p parts, accumulating in place where owns_buffer()
                 *        allows.
                 * @param parts One or more contributions to one primal value.
                 * @return The sum, or the failure an emitter returned.
                 */
                [[nodiscard]] Result<Graph::ValueID> sum_contributions(
                    const std::vector<Graph::ValueID>& parts);

                /**
                 * @brief Returns whether @p gradient may be accumulated into in place: no other
                 *        contribution, resolved gradient or forward node can read it.
                 */
                bool owns_buffer(Graph::ValueID gradient) const;

                /**
                 * @brief Returns the gradient a caller requested for @p value, after run().
                 * @return The gradient or Graph::K_NO_VALUE, or InvalidDescription when the
                 *         sum appended a node that no producer of @p value can carry as origin.
                 * @see modules/core/eager/README.md, "Provenance".
                 */
                [[nodiscard]] Result<Graph::ValueID> resolve_requested(Graph::ValueID value);

                /**
                 * @brief Marks the graph and keeps @p outcome when it is the first failure.
                 * @param outcome A success, which changes nothing, or a refused emission.
                 */
                void observe(const Status& outcome) noexcept;

                /** @brief The first failure observe() was given, or a success. */
                [[nodiscard]] const Status& failure() const noexcept { return failure_; }

                /** @brief Records one gradient contribution to @p primal. */
                void contribute(Graph::ValueID primal, Graph::ValueID gradient);

                /** @brief Marks @p node as already differentiated by a fusion. */
                void skip(Graph::NodeID node);

                /** @brief Whether @p value is the scalar being differentiated. */
                bool is_root(Graph::ValueID value) const noexcept { return value == loss_; }

                /**
                 * @brief Returns a snapshot copy of the node currently being differentiated.
                 * @return Copied Node snapshot to prevent invalidation during graph growth.
                 */
                const Graph::Node& current_node() const noexcept { return current_node_; }

                /** @brief The graph being extended. */
                Graph::Graph& graph() noexcept { return graph_; }

            private:
                /**
                 * @brief Checks that the root can take an explicit seed.
                 * @param producer The node that produces the root.
                 * @return A success, or InvalidDescription naming why the root cannot be seeded.
                 * @pre finalise() has run on the graph.
                 */
                [[nodiscard]] Status require_seedable_root(Graph::NodeID producer) const;

                /**
                 * @brief Emits the explicit one that seeds the root and records it as a gradient.
                 * @param producer The node that produces the root; the seed nodes get it as origin.
                 * @pre require_seedable_root(producer) has passed.
                 */
                void seed_root(Graph::NodeID producer);

                /**
                 * @brief Emits the gradient nodes of forward node @p id, when it needs any.
                 * @return A success, or InvalidDescription for a node without a backward rule
                 *         whose operands require a gradient.
                 */
                [[nodiscard]] Status differentiate_node(Graph::NodeID id);

                /**
                 * @brief Records @p origin for every node appended since the last call.
                 * @see modules/core/eager/README.md, "Provenance".
                 */
                void stamp_appended(Graph::NodeID origin);

                /** @brief Holds the graph being extended. */
                Graph::Graph& graph_;

                /** @brief Holds the scalar being differentiated. */
                Graph::ValueID loss_;

                /** @brief Emits the ACCUMULATE nodes that sum contributions. */
                Graph::GraphBuilder builder_;

                /** @brief Holds the node count of the graph when the run began. */
                std::size_t forward_node_count_;

                /** @brief Holds the forward nodes a fusion already differentiated, by node id. */
                std::vector<bool> skipped_;

                /** @brief Holds the gradient contributions received so far, per primal value. */
                std::unordered_map<Graph::ValueID, std::vector<Graph::ValueID>> contributions_;

                /** @brief Holds the summed gradients already read, per primal value. */
                std::unordered_map<Graph::ValueID, Graph::ValueID> resolved_;

                /** @brief Holds how many contributions name each gradient value. */
                std::unordered_map<Graph::ValueID, std::size_t> contribution_counts_;

                /** @brief Holds every gradient value resolved_ returns. */
                std::unordered_set<Graph::ValueID> resolved_gradients_;

                /** @brief Holds the view of this run that backward rules receive. */
                GradientTape tape_;

                /** @brief Receives the origin of every appended node; null when not requested. */
                Provenance* provenance_;

                /** @brief Holds a snapshot of the forward node the walk is at. */
                Graph::Node current_node_{};

                /** @brief Holds the first refused emission of this run, or a success. */
                Status failure_;
        };

    } // namespace Autograd
} // namespace SushiAI
