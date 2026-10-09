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
 * @see include/SushiAI/eager/README.md
 */

#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/autograd/tape.hpp>

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
                Differentiator(Graph::Graph& graph, Graph::ValueId loss,
                               Graph::ValueId seed_factor, Provenance* provenance)
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
                 * @brief Throws unless a gradient may be requested for @p value.
                 * @param value A value of the graph whose lifetime is not BUFFER.
                 */
                void require_requestable(Graph::ValueId value) const;

                /** @brief Runs the reverse sweep over the forward nodes. */
                void run();

                /** @brief The summed gradient of @p primal, materialising the sum on first read. */
                Graph::ValueId resolve(Graph::ValueId primal);

                /**
                 * @brief Returns the gradient a caller requested for @p value, after run().
                 * @see include/SushiAI/eager/README.md, "Provenance".
                 */
                Graph::ValueId resolve_requested(Graph::ValueId value);

                /** @brief Records one gradient contribution to @p primal. */
                void contribute(Graph::ValueId primal, Graph::ValueId gradient);

                /** @brief Marks @p node as already differentiated by a fusion. */
                void skip(Graph::NodeId node);

                /** @brief Whether @p value is the scalar being differentiated. */
                bool is_root(Graph::ValueId value) const noexcept { return value == loss_; }

                /**
                 * @brief Returns a snapshot copy of the node currently being differentiated.
                 * @return Copied Node snapshot to prevent invalidation during graph growth.
                 */
                const Graph::Node& current_node() const noexcept { return current_node_; }

                /** @brief The graph being extended. */
                Graph::Graph& graph() noexcept { return graph_; }

            private:
                /**
                 * @brief Throws unless the root can take an explicit seed.
                 * @param producer The node that produces the root.
                 * @pre finalise() has run on the graph.
                 */
                void require_seedable_root(Graph::NodeId producer) const;

                /**
                 * @brief Emits the explicit one that seeds the root and records it as a gradient.
                 * @param producer The node that produces the root; the seed nodes get it as origin.
                 * @pre require_seedable_root(producer) has passed.
                 */
                void seed_root(Graph::NodeId producer);

                /** @brief Emits the gradient nodes of forward node @p id, when it needs any. */
                void differentiate_node(Graph::NodeId id);

                /**
                 * @brief Records @p origin for every node appended since the last call.
                 * @see include/SushiAI/eager/README.md, "Provenance".
                 */
                void stamp_appended(Graph::NodeId origin);

                /** @brief Holds the graph being extended. */
                Graph::Graph& graph_;

                /** @brief Holds the scalar being differentiated. */
                Graph::ValueId loss_;

                /** @brief Emits the ACCUMULATE nodes that sum contributions. */
                Graph::GraphBuilder builder_;

                /** @brief Holds the node count of the graph when the run began. */
                std::size_t forward_node_count_;

                /** @brief Holds the forward nodes a fusion already differentiated, by node id. */
                std::vector<bool> skipped_;

                /** @brief Holds the gradient contributions received so far, per primal value. */
                std::unordered_map<Graph::ValueId, std::vector<Graph::ValueId>> contributions_;

                /** @brief Holds the summed gradients already read, per primal value. */
                std::unordered_map<Graph::ValueId, Graph::ValueId> resolved_;

                /** @brief Holds the view of this run that backward rules receive. */
                GradientTape tape_;

                /** @brief Receives the origin of every appended node; null when not requested. */
                Provenance* provenance_;

                /** @brief Holds a snapshot of the forward node the walk is at. */
                Graph::Node current_node_{};
        };

    } // namespace Autograd
} // namespace SushiAI
