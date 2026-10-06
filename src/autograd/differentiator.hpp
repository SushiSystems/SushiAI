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
 */

#pragma once

#include <unordered_map>
#include <vector>
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
        class Differentiator
        {
            public:
                Differentiator(Graph::Graph& graph, Graph::ValueId loss,
                               Graph::ValueId seed_factor)
                    : graph_(graph),
                      loss_(loss),
                      builder_(graph),
                      forward_node_count_(graph.node_count()),
                      skipped_(graph.node_count(), false),
                      tape_(Graph::GraphBuilder(graph), *this, seed_factor)
                {
                }

                /** @brief Runs the reverse sweep over the forward nodes. */
                void run();

                /** @brief The summed gradient of @p primal, materialising the sum on first read. */
                Graph::ValueId resolve(Graph::ValueId primal);

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
                Graph::Graph& graph_;
                Graph::ValueId loss_;
                Graph::GraphBuilder builder_;
                std::size_t forward_node_count_;
                std::vector<bool> skipped_;
                std::unordered_map<Graph::ValueId, std::vector<Graph::ValueId>> contributions_;
                std::unordered_map<Graph::ValueId, Graph::ValueId> resolved_;
                GradientTape tape_;
                Graph::Node current_node_{};
        };

    } // namespace Autograd
} // namespace SushiAI
