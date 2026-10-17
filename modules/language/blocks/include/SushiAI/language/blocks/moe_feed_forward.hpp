/**************************************************************************/
/* moe_feed_forward.hpp                                                   */
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
 * @file moe_feed_forward.hpp
 * @brief Declares MoEFeedForward, a mixture of feed-forward experts gated by a router.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/expert_projection.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /**
         * @brief Maps each position of a [N, T, dim] sequence through every expert and sums
         *        the results by the gate its router gives the position.
         *
         * README.md states the mixture, the gate of top_k and the balance term.
         */
        struct MoEFeedForward
        {
            /** @brief Holds the width of the input, which is that of the result. */
            int64_t dim = 0;

            /** @brief Holds the width between the two linear layers of one expert. */
            int64_t hidden = 0;

            /** @brief Holds the number of experts. */
            int64_t experts = 0;

            /** @brief Holds how many experts a position keeps; 0 keeps every one. */
            int64_t top_k = 0;

            /** @brief Holds the linear layer from dim to one score per expert. */
            NN::Linear router{};

            /** @brief Holds the first linear layer of every expert, dim to experts * hidden. */
            NN::Linear expand{};

            /** @brief Holds the activation between the two linear layers. */
            NN::GELU activation{};

            /** @brief Holds the second linear layer of every expert and their biases. */
            ExpertProjection project{};

            /** @brief Builds a block of no width, which no input fits. */
            MoEFeedForward() = default;

            /** @brief Builds the router and the stacked layers; trace_experts() checks them. */
            MoEFeedForward(int64_t dim, int64_t hidden, int64_t experts, int64_t top_k) noexcept;

            /** @brief Lists the three children, under "router", "expand" and "project". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("router", &MoEFeedForward::router),
                    NN::child("expand", &MoEFeedForward::expand),
                    NN::child("project", &MoEFeedForward::project),
                };
            }

            /**
             * @brief Traces the mixture over the [N * T, dim] rows of @p x, with the usage
             *        of the experts and its balance term.
             * @param x A [N, T, dim] value of a real floating type.
             * @return The three ids, or InvalidDescription when @p x or this block's
             *         description does not fit; nothing is recorded on @p trace.
             */
            [[nodiscard]] Result<ExpertTrace> trace_experts(const NN::Trace& trace,
                                                            Graph::ValueID x);

            /**
             * @brief Traces trace_experts() over the newest slice, as a position reads no other.
             * @param sample Not read.
             */
            [[nodiscard]] Result<ExpertTrace> trace_experts_incremental(const NN::Trace& trace,
                                                                        Graph::ValueID x,
                                                                        const Shape& sample);

            /**
             * @brief Traces the mixture and returns its result alone.
             * @return The [N, T, dim] result's id, or @p x after trace.fail() when
             *         trace_experts() refuses.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces forward() over the newest slice.
             * @param sample Not read.
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample);

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<MoEFeedForward>,
                      "MoEFeedForward must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<MoEFeedForward>,
                      "MoEFeedForward must have an incremental form");

        /**
         * @brief Checks the counts of a mixture of experts.
         * @return InvalidDescription unless the three widths are positive and @p top_k is
         *         0 or at most @p experts.
         */
        [[nodiscard]] Status check_expert_counts(int64_t dim, int64_t hidden, int64_t experts,
                                                 int64_t top_k);

        /**
         * @brief Builds a three-port MoEFeedForward node from "experts" and "hidden", with
         *        optional "top_k", 0 when absent.
         * @param inputs One per-sample [T, C] shape; C becomes dim.
         * @return The node, with the ports "output", "balance" and "usage", or
         *         InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_moe_feed_forward_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Language::Blocks
} // namespace SushiAI
