/**************************************************************************/
/* moe_feed_forward.cpp                                                   */
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
 * @file moe_feed_forward.cpp
 * @brief Implements MoEFeedForward's trace, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/moe_feed_forward.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/traced.hpp>
#include "block_attributes.hpp"
#include "expert_node.hpp"
#include "sequence.hpp"
#include "top_scores.hpp"
#include "usage_balance.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "moe_feed_forward" row accepts. */
            constexpr std::string_view K_MOE_FEED_FORWARD_KEYS[] = {
                Detail::K_EXPERTS_KEY, Detail::K_HIDDEN_KEY, Detail::K_TOP_K_KEY};

            /**
             * @brief Emits the [R, E] @p gate with each column repeated @p hidden times.
             * @return The [R, E * hidden] result, whose entry e * hidden + j is gate[r, e],
             *         or the refusal of an emitter.
             */
            [[nodiscard]] Graph::Traced widen_gate(const Graph::GraphBuilder& builder,
                                                   Graph::ValueID gate, int64_t hidden)
            {
                // By value: every emission below can reallocate the value vector.
                const Shape shape = builder.graph().value(gate).shape;

                // A row is one image row of E pixels, which the upsample repeats in place.
                const Graph::Traced map = builder.reshape(gate, Shape{shape[0], 1, shape[1], 1});

                return builder.reshape(builder.upsample2d(map, 1, hidden),
                                       Shape{shape[0], shape[1] * hidden});
            }

            /** @brief Holds what the rows of a sequence give before the experts project. */
            struct GatedRows
            {
                Graph::ValueID gated = Graph::K_NO_VALUE;   /**< [R, E * hidden], gated. */
                Graph::ValueID gate = Graph::K_NO_VALUE;    /**< The [R, E] gate. */
                Graph::ValueID usage = Graph::K_NO_VALUE;   /**< The [1, E] mean gate. */
                Graph::ValueID balance = Graph::K_NO_VALUE; /**< The [1, 1] spread. */
            };

            /**
             * @brief Traces the router, the gate and the gated hidden values of every expert.
             * @param sequence The [N, T, dim] shape of @p x.
             * @return The four ids, or the failure of an emitter or a child.
             */
            [[nodiscard]] Result<GatedRows> trace_gated_rows(MoEFeedForward& block,
                                                             const NN::Trace& trace,
                                                             Graph::ValueID x,
                                                             const Shape& sequence)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY_ASSIGN(const Graph::ValueID rows,
                              builder.reshape(x, Shape{sequence[0] * sequence[1], sequence[2]}));
                const Graph::ValueID scores = NN::trace_layer(block.router, trace, rows);
                SA_TRY(trace.status());
                SA_TRY_ASSIGN(const Graph::ValueID probabilities, builder.softmax(scores));

                // The balance reads every probability, so a dropped expert is pulled back.
                SA_TRY_ASSIGN(const Detail::UsageBalance spread,
                              Detail::trace_usage_balance(builder, probabilities));

                SA_TRY_ASSIGN(const Graph::ValueID gate,
                              block.top_k == 0
                                  ? Graph::Traced(probabilities)
                                  : builder.softmax(
                                        Detail::keep_top_scores(builder, scores, block.top_k)));

                const Graph::ValueID widened = NN::trace_layer(block.expand, trace, rows);
                const Graph::ValueID active = NN::trace_layer(block.activation, trace, widened);
                SA_TRY(trace.status());
                SA_TRY_ASSIGN(const Graph::ValueID gated,
                              builder.mul(active, widen_gate(builder, gate, block.hidden)));

                return GatedRows{gated, gate, spread.usage, spread.balance};
            }
        } // namespace

        MoEFeedForward::MoEFeedForward(int64_t dim_in, int64_t hidden_in, int64_t experts_in,
                                       int64_t top_k_in) noexcept
            : dim(dim_in), hidden(hidden_in), experts(experts_in), top_k(top_k_in)
        {
            router.in_features = dim;
            router.out_features = experts;
            expand.in_features = dim;
            expand.out_features = experts * hidden;
            project.dim = dim;
            project.hidden = hidden;
            project.experts = experts;
        }

        Result<ExpertTrace> MoEFeedForward::trace_experts(const NN::Trace& trace,
                                                          Graph::ValueID x) noexcept
        {
            const Graph::GraphBuilder& builder = trace.builder();
            SA_TRY(SA_ANNOTATE(check_expert_counts(dim, hidden, experts, top_k),
                               "MoEFeedForward"));
            SA_TRY_ASSIGN(const Shape sequence,
                          Detail::checked_sequence(builder, x, dim, "MoEFeedForward"));

            SA_TRY_ASSIGN(const GatedRows rows, trace_gated_rows(*this, trace, x, sequence));
            SA_TRY_ASSIGN(const Graph::ValueID mixed,
                          project.mix(builder, rows.gated, rows.gate));
            SA_TRY_ASSIGN(const Graph::ValueID output, builder.reshape(mixed, sequence));

            return ExpertTrace{output, rows.balance, rows.usage};
        }

        Result<ExpertTrace> MoEFeedForward::trace_experts_incremental(const NN::Trace& trace,
                                                                      Graph::ValueID x,
                                                                      const Shape&) noexcept
        {
            return trace_experts(trace, x);
        }

        Graph::ValueID MoEFeedForward::forward(const NN::Trace& trace, Graph::ValueID x) noexcept
        {
            const Result<ExpertTrace> traced = trace_experts(trace, x);
            if (!traced.has_value())
            {
                trace.fail(traced.status());
                return x;
            }

            return traced.value().output;
        }

        Graph::ValueID MoEFeedForward::forward_incremental(const NN::Trace& trace,
                                                           Graph::ValueID x, const Shape&) noexcept
        {
            return forward(trace, x);
        }

        Result<Shape> MoEFeedForward::output_shape(const Shape& input) const noexcept
        {
            return input;
        }

        Status check_expert_counts(int64_t dim, int64_t hidden, int64_t experts, int64_t top_k) noexcept
        {
            SA_FAIL_IF(dim <= 0 || hidden <= 0 || experts <= 0, StatusCode::InvalidDescription,
                       "a mixture of experts needs positive widths and at least one expert; "
                       "got dim={} hidden={} experts={}",
                       dim, hidden, experts);
            SA_FAIL_IF(top_k < 0 || top_k > experts, StatusCode::InvalidDescription,
                       "a position keeps 0 experts, which means every one, or 1 to {} of "
                       "them; got top_k={}",
                       experts, top_k);

            return Status();
        }

        Result<std::unique_ptr<NN::Node>> make_moe_feed_forward_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(
                attributes, Detail::KeySpan(K_MOE_FEED_FORWARD_KEYS), context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_sequence(inputs, context));
            SA_TRY_ASSIGN(const int64_t experts,
                          Detail::read_positive_int(attributes, Detail::K_EXPERTS_KEY, context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int(attributes, Detail::K_HIDDEN_KEY, context));
            SA_TRY_ASSIGN(const int64_t top_k,
                          Detail::read_count_or(attributes, Detail::K_TOP_K_KEY, context,
                                                Detail::K_DEFAULT_TOP_K));
            SA_TRY(SA_ANNOTATE(check_expert_counts(input[1], hidden, experts, top_k), "{}",
                               context));

            return Detail::ExpertNode<MoEFeedForward>::create(
                MoEFeedForward(input[1], hidden, experts, top_k), input, experts);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
