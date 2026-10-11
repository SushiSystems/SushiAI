/**************************************************************************/
/* expert_transformer_block.cpp                                           */
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
 * @file expert_transformer_block.cpp
 * @brief Implements ExpertTransformerBlock's trace, its incremental form and its shape rule.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/expert_transformer_block.hpp>

#include <utility>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include "normalised_branch.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /**
             * @brief Traces the attention branch through @p trace_child, then the experts
             *        through @p trace_experts, and adds each to what it read.
             * @param trace_child Called as trace_child(layer, value); returns the result's id.
             * @param trace_experts Called as trace_experts(value); returns Result<ExpertTrace>.
             * @return The block's three values, or the failure of a child.
             */
            template<typename TraceChild, typename TraceExperts>
            [[nodiscard]] Result<ExpertTrace> trace_block(ExpertTransformerBlock& block,
                                                          const NN::Trace& trace,
                                                          Graph::ValueID x,
                                                          TraceChild&& trace_child,
                                                          TraceExperts&& trace_experts)
            {
                SA_TRY_ASSIGN(const Graph::ValueID attended,
                              Detail::add_normalised_branch(block.norm1, block.attn, trace, x,
                                                            trace_child));

                const Graph::ValueID normalised = trace_child(block.norm2, attended);
                SA_TRY(trace.status());

                SA_TRY_ASSIGN(ExpertTrace routed, trace_experts(normalised));
                SA_TRY_ASSIGN(routed.output, NN::Detail::bridge_throwing([&]()
                {
                    return trace.builder().add(attended, routed.output);
                }));

                return routed;
            }

            /** @brief Returns the output of @p traced, or records its failure and returns @p x. */
            [[nodiscard]] Graph::ValueID output_or_fail(const NN::Trace& trace,
                                                        const Result<ExpertTrace>& traced,
                                                        Graph::ValueID x)
            {
                if (traced.has_value())
                    return traced.value().output;

                // A child that refused has already recorded its failure, which fail() keeps.
                trace.fail(traced.status());
                return x;
            }
        } // namespace

        ExpertTransformerBlock::ExpertTransformerBlock(int64_t heads,
                                                       MoEFeedForward feed_forward,
                                                       AttentionMask mask, double epsilon,
                                                       AttentionPositions positions)
            : attn(feed_forward.dim, heads, mask, positions), ff(std::move(feed_forward))
        {
            norm1.dim = ff.dim;
            norm1.epsilon = epsilon;
            norm2.dim = ff.dim;
            norm2.epsilon = epsilon;
        }

        Result<ExpertTrace> ExpertTransformerBlock::trace_experts(const NN::Trace& trace,
                                                                  Graph::ValueID x)
        {
            return trace_block(
                *this, trace, x,
                [&trace](auto& child, Graph::ValueID value)
                {
                    return NN::trace_layer(child, trace, value);
                },
                [this, &trace](Graph::ValueID value)
                {
                    return ff.trace_experts(trace, value);
                });
        }

        Result<ExpertTrace> ExpertTransformerBlock::trace_experts_incremental(
            const NN::Trace& trace, Graph::ValueID x, const Shape& sample)
        {
            return trace_block(
                *this, trace, x,
                [&trace, &sample](auto& child, Graph::ValueID value)
                {
                    return NN::trace_layer_incremental(child, trace, value, sample);
                },
                [this, &trace, &sample](Graph::ValueID value)
                {
                    return ff.trace_experts_incremental(trace, value, sample);
                });
        }

        Graph::ValueID ExpertTransformerBlock::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            return output_or_fail(trace, trace_experts(trace, x), x);
        }

        Graph::ValueID ExpertTransformerBlock::forward_incremental(const NN::Trace& trace,
                                                                   Graph::ValueID x,
                                                                   const Shape& sample)
        {
            return output_or_fail(trace, trace_experts_incremental(trace, x, sample), x);
        }

        Shape ExpertTransformerBlock::output_shape(const Shape& input) const
        {
            return input;
        }

    } // namespace Language::Blocks
} // namespace SushiAI
