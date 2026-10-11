/**************************************************************************/
/* feed_forward.cpp                                                       */
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
 * @file feed_forward.cpp
 * @brief Implements FeedForward's trace, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/feed_forward.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include "block_attributes.hpp"
#include "sequence.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "feed_forward" row accepts. */
            constexpr std::string_view K_FEED_FORWARD_KEYS[] = {Detail::K_HIDDEN_KEY};

            /** @brief Traces the block, or returns why @p x or @p block does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_feed_forward(FeedForward& block,
                                                                    const NN::Trace& trace,
                                                                    Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_FAIL_IF(block.dim <= 0 || block.hidden <= 0, StatusCode::InvalidDescription,
                           "FeedForward needs positive widths; got dim={} hidden={}", block.dim,
                           block.hidden);
                SA_TRY_ASSIGN(const Shape sequence,
                              Detail::checked_sequence(builder, x, block.dim, "FeedForward"));

                return Detail::over_rows(builder, x, sequence, block.dim,
                                         [&](Graph::ValueID rows)
                {
                    const Graph::ValueID widened = NN::trace_layer(block.expand, trace, rows);
                    const Graph::ValueID active =
                        NN::trace_layer(block.activation, trace, widened);

                    return NN::trace_layer(block.project, trace, active);
                });
            }
        } // namespace

        FeedForward::FeedForward(int64_t dim_in, int64_t hidden_in) noexcept
            : dim(dim_in), hidden(hidden_in)
        {
            expand.in_features = dim;
            expand.out_features = hidden;
            project.in_features = hidden;
            project.out_features = dim;
        }

        Graph::ValueID FeedForward::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            return trace.value_or_fail(trace_feed_forward(*this, trace, x), x);
        }

        Graph::ValueID FeedForward::forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                                        const Shape&)
        {
            return forward(trace, x);
        }

        Shape FeedForward::output_shape(const Shape& input) const
        {
            return input;
        }

        Result<std::unique_ptr<NN::Node>> make_feed_forward_node(NN::ShapeSpan inputs,
                                                                 const Config::Value& attributes,
                                                                 std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_FEED_FORWARD_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_sequence(inputs, context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int(attributes, Detail::K_HIDDEN_KEY, context));

            return Detail::wrap_block(FeedForward(input[1], hidden), input);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
