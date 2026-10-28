/**************************************************************************/
/* transformer_block.cpp                                                  */
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
 * @file transformer_block.cpp
 * @brief Implements TransformerBlock's trace, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/transformer_block.hpp>

#include <utility>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/language/blocks/expert_transformer_block.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include "block_attributes.hpp"
#include "expert_node.hpp"
#include "normalised_branch.hpp"
#include "positions_attribute.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "transformer_block" row accepts. */
            constexpr std::string_view K_TRANSFORMER_KEYS[] = {
                Detail::K_HEADS_KEY, Detail::K_HIDDEN_KEY, Detail::K_CAUSAL_KEY,
                Detail::K_LAYER_NORM_EPSILON_KEY, Detail::K_EXPERTS_KEY, Detail::K_TOP_K_KEY,
                Detail::K_POSITIONS_KEY};

            /**
             * @brief Traces both branches of the block through @p trace_child.
             * @return The result's id, or the failure a child recorded.
             */
            template<typename TraceChild>
            [[nodiscard]] Result<Graph::ValueID> trace_block(TransformerBlock& block,
                                                             const NN::Trace& trace,
                                                             Graph::ValueID x,
                                                             TraceChild&& trace_child)
            {
                SA_TRY_ASSIGN(const Graph::ValueID attended,
                              Detail::add_normalised_branch(block.norm1, block.attn, trace, x,
                                                            trace_child));

                return Detail::add_normalised_branch(block.norm2, block.ff, trace, attended,
                                                     trace_child);
            }
        } // namespace

        TransformerBlock::TransformerBlock(int64_t dim, int64_t heads, int64_t hidden,
                                           AttentionMask mask, double epsilon,
                                           AttentionPositions positions) noexcept
            : attn(dim, heads, mask, positions), ff(dim, hidden)
        {
            norm1.dim = dim;
            norm1.epsilon = epsilon;
            norm2.dim = dim;
            norm2.epsilon = epsilon;
        }

        Graph::ValueID TransformerBlock::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            const Result<Graph::ValueID> traced =
                trace_block(*this, trace, x, [&trace](auto& child, Graph::ValueID value)
            {
                return NN::trace_layer(child, trace, value);
            });

            // A child that refused has already recorded its failure, which fail() keeps.
            return trace.value_or_fail(traced, x);
        }

        Graph::ValueID TransformerBlock::forward_incremental(const NN::Trace& trace,
                                                             Graph::ValueID x,
                                                             const Shape& sample)
        {
            const Result<Graph::ValueID> traced =
                trace_block(*this, trace, x, [&trace, &sample](auto& child, Graph::ValueID value)
            {
                return NN::trace_layer_incremental(child, trace, value, sample);
            });

            // A child that refused has already recorded its failure, which fail() keeps.
            return trace.value_or_fail(traced, x);
        }

        Result<Shape> TransformerBlock::output_shape(const Shape& input) const
        {
            return input;
        }

        Result<std::unique_ptr<NN::Node>> make_transformer_block_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_TRANSFORMER_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_sequence(inputs, context));

            const int64_t dim = input[1];
            SA_TRY_ASSIGN(const int64_t heads,
                          Detail::read_positive_int(attributes, Detail::K_HEADS_KEY, context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int_or(attributes, Detail::K_HIDDEN_KEY, context,
                                                       Detail::K_DEFAULT_HIDDEN_RATIO * dim));
            SA_TRY_ASSIGN(const bool causal,
                          Detail::read_bool_or(attributes, Detail::K_CAUSAL_KEY, context,
                                               Detail::K_DEFAULT_CAUSAL));
            SA_TRY_ASSIGN(const double epsilon,
                          Detail::read_layer_norm_epsilon(attributes, context));
            SA_TRY_ASSIGN(const int64_t experts,
                          Detail::read_count_or(attributes, Detail::K_EXPERTS_KEY, context,
                                                Detail::K_NO_EXPERTS));
            SA_TRY_ASSIGN(const int64_t top_k,
                          Detail::read_count_or(attributes, Detail::K_TOP_K_KEY, context,
                                                Detail::K_DEFAULT_TOP_K));
            SA_TRY_ASSIGN(const AttentionPositions positions,
                          Detail::read_positions(attributes, context));
            SA_TRY(SA_ANNOTATE(check_attention_widths(dim, heads), "{}", context));
            SA_TRY(SA_ANNOTATE(check_attention_positions(dim, heads, positions), "{}",
                               context));

            const AttentionMask mask = causal ? AttentionMask::Causal : AttentionMask::None;
            if (experts == Detail::K_NO_EXPERTS)
            {
                SA_FAIL_IF(top_k != Detail::K_DEFAULT_TOP_K, StatusCode::InvalidDescription,
                           "{} states '{}' as {} and no '{}'; a block without experts keeps "
                           "none",
                           context, Detail::K_TOP_K_KEY, top_k, Detail::K_EXPERTS_KEY);

                return Detail::wrap_block(
                    TransformerBlock(dim, heads, hidden, mask, epsilon, positions), input);
            }

            SA_TRY(SA_ANNOTATE(check_expert_counts(dim, hidden, experts, top_k), "{}", context));

            ExpertTransformerBlock block(heads, MoEFeedForward(dim, hidden, experts, top_k),
                                         mask, epsilon, positions);

            return Detail::ExpertNode<ExpertTransformerBlock>::create(std::move(block), input,
                                                                      experts);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
