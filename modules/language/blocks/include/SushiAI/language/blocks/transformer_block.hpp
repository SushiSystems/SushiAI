/**************************************************************************/
/* transformer_block.hpp                                                  */
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
 * @file transformer_block.hpp
 * @brief Declares TransformerBlock, attention and a feed-forward, each normalised first and
 *        added to its input.
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
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/feed_forward.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Adds attention over the normalised input, then a feed-forward over that sum. */
        struct TransformerBlock
        {
            /** @brief Holds the normalisation the attention reads. */
            NN::LayerNorm norm1{};

            /** @brief Holds the attention whose result is added to the input. */
            CausalSelfAttention attn{};

            /** @brief Holds the normalisation the feed-forward reads. */
            NN::LayerNorm norm2{};

            /** @brief Holds the feed-forward whose result is added to the first sum. */
            FeedForward ff{};

            /** @brief Builds a block of no width, which no input fits. */
            TransformerBlock() = default;

            /**
             * @brief Builds the four children over @p dim; forward() checks the widths.
             * @param hidden The width between the two linear layers of the feed-forward.
             * @param mask Causal for a decoder, None for an encoder.
             * @param epsilon The epsilon of both normalisations.
             * @param positions What the attention takes; Rotary needs dim / heads even.
             */
            TransformerBlock(
                int64_t dim, int64_t heads, int64_t hidden, AttentionMask mask, double epsilon,
                AttentionPositions positions = AttentionPositions::Learned) noexcept;

            /** @brief Lists the four children, under "norm1", "attn", "norm2" and "ff". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("norm1", &TransformerBlock::norm1),
                    NN::child("attn", &TransformerBlock::attn),
                    NN::child("norm2", &TransformerBlock::norm2),
                    NN::child("ff", &TransformerBlock::ff),
                };
            }

            /**
             * @brief Traces x + attn(norm1(x)), then that sum plus ff(norm2(sum)).
             * @param x A [N, T, dim] value of a real floating type.
             * @return The [N, T, dim] result's id, or @p x once a child has failed the trace.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces the incremental form of each child in the order forward() has.
             * @param x The [1, 1, dim] slice of one sequence, of a real floating type.
             * @param sample The per-sample [T, dim] shape of the ordinary form.
             * @return The [1, 1, dim] result's id, or @p x once a child has failed the trace.
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample);

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<TransformerBlock>,
                      "TransformerBlock must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<TransformerBlock>,
                      "TransformerBlock must have an incremental form");

        /**
         * @brief Builds a TransformerBlock node from "heads", with optional "hidden",
         *        "causal", "ln_epsilon" and "positions", which its attention takes.
         * @param inputs One per-sample [T, C] shape; C becomes the width.
         * @param attributes "experts" above 0, with optional "top_k", builds the three-port
         *        node of an ExpertTransformerBlock in its place.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_transformer_block_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Language::Blocks
} // namespace SushiAI
