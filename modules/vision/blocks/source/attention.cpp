/**************************************************************************/
/* attention.cpp                                                          */
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
 * @file attention.cpp
 * @brief Implements Attention's constructor, its trace and its shape rule.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/attention.hpp>

#include <cmath>
#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/error.hpp>
#include <SushiAI/graph/builder.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Holds the side of the positional convolution's window. */
            constexpr int64_t K_POSITION_KERNEL = 3;

            /**
             * @brief Returns channels / head_count, the value width of one head.
             * @param channels Must be positive, a multiple of @p head_count, and hold
             *                 Detail::K_SMALLEST_HEAD_WIDTH channels per head; others throw.
             */
            [[nodiscard]] int64_t divide_into_heads(int64_t channels, int64_t head_count)
            {
                SA_THROW_IF(channels <= 0 || head_count <= 0 || channels % head_count != 0,
                            "Attention needs a positive width that its head count divides; got "
                            "{} channels and {} heads",
                            channels, head_count);
                SA_THROW_IF(channels / head_count < Detail::K_SMALLEST_HEAD_WIDTH,
                            "Attention needs at least {} channels per head, because the key "
                            "width is half of them; got {} channels and {} heads",
                            Detail::K_SMALLEST_HEAD_WIDTH, channels, head_count);

                return channels / head_count;
            }

            /** @brief Joins @p parts along the channel axis. */
            [[nodiscard]] Graph::ValueID join(const Graph::GraphBuilder& builder,
                                              const std::vector<Graph::ValueID>& parts)
            {
                return builder.concat_channels(
                    SushiRuntime::span<const Graph::ValueID>(parts.data(), parts.size()));
            }

        } // namespace

        Attention::Attention(int64_t channels, int64_t head_count, NN::NormOptions norm)
            : heads(head_count),
              head_dim(divide_into_heads(channels, head_count)),
              key_dim(head_dim / 2),
              qkv(channels, channels + 2 * heads * key_dim, Detail::K_POINTWISE,
                  Detail::K_POINTWISE, norm),
              proj(channels, channels, Detail::K_POINTWISE, Detail::K_POINTWISE, norm),
              pe(channels, channels, K_POSITION_KERNEL, Detail::K_POINTWISE, norm)
        {

        }

        Graph::ValueID Attention::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            const Graph::GraphBuilder& builder = trace.builder();
            const Graph::ValueID mixed = NN::trace_layer(qkv, trace, x);

            // The shape is held by value; each emitter below grows the table it would refer to.
            const Shape image = builder.graph().value(mixed).shape;
            const int64_t tokens = image[1] * image[2];
            const Shape key_tokens{image[0], tokens, key_dim};
            const Shape value_tokens{image[0], tokens, head_dim};
            const Shape value_map{image[0], image[1], image[2], head_dim};
            const int64_t head_width = 2 * key_dim + head_dim;
            const double scale = 1.0 / std::sqrt(static_cast<double>(key_dim));

            std::vector<Graph::ValueID> attended;
            std::vector<Graph::ValueID> values;
            attended.reserve(static_cast<std::size_t>(heads));
            values.reserve(static_cast<std::size_t>(heads));
            for (int64_t head = 0; head < heads; ++head)
            {
                // qkv is head-major: each head holds its queries, then its keys, then its values.
                const int64_t offset = head * head_width;
                const Graph::ValueID q = builder.scale(
                    builder.reshape(builder.slice_channels(mixed, offset, key_dim), key_tokens),
                    scale);
                const Graph::ValueID k = builder.reshape(
                    builder.slice_channels(mixed, offset + key_dim, key_dim), key_tokens);
                const Graph::ValueID v =
                    builder.slice_channels(mixed, offset + 2 * key_dim, head_dim);

                const Graph::ValueID scores = builder.softmax(builder.matmul(q, k, false, true));
                const Graph::ValueID weighed =
                    builder.matmul(scores, builder.reshape(v, value_tokens));

                attended.push_back(builder.reshape(weighed, value_map));
                values.push_back(v);
            }

            const Graph::ValueID joined = join(builder, attended);
            const Graph::ValueID position = NN::trace_layer(pe, trace, join(builder, values));

            return NN::trace_layer(proj, trace, builder.add(joined, position));
        }

        Shape Attention::output_shape(const Shape& input) const
        {
            // qkv refuses an input that is not [H, W, channels] and keeps H and W.
            const Shape mixed = qkv.output_shape(input);

            return proj.output_shape(Shape{mixed[0], mixed[1], heads * head_dim});
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
