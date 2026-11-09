/**************************************************************************/
/* attention.hpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file attention.hpp
 * @brief Declares Attention, the multi-head self-attention over a feature map's positions.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /**
         * @brief Checks that @p head_count heads split @p channels into heads of at least two.
         * @param channels The width of the input and of the output.
         * @param head_count The number of heads.
         * @return A success, or InvalidDescription naming both counts; the constructor stops on
         *         the same refusal.
         */
        [[nodiscard]] Status check_attention_heads(int64_t channels, int64_t head_count) noexcept;

        /** @brief Holds the channels per attention head of a row that states no head count. */
        inline constexpr int64_t K_CHANNELS_PER_DEFAULT_HEAD = 64;

        /** @brief Returns max(hidden / K_CHANNELS_PER_DEFAULT_HEAD, 1), the head count of a row. */
        [[nodiscard]] constexpr int64_t default_attention_heads(int64_t hidden) noexcept
        {
            return hidden / K_CHANNELS_PER_DEFAULT_HEAD > 1 ? hidden / K_CHANNELS_PER_DEFAULT_HEAD
                                                            : 1;
        }

        /** @brief Attends every position of a feature map to every other, head by head. */
        struct Attention
        {
            /** @brief Holds the number of heads the channels are split into. */
            int64_t heads = 0;

            /** @brief Holds the value width of one head, channels / heads. */
            int64_t head_dim = 0;

            /** @brief Holds the query and key width of one head, head_dim / 2. */
            int64_t key_dim = 0;

            /** @brief Holds the 1x1 convolution to channels + 2 * heads * key_dim, head-major. */
            ConvLinear qkv;

            /** @brief Holds the 1x1 convolution over the attended value, with no activation. */
            ConvLinear proj;

            /** @brief Holds the depthwise 3x3 convolution over the values of all heads joined. */
            DWConvLinear pe;

            /**
             * @brief Builds the three convolutions over the width and the head count given.
             * @param channels The width of the input and of the output.
             * @param norm The epsilon and the momentum of the three normalisations.
             * @pre check_attention_heads() accepts @p channels and @p head_count; a width read
             *      from a document is checked before this call.
             */
            Attention(int64_t channels, int64_t head_count, NN::NormOptions norm) noexcept;

            /** @brief Lists the three children that hold state, under "qkv", "proj" and "pe". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("qkv", &Attention::qkv),
                    NN::child("proj", &Attention::proj),
                    NN::child("pe", &Attention::pe),
                };
            }

            /**
             * @brief Traces qkv, then per head the scaled query-key product, its softmax and the
             *        product with the values, then pe over the values added in, then proj.
             * @param x An NHWC [N, H, W, channels] value.
             * @return The [N, H, W, channels] result's id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x) noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces, which is the input's.
             * @param input A per-sample [H, W, channels] shape.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(NN::TraceModule<Attention>,
                      "Attention must satisfy the TraceModule contract");

    } // namespace Vision::Blocks
} // namespace SushiAI
