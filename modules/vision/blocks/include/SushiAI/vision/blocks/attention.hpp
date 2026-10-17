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
 * @file attention.hpp
 * @brief Declares Attention, the multi-head self-attention over a feature map's positions.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
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
             * @param channels A positive multiple of @p head_count with two or more channels
             *                 per head; the constructor throws otherwise.
             * @param norm The epsilon and the momentum of the three normalisations.
             */
            Attention(int64_t channels, int64_t head_count, NN::NormOptions norm);

            /** @brief Lists the three children that hold state, under "qkv", "proj" and "pe". */
            static constexpr auto fields()
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
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Computes the per-sample shape forward() produces, which is the input's.
             * @param input A per-sample [H, W, channels] shape.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<Attention>,
                      "Attention must satisfy the TraceModule contract");

    } // namespace Vision::Blocks
} // namespace SushiAI
