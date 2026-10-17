/**************************************************************************/
/* psa_block.hpp                                                          */
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
 * @file psa_block.hpp
 * @brief Declares PSABlock, attention and a feed-forward pair, each added to its input.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/attention.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Adds attention to its input, then adds a feed-forward pair to that sum. */
        struct PSABlock
        {
            /** @brief Holds the attention whose result is added to the input. */
            Attention attn;

            /**
             * @brief Holds the feed-forward pair: a Conv to twice the width, then a ConvLinear
             *        back to the width.
             */
            NN::Sequential<Conv, ConvLinear> ffn;

            /**
             * @brief Builds the attention and the feed-forward pair over @p channels.
             * @param channels The width of the input and of the output; the constructor throws
             *                 on a width Attention refuses.
             * @param heads The head count the attention splits @p channels into.
             * @param norm The epsilon and the momentum of every normalisation.
             */
            PSABlock(int64_t channels, int64_t heads, NN::NormOptions norm);

            /** @brief Lists the two children, under "attn" and "ffn". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("attn", &PSABlock::attn),
                    NN::child("ffn", &PSABlock::ffn),
                };
            }

            /**
             * @brief Traces attn over @p x and adds @p x to it, then ffn over that sum and adds
             *        the sum to it.
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

        static_assert(NN::TraceModule<PSABlock>, "PSABlock must satisfy the TraceModule contract");

    } // namespace Vision::Blocks
} // namespace SushiAI
