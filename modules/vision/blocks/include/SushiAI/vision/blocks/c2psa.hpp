/**************************************************************************/
/* c2psa.hpp                                                              */
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
 * @file c2psa.hpp
 * @brief Declares C2PSA, a split stem whose second half runs through PSABlocks.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/psa_block.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Splits a stem in two, chains PSABlocks off the second half, joins both. */
        struct C2PSA
        {
            /** @brief Holds the width of each half of the stem. */
            int64_t hidden = 0;

            /** @brief Holds the 1x1 convolution that makes the stem, 2 * hidden wide. */
            Conv cv1;

            /** @brief Holds the 1x1 convolution over the kept half and the attended half. */
            Conv cv2;

            /** @brief Holds the PSABlocks, chained in the order they were added. */
            NN::ModuleList<PSABlock> m;

            /**
             * @brief Builds the two convolutions and @p repeats PSABlocks; a bad width throws.
             * @param in_channels The input width; cv1 reads it.
             * @param out_channels The output width; cv2 writes it.
             * @param hidden_channels The width of each half; the stem is twice as wide.
             * @param heads The head count of every PSABlock.
             */
            C2PSA(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
                  int64_t repeats, int64_t heads, NN::NormOptions norm);

            /** @brief Lists the three children, under "cv1", "cv2" and "m". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("cv1", &C2PSA::cv1),
                    NN::child("cv2", &C2PSA::cv2),
                    NN::child("m", &C2PSA::m),
                };
            }

            /**
             * @brief Traces cv1, keeps the first half, chains the PSABlocks off the second half,
             *        then traces cv2 over both halves joined.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The [N, H, W, out_channels] result's id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<C2PSA>, "C2PSA must satisfy the TraceModule contract");

        /**
         * @brief Builds a C2PSA node from "filters" and "hidden", with optional "repeats", 1,
         *        "heads", max(hidden / 64, 1), "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C must equal "filters".
         * @return The node, or InvalidDescription when a key does not fit, or "filters" is not
         *         C, or "heads" does not split "hidden" into parts of two or more channels.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c2psa_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Vision::Blocks
} // namespace SushiAI
