/**************************************************************************/
/* c3k.hpp                                                                */
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
 * @file c3k.hpp
 * @brief Declares C3k, a chain of bottlenecks on one branch joined with a second branch.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Narrows the input twice, chains bottlenecks off one half, joins both halves. */
        struct C3k
        {
            /** @brief Holds the 1x1 convolution that feeds the bottleneck chain. */
            Conv cv1;

            /** @brief Holds the 1x1 convolution of the branch that skips the chain. */
            Conv cv2;

            /** @brief Holds the 1x1 convolution over both branches joined. */
            Conv cv3;

            /** @brief Holds the chain; each bottleneck keeps the hidden width with 3x3 windows. */
            NN::ModuleList<Bottleneck> m;

            /**
             * @brief Builds the three convolutions and @p repeats bottlenecks.
             * @param out_channels The output width; the hidden width is half of it, rounded down.
             * @param shortcut Asks every bottleneck of the chain to add its input.
             */
            C3k(int64_t in_channels, int64_t out_channels, int64_t repeats, bool shortcut,
                NN::NormOptions norm);

            /** @brief Lists the four children, under "cv1", "cv2", "cv3" and "m". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("cv1", &C3k::cv1),
                    NN::child("cv2", &C3k::cv2),
                    NN::child("cv3", &C3k::cv3),
                    NN::child("m", &C3k::m),
                };
            }

            /**
             * @brief Traces cv1 and the chain, then cv2, then cv3 over the two results joined.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The [N, H, W, out_channels] result's id.
             */
            Graph::ValueId forward(const NN::Trace& trace, Graph::ValueId x);

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<C3k>, "C3k must satisfy the TraceModule contract");

        /**
         * @brief Builds a C3k node from "filters", with optional "repeats", 1 when absent,
         *        "shortcut", true when absent, "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c3k_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Vision::Blocks
} // namespace SushiAI
