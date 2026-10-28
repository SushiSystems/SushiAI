/**************************************************************************/
/* segment_head.hpp                                                       */
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
 * @file segment_head.hpp
 * @brief Declares SegmentHead, which maps a feature map to class logits at the input size.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/upsample2d.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Scores every class at every position, then repeats each score over a block. */
        struct SegmentHead
        {
            /** @brief Holds the 3x3 convolution with its normalisation and its SiLU. */
            Conv cv1;

            /** @brief Holds the biased, unpadded 1x1 convolution to one channel per class. */
            NN::Conv2d cv2;

            /** @brief Holds the nearest-neighbour upsample by the factor given. */
            NN::Upsample2d up;

            /**
             * @brief Builds the two convolutions and the upsample.
             * @param hidden The width between the two convolutions.
             * @param scale How many times larger H and W leave than they enter; at least 1.
             */
            SegmentHead(int64_t in_channels, int64_t hidden, int64_t classes, int64_t scale,
                        NN::NormOptions norm) noexcept;

            /** @brief Lists the two children that hold state, under "cv1" and "cv2". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("cv1", &SegmentHead::cv1),
                    NN::child("cv2", &SegmentHead::cv2),
                };
            }

            /**
             * @brief Traces cv1, cv2, then the upsample.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The [N, H * scale, W * scale, classes] logits' id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x) noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(NN::TraceModule<SegmentHead>,
                      "SegmentHead must satisfy the TraceModule contract");

        /**
         * @brief Builds a SegmentHead node from "classes", with optional "hidden", the input
         *        width when absent, "scale", 1 when absent, "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_segment_head_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace Vision::Blocks
} // namespace SushiAI
