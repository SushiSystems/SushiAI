/**************************************************************************/
/* bottleneck.hpp                                                         */
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
 * @file bottleneck.hpp
 * @brief Declares Bottleneck, two convolutions with an optional add of the input.
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
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Applies two convolutions and adds the input when the widths allow it. */
        struct Bottleneck
        {
            /** @brief Holds the side of both windows when the caller states neither. */
            static constexpr int64_t K_DEFAULT_KERNEL = 3;

            /** @brief Holds the convolution from the input width to the hidden width. */
            Conv cv1;

            /** @brief Holds the convolution from the hidden width to the output width. */
            Conv cv2;

            /** @brief Tells whether forward() adds its input to the second convolution's result. */
            bool adds_input = false;

            /**
             * @brief Builds both convolutions at stride 1.
             * @param shortcut Asks for the add, which is made only when in_channels equals
             *                 out_channels.
             * @param kernel1 The side of cv1's window; odd, so that H and W are kept.
             * @param kernel2 The side of cv2's window; odd, so that H and W are kept.
             */
            Bottleneck(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
                       bool shortcut, NN::NormOptions norm, int64_t kernel1 = K_DEFAULT_KERNEL,
                       int64_t kernel2 = K_DEFAULT_KERNEL);

            /** @brief Lists the two children, under "cv1" and "cv2". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("cv1", &Bottleneck::cv1),
                    NN::child("cv2", &Bottleneck::cv2),
                };
            }

            /**
             * @brief Traces cv1, then cv2, then the add of @p x when adds_input holds.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The [N, H, W, out_channels] result's id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape.
             */
            Result<Shape> output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<Bottleneck>,
                      "Bottleneck must satisfy the TraceModule contract");

        /**
         * @brief Builds a Bottleneck node from "filters" and "hidden", with optional "shortcut",
         *        true when absent, and the optional "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_bottleneck_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Vision::Blocks
} // namespace SushiAI
