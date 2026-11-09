/**************************************************************************/
/* depthwise_conv2d.hpp                                                   */
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
 * @file depthwise_conv2d.hpp
 * @brief Declares the depthwise 2-D convolution layer that traces into graph IR.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Checks that a depthwise layer can keep its channel count from @p in_channels to
         *        @p out_channels.
         * @return A success, or InvalidDescription naming both counts; same_padded() stops on
         *         the same refusal.
         */
        [[nodiscard]] Status check_depthwise_channels(int64_t in_channels,
                                                      int64_t out_channels) noexcept;

        /** @brief Filters each channel of an NHWC activation with its own window of taps. */
        struct DepthwiseConv2d
        {
            /** @brief Holds the channel count of the input, which the result keeps. */
            int64_t channels = 0;

            /** @brief Holds the window; its top and left pads apply to both sides of an axis. */
            Graph::ConvGeometry geometry{};

            /** @brief Holds the [kernel_h, kernel_w, channels] taps, one window per channel. */
            Parameter weight{};

            /** @brief Lists this layer's parameters for the generic walk. */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    field("weight", &DepthwiseConv2d::weight),
                };
            }

            /**
             * @brief Builds a bias-free depthwise convolution padded by kernel / 2 on every side.
             * @pre check_depthwise_channels() accepts @p in_channels and @p out_channels.
             * @return A layer with dilation 1 whose geometry output_shape() and forward() check.
             */
            [[nodiscard]] static DepthwiseConv2d same_padded(int64_t in_channels,
                                                             int64_t out_channels,
                                                             int64_t kernel, int64_t stride) noexcept;

            /**
             * @brief Traces the convolution, declaring its weight on the first call.
             * @param input An NHWC [N, H, W, channels] value that the window fits.
             * @return The [N, out_h, out_w, channels] result; the failure of @p input,
             *         or InvalidDescription for an input the layer refuses.
             */
            Graph::Traced forward(const Graph::GraphBuilder& builder,
                                  const Graph::Traced& input) noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, channels] shape that the window fits.
             * @return The per-sample [out_h, out_w, channels] shape, or
             *         InvalidDescription for a geometry or a shape the window does not fit.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(Module<DepthwiseConv2d>, "DepthwiseConv2d must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
