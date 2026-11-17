/**************************************************************************/
/* focus.hpp                                                              */
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
 * @file focus.hpp
 * @brief Declares Focus, the stem that stacks the four pixel phases into channels and convolves.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
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
        /**
         * @brief Halves H and W by stacking the four pixel phases into channels, then convolves.
         *
         * The phases stack in the order of YOLOX: x[::2, ::2], x[1::2, ::2], x[::2, 1::2] and
         * x[1::2, 1::2], that is top-left, bottom-left, top-right, bottom-right, so a
         * [N, H, W, C] input gives [N, H/2, W/2, 4C] before the convolution.
         */
        struct Focus
        {
            /** @brief Holds the convolution over the 4 * in_channels stacked phases. */
            Conv conv;

            /**
             * @brief Builds the convolution over four times @p in_channels.
             * @param kernel The convolution's window side; odd, so that it keeps H/2 and W/2.
             */
            Focus(int64_t in_channels, int64_t out_channels, int64_t kernel,
                  NN::NormOptions norm) noexcept;

            /** @brief Lists the child that holds state, under "conv". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("conv", &Focus::conv),
                };
            }

            /**
             * @brief Emits the phase stack: a [N, H, W, C] value to [N, H/2, W/2, 4C].
             * @param x An NHWC value whose H and W are even.
             * @return The stacked value's id, or InvalidDescription when @p x is not rank 4 or
             *         H or W is odd.
             */
            [[nodiscard]] static Result<Graph::ValueID> stack_phases(
                const Graph::GraphBuilder& builder, Graph::ValueID x) noexcept;

            /**
             * @brief Traces the phase stack, then the convolution, the norm and the SiLU.
             * @param x An NHWC [N, H, W, in_channels] value with H and W even.
             * @return The [N, H/2, W/2, out_channels] result's id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x) noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape with H and W even.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(NN::TraceModule<Focus>, "Focus must satisfy the TraceModule contract");

        /**
         * @brief Builds a Focus node from "filters", with optional "kernel", 3 when absent,
         *        "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape with H and W even.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit, or
         *         "kernel" is even.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_focus_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace Vision::Blocks
} // namespace SushiAI
