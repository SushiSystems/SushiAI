/**************************************************************************/
/* focus.cpp                                                              */
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
 * @file focus.cpp
 * @brief Implements Focus's phase stack, its trace and the factory that builds its node.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/focus.hpp>

#include <array>
#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "focus" row accepts. */
            constexpr std::string_view K_FOCUS_KEYS[] = {
                "filters", "kernel", Detail::K_NORM_EPSILON_KEY, Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Holds the convolution's window side in a row that states none. */
            constexpr int64_t K_DEFAULT_FOCUS_KERNEL = 3;

            /** @brief Holds the side of the window that cuts the image into pixel phases. */
            constexpr int64_t K_PHASE_SIDE = 2;

            /** @brief Holds how many pixel phases the stack joins. */
            constexpr int64_t K_PHASE_COUNT = K_PHASE_SIDE * K_PHASE_SIDE;

            /**
             * @brief Lists which window tap holds each phase of the stack, in the stack's order.
             *
             * The window reads its taps row by row, so tap 0 is top-left, 1 top-right, 2
             * bottom-left and 3 bottom-right. The stack is top-left, bottom-left, top-right,
             * bottom-right.
             */
            constexpr std::array<int64_t, K_PHASE_COUNT> K_PHASE_TAPS = {0, 2, 1, 3};

            /** @brief Returns the non-overlapping 2x2 window that cuts out the phases. */
            [[nodiscard]] Graph::ConvGeometry phase_geometry() noexcept
            {
                Graph::ConvGeometry geometry;
                geometry.kernel_h = K_PHASE_SIDE;
                geometry.kernel_w = K_PHASE_SIDE;
                geometry.stride_h = K_PHASE_SIDE;
                geometry.stride_w = K_PHASE_SIDE;
                return geometry;
            }

            /** @brief Tells whether @p extent holds a whole number of phase windows. */
            [[nodiscard]] bool is_even(int64_t extent) noexcept
            {
                return extent >= K_PHASE_SIDE && extent % K_PHASE_SIDE == 0;
            }

        } // namespace

        Focus::Focus(int64_t in_channels, int64_t out_channels, int64_t kernel,
                     NN::NormOptions norm) noexcept
            : conv(K_PHASE_COUNT * in_channels, out_channels, kernel, 1, norm)
        {

        }

        Result<Graph::ValueID> Focus::stack_phases(const Graph::GraphBuilder& builder,
                                                   Graph::ValueID x) noexcept
        {
            // The shape is held by value; each emitter below grows the table it would refer to.
            const Shape image = builder.graph().value(x).shape;
            SA_FAIL_IF(image.rank() != 4 || !is_even(image[1]) || !is_even(image[2]),
                       StatusCode::InvalidDescription,
                       "focus needs a rank-4 [N, H, W, C] input with an even H and W; it was "
                       "given {}",
                       image.to_string());

            const int64_t channels = image[3];
            SA_TRY_ASSIGN(const Graph::ValueID windows, builder.im2col(x, phase_geometry()));
            SA_TRY_ASSIGN(const Graph::ValueID taps,
                          builder.reshape(windows,
                                          Shape{image[0], image[1] / K_PHASE_SIDE,
                                                image[2] / K_PHASE_SIDE,
                                                K_PHASE_COUNT * channels}));

            std::vector<Graph::ValueID> phases;
            phases.reserve(static_cast<std::size_t>(K_PHASE_COUNT));
            for (const int64_t tap : K_PHASE_TAPS)
            {
                SA_TRY_ASSIGN(const Graph::ValueID phase,
                              builder.slice_channels(taps, tap * channels, channels));
                phases.push_back(phase);
            }

            SA_TRY_ASSIGN(const Graph::ValueID stacked,
                          builder.concat_channels(SushiRuntime::span<const Graph::ValueID>(
                              phases.data(), phases.size())));

            return stacked;
        }

        Graph::ValueID Focus::forward(const NN::Trace& trace, Graph::ValueID x) noexcept
        {
            const Graph::ValueID stacked =
                trace.value_or_fail(stack_phases(trace.builder(), x), x);
            if (!trace.status().is_ok())
                return x;

            return NN::trace_layer(conv, trace, stacked);
        }

        Result<Shape> Focus::output_shape(const Shape& input) const noexcept
        {
            SA_FAIL_IF(input.rank() != 3 || !is_even(input[0]) || !is_even(input[1]),
                       StatusCode::InvalidDescription,
                       "focus needs a per-sample [H, W, C] input with an even H and W; it was "
                       "given {}",
                       input.to_string());

            return conv.output_shape(Shape{input[0] / K_PHASE_SIDE, input[1] / K_PHASE_SIDE,
                                           K_PHASE_COUNT * input[2]});
        }

        Result<std::unique_ptr<NN::Node>> make_focus_node(NN::ShapeSpan inputs,
                                                          const Config::Value& attributes,
                                                          std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_FOCUS_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const int64_t filters,
                          Detail::read_positive_int(attributes, "filters", context));
            SA_TRY_ASSIGN(const int64_t kernel,
                          Detail::read_positive_int_or(attributes, "kernel", context,
                                                       K_DEFAULT_FOCUS_KERNEL));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            SA_FAIL_IF(kernel % 2 == 0, StatusCode::InvalidDescription,
                       "{} 'kernel' must be odd, so that the convolution keeps H/2 and W/2; it "
                       "is {}",
                       context, kernel);

            return Detail::wrap_block(Focus(input[2], filters, kernel, norm), input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
