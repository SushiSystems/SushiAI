/**************************************************************************/
/* conv2d.cpp                                                             */
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
 * @file conv2d.cpp
 * @brief Implements tracing and shape inference for the 2-D convolution layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/conv2d.hpp>

#include <cstdint>
#include <string>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiLog/text/text_format.hpp>
#include "grouped_convolution.hpp"
#include "trailing_window.hpp"

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Tells whether @p layer filters each channel with its own taps. */
            [[nodiscard]] bool is_depthwise(const Conv2d& layer) noexcept
            {
                return layer.groups > 1 && layer.groups == layer.in_channels &&
                       layer.groups == layer.out_channels;
            }

            /**
             * @brief Computes the extents the window of @p layer yields over @p sample.
             * @param sample A per-sample shape.
             * @return The number of window rows and of window columns, or InvalidDescription
             *         for non-positive channel counts or for anything but an
             *         [H, W, in_channels] shape that the window fits.
             */
            [[nodiscard]] Result<Graph::SpatialExtents> fit_window(const Conv2d& layer,
                                                                   const Shape& sample) noexcept
            {
                SA_FAIL_IF(layer.in_channels <= 0 || layer.out_channels <= 0,
                           StatusCode::InvalidDescription,
                           "Conv2d needs positive channel counts; got in={} out={}",
                           layer.in_channels, layer.out_channels);
                SA_FAIL_IF(layer.groups < 1 || layer.in_channels % layer.groups != 0 ||
                               layer.out_channels % layer.groups != 0,
                           StatusCode::InvalidDescription,
                           "Conv2d({}, {}) needs groups that divide both channel counts; got {}",
                           layer.in_channels, layer.out_channels, layer.groups);
                SA_FAIL_IF(is_depthwise(layer) &&
                               (layer.trailing_pad_h != 0 || layer.trailing_pad_w != 0),
                           StatusCode::InvalidDescription,
                           "Conv2d({}, {}) with {} groups is depthwise, which has no trailing "
                           "padding; got {},{}",
                           layer.in_channels, layer.out_channels, layer.groups,
                           layer.trailing_pad_h, layer.trailing_pad_w);
                SA_FAIL_IF(sample.rank() != 3 || sample[2] != layer.in_channels,
                           StatusCode::InvalidDescription,
                           "Conv2d({}, {}) needs a per-sample shape of [H, W, {}]; got {}",
                           layer.in_channels, layer.out_channels, layer.in_channels,
                           sample.to_string());

                const std::string name = SushiLog::format_text(
                    "Conv2d({}, {})", layer.in_channels, layer.out_channels);
                SA_TRY(Detail::require_trailing_pad(name, layer.trailing_pad_h,
                                                    layer.trailing_pad_w));
                return Detail::trailing_window_extents(name, sample, layer.geometry,
                                                       layer.trailing_pad_h, layer.trailing_pad_w);
            }

            /**
             * @brief Traces the depthwise form of @p layer: the taps over @p x, then the bias.
             * @param window The taps of one channel, which is both fan of the weight.
             */
            [[nodiscard]] Graph::Traced trace_depthwise(Conv2d& layer,
                                                        const Graph::GraphBuilder& builder,
                                                        Graph::ValueID x, DType dtype,
                                                        int64_t window) noexcept
            {
                // Each channel reads and writes only its own taps, so both fans are the window.
                SA_TRY_ASSIGN(const Graph::ValueID w,
                              layer.weight.declare(builder,
                                                   Shape{layer.geometry.kernel_h,
                                                         layer.geometry.kernel_w,
                                                         layer.in_channels},
                                                   ParameterRole::Weight, window, window, "weight",
                                                   dtype));
                layer.weight.set_layout(StateLayout::DepthwiseKernelLast);
                SA_TRY_ASSIGN(const Graph::ValueID filtered,
                              builder.depthwise_conv2d(x, w, layer.geometry));
                if (!layer.use_bias)
                    return filtered;

                SA_TRY_ASSIGN(const Graph::ValueID b,
                              layer.bias.declare(builder, Shape{layer.out_channels},
                                                 ParameterRole::Bias,
                                                 ParameterInit::constant_value(layer.bias_init),
                                                 window, window, "bias", dtype));
                return builder.broadcast_add(filtered, b, Graph::BroadcastSide::Trailing);
            }
        } // namespace

        Conv2d Conv2d::same_padded(int64_t in_channels, int64_t out_channels, int64_t kernel,
                                   int64_t stride) noexcept
        {
            Conv2d layer;
            layer.in_channels = in_channels;
            layer.out_channels = out_channels;
            layer.geometry.kernel_h = kernel;
            layer.geometry.kernel_w = kernel;
            layer.geometry.stride_h = stride;
            layer.geometry.stride_w = stride;
            layer.geometry.pad_top = kernel / 2;
            layer.geometry.pad_left = kernel / 2;
            layer.use_bias = false;

            return layer;
        }

        Graph::Traced Conv2d::forward(const Graph::GraphBuilder& builder,
                                      const Graph::Traced& input) noexcept
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);

            // The shape and the type are held by value; declaring a parameter grows their table.
            const Shape image = builder.graph().value(x).shape;
            const DType dtype = builder.graph().value(x).dtype;

            SA_REFUSE_IF(builder, image.rank() != 4,
                         "Conv2d({}, {}) needs a rank-4 [N, H, W, {}] input; got {}",
                         in_channels, out_channels, in_channels, image.to_string());
            SA_REFUSE_IF(builder, builder.graph().value(x).layout != Graph::TensorLayout::NHWC,
                         "Conv2d needs an NHWC input; got {} tagged NCHW. Cross with "
                         "layout_convert first (CV-3).",
                         image.to_string());
            SA_TRY_ASSIGN(const Graph::SpatialExtents out,
                          builder.observe(fit_window(*this, Shape{image[1], image[2], image[3]})));

            const int64_t window = geometry.kernel_h * geometry.kernel_w;
            if (is_depthwise(*this))
                return trace_depthwise(*this, builder, x, dtype, window);

            const int64_t group_inputs = in_channels / groups;
            const int64_t fan_in = window * group_inputs;
            const int64_t fan_out = window * out_channels / groups;

            SA_TRY_ASSIGN(const Graph::ValueID w,
                          weight.declare(builder,
                                         Shape{out_channels, geometry.kernel_h, geometry.kernel_w,
                                               group_inputs},
                                         ParameterRole::Weight, fan_in, fan_out, "weight",
                                         dtype));
            weight.set_layout(StateLayout::KernelLast);

            Graph::ValueID b = Graph::K_NO_VALUE;
            if (use_bias)
            {
                SA_TRY_ASSIGN(b, bias.declare(builder, Shape{out_channels}, ParameterRole::Bias,
                                              ParameterInit::constant_value(bias_init), fan_in,
                                              fan_out, "bias", dtype));
            }

            if (groups > 1)
                return Detail::trace_grouped_convolution(builder, x, w, b, geometry, out.height,
                                                         out.width, groups);

            return builder.conv2d_with_extents(x, w, b, geometry, out.height, out.width);
        }

        Result<Shape> Conv2d::output_shape(const Shape& input) const noexcept
        {
            SA_TRY_ASSIGN(const Graph::SpatialExtents out, fit_window(*this, input));

            return Shape{out.height, out.width, out_channels};
        }

    } // namespace NN
} // namespace SushiAI
