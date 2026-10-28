/**************************************************************************/
/* depthwise_conv2d.cpp                                                   */
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
 * @file depthwise_conv2d.cpp
 * @brief Implements tracing and shape inference for the depthwise 2-D convolution layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/depthwise_conv2d.hpp>

#include <cstdint>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/error.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /**
             * @brief Computes the extents the window of @p layer yields over @p sample.
             * @param sample A per-sample shape.
             * @return The number of window rows and of window columns, or InvalidDescription
             *         for a non-positive channel count, a shape other than [H, W, channels] or
             *         a window window_extents refuses.
             */
            [[nodiscard]] Result<Graph::SpatialExtents> fit_window(const DepthwiseConv2d& layer,
                                                                   const Shape& sample) noexcept
            {
                SA_FAIL_IF(layer.channels <= 0, StatusCode::InvalidDescription,
                           "DepthwiseConv2d needs a positive channel count; got {}",
                           layer.channels);
                SA_FAIL_IF(sample.rank() != 3 || sample[2] != layer.channels,
                           StatusCode::InvalidDescription,
                           "DepthwiseConv2d({}) needs a per-sample shape of [H, W, {}]; got {}",
                           layer.channels, layer.channels, sample.to_string());

                return Graph::window_extents(sample[0], sample[1], layer.geometry);
            }
        } // namespace

        DepthwiseConv2d DepthwiseConv2d::same_padded(int64_t in_channels, int64_t out_channels,
                                                     int64_t kernel, int64_t stride)
        {
            SA_THROW_IF(in_channels != out_channels,
                        "DepthwiseConv2d keeps its channel count; got in={} and out={}",
                        in_channels, out_channels);

            DepthwiseConv2d layer;
            layer.channels = in_channels;
            layer.geometry.kernel_h = kernel;
            layer.geometry.kernel_w = kernel;
            layer.geometry.stride_h = stride;
            layer.geometry.stride_w = stride;
            layer.geometry.pad_top = kernel / 2;
            layer.geometry.pad_left = kernel / 2;

            return layer;
        }

        Graph::Traced DepthwiseConv2d::forward(const Graph::GraphBuilder& builder,
                                               const Graph::Traced& input)
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);

            // The shape and the type are held by value; declaring a parameter grows their table.
            const Shape image = builder.graph().value(x).shape;
            const DType dtype = builder.graph().value(x).dtype;

            SA_REFUSE_IF(builder, image.rank() != 4,
                         "DepthwiseConv2d({}) needs a rank-4 [N, H, W, {}] input; got {}",
                         channels, channels, image.to_string());
            SA_REFUSE_IF(builder, builder.graph().value(x).layout != Graph::TensorLayout::NHWC,
                         "DepthwiseConv2d needs an NHWC input; got {}. Cross with layout_convert "
                         "first.",
                         image.to_string());
            SA_TRY(builder.observe(fit_window(*this, Shape{image[1], image[2], image[3]})));

            // Each channel reads and writes only its own taps, so both fans are the window.
            const int64_t window = geometry.kernel_h * geometry.kernel_w;
            SA_TRY_ASSIGN(const Graph::ValueID w,
                          weight.declare(builder,
                                         Shape{geometry.kernel_h, geometry.kernel_w, channels},
                                         ParameterRole::Weight, window, window, "weight", dtype));

            return builder.depthwise_conv2d(x, w, geometry);
        }

        Result<Shape> DepthwiseConv2d::output_shape(const Shape& input) const
        {
            SA_TRY_ASSIGN(const Graph::SpatialExtents out, fit_window(*this, input));

            return Shape{out.height, out.width, channels};
        }

    } // namespace NN
} // namespace SushiAI
