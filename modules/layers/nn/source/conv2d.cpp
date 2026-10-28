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
 * @file conv2d.cpp
 * @brief Implements tracing and shape inference for the 2-D convolution layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/conv2d.hpp>

#include <cstdint>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>

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
                SA_FAIL_IF(sample.rank() != 3 || sample[2] != layer.in_channels,
                           StatusCode::InvalidDescription,
                           "Conv2d({}, {}) needs a per-sample shape of [H, W, {}]; got {}",
                           layer.in_channels, layer.out_channels, layer.in_channels,
                           sample.to_string());

                Result<Graph::SpatialExtents> extents =
                    Graph::window_extents(sample[0], sample[1], layer.geometry);
                if (!extents.has_value())
                    return SA_ANNOTATE(extents.status(),
                                       "Conv2d({}, {}) needs a legal window over its per-sample "
                                       "shape; got {}",
                                       layer.in_channels, layer.out_channels, sample.to_string());

                return extents;
            }
        } // namespace

        Conv2d Conv2d::same_padded(int64_t in_channels, int64_t out_channels, int64_t kernel,
                                   int64_t stride)
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
                                      const Graph::Traced& input)
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
            SA_TRY(builder.observe(fit_window(*this, Shape{image[1], image[2], image[3]})));

            const int64_t window = geometry.kernel_h * geometry.kernel_w;
            const int64_t fan_in = window * in_channels;
            const int64_t fan_out = window * out_channels;

            SA_TRY_ASSIGN(const Graph::ValueID w,
                          weight.declare(builder,
                                         Shape{out_channels, geometry.kernel_h, geometry.kernel_w,
                                               in_channels},
                                         ParameterRole::Weight, fan_in, fan_out, "weight",
                                         dtype));

            Graph::ValueID b = Graph::K_NO_VALUE;
            if (use_bias)
            {
                SA_TRY_ASSIGN(b, bias.declare(builder, Shape{out_channels}, ParameterRole::Bias,
                                              ParameterInit::constant_value(bias_init), fan_in,
                                              fan_out, "bias", dtype));
            }

            return builder.conv2d(x, w, b, geometry);
        }

        Result<Shape> Conv2d::output_shape(const Shape& input) const
        {
            SA_TRY_ASSIGN(const Graph::SpatialExtents out, fit_window(*this, input));

            return Shape{out.height, out.width, out_channels};
        }

    } // namespace NN
} // namespace SushiAI
