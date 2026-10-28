/**************************************************************************/
/* max_pool2d.cpp                                                         */
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
 * @file max_pool2d.cpp
 * @brief Implements tracing and shape inference for the 2-D max pooling layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/max_pool2d.hpp>

#include <limits>
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
             *         for anything but an [H, W, C] shape that the window fits.
             */
            [[nodiscard]] Result<Graph::SpatialExtents> fit_window(const MaxPool2d& layer,
                                                                   const Shape& sample) noexcept
            {
                SA_FAIL_IF(sample.rank() != 3, StatusCode::InvalidDescription,
                           "MaxPool2d needs a per-sample shape of [H, W, C]; got {}",
                           sample.to_string());
                SA_FAIL_IF(layer.trailing_pad_h < 0 || layer.trailing_pad_w < 0,
                           StatusCode::InvalidDescription,
                           "MaxPool2d needs a non-negative trailing padding; got {},{}",
                           layer.trailing_pad_h, layer.trailing_pad_w);

                constexpr int64_t K_LARGEST_EXTENT = std::numeric_limits<int64_t>::max();
                SA_FAIL_IF(layer.trailing_pad_h > K_LARGEST_EXTENT - sample[0] ||
                               layer.trailing_pad_w > K_LARGEST_EXTENT - sample[1],
                           StatusCode::InvalidDescription,
                           "MaxPool2d needs a trailing padding that keeps the input within 64 "
                           "bits; got {},{} over {}",
                           layer.trailing_pad_h, layer.trailing_pad_w, sample.to_string());

                // The trailing pad lengthens the input, so the window rule covers it.
                const Result<Graph::SpatialExtents> fitted =
                    Graph::window_extents(sample[0] + layer.trailing_pad_h,
                                          sample[1] + layer.trailing_pad_w, layer.geometry);
                if (!fitted.has_value())
                    return SA_ANNOTATE(fitted.status(),
                                       "MaxPool2d needs a legal window over its per-sample "
                                       "shape; got {}",
                                       sample.to_string());

                SA_TRY_ASSIGN(const Graph::SpatialExtents extents, fitted);

                // A window that starts at or past the input's end holds padding alone.
                const Graph::ConvGeometry& g = layer.geometry;
                SA_FAIL_IF((extents.height - 1) * g.stride_h - g.pad_top >= sample[0] ||
                               (extents.width - 1) * g.stride_w - g.pad_left >= sample[1],
                           StatusCode::InvalidDescription,
                           "MaxPool2d needs a trailing padding that leaves every window a tap "
                           "inside the input; got {},{} over {}",
                           layer.trailing_pad_h, layer.trailing_pad_w, sample.to_string());

                return extents;
            }
        } // namespace

        Graph::Traced MaxPool2d::forward(const Graph::GraphBuilder& builder,
                                         const Graph::Traced& input) const noexcept
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);

            const Shape& image = builder.graph().value(x).shape;
            SA_REFUSE_IF(builder, image.rank() != 4,
                         "MaxPool2d needs a rank-4 [N, H, W, C] input; got {}",
                         image.to_string());

            // Only NHWC reaches the fit; the emitter refuses any other layout first.
            if (builder.graph().value(x).layout != Graph::TensorLayout::NHWC)
            {
                SA_TRY_ASSIGN(const Graph::PoolResult pooled, builder.maxpool2d(x, geometry));
                return pooled.output;
            }

            const Shape sample{image[1], image[2], image[3]};
            SA_TRY_ASSIGN(const Graph::SpatialExtents out,
                          builder.observe(fit_window(*this, sample)));
            SA_TRY_ASSIGN(const Graph::PoolResult pooled,
                          builder.maxpool2d_with_extents(x, geometry, out.height, out.width));
            return pooled.output;
        }

        Result<Shape> MaxPool2d::output_shape(const Shape& input) const noexcept
        {
            SA_TRY_ASSIGN(const Graph::SpatialExtents out, fit_window(*this, input));

            return Shape{out.height, out.width, input[2]};
        }

    } // namespace NN
} // namespace SushiAI
