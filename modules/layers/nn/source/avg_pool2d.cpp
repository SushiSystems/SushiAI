/**************************************************************************/
/* avg_pool2d.cpp                                                         */
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
 * @file avg_pool2d.cpp
 * @brief Implements tracing and shape inference for the 2-D average pooling layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/avg_pool2d.hpp>

#include <string_view>
#include <SushiAI/failure/error.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /**
             * @brief Computes the extents the window of @p layer yields over @p sample.
             * @param sample A per-sample shape; anything but an [H, W, C] shape that the window
             *        fits throws.
             * @return The number of window rows and of window columns.
             */
            [[nodiscard]] Graph::SpatialExtents fit_window(const AvgPool2d& layer,
                                                           const Shape& sample)
            {
                SA_THROW_IF(sample.rank() != 3,
                            "AvgPool2d needs a per-sample shape of [H, W, C]; got {}",
                            sample.to_string());

                Graph::SpatialExtents extents;
                try
                {
                    extents = Graph::window_extents(sample[0], sample[1], layer.geometry);
                }
                catch (const Error& error)
                {
                    SA_THROW_IF(true,
                                "AvgPool2d needs a legal window over its per-sample shape; "
                                "got {}: {}",
                                sample.to_string(), std::string_view(error.what()));
                }

                return extents;
            }
        } // namespace

        Graph::ValueID AvgPool2d::forward(const Graph::GraphBuilder& builder,
                                          Graph::ValueID x) const
        {
            const Shape& image = builder.graph().value(x).shape;
            SA_THROW_IF(image.rank() != 4,
                        "AvgPool2d needs a rank-4 [N, H, W, C] input; got {}", image.to_string());

            // Axes 1 and 2 hold H and W under NHWC only; the emitter refuses any other layout.
            if (builder.graph().value(x).layout == Graph::TensorLayout::NHWC)
                (void)fit_window(*this, Shape{image[1], image[2], image[3]});

            return builder.avgpool2d(x, geometry);
        }

        Shape AvgPool2d::output_shape(const Shape& input) const
        {
            const Graph::SpatialExtents out = fit_window(*this, input);

            return Shape{out.height, out.width, input[2]};
        }

    } // namespace NN
} // namespace SushiAI
