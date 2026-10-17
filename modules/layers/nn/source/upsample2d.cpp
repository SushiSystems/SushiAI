/**************************************************************************/
/* upsample2d.cpp                                                         */
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
 * @file upsample2d.cpp
 * @brief Implements the tracing and shape rule of the nearest-neighbour upsampling layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/upsample2d.hpp>

#include <cstdint>
#include <string_view>
#include <SushiAI/failure/error.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /**
             * @brief Computes the extents the scale of @p layer yields over @p sample.
             * @param sample A per-sample shape; anything but an [H, W, C] shape that the scale
             *        can take throws.
             * @return The scaled height and width.
             */
            [[nodiscard]] Graph::SpatialExtents scale_extents(const Upsample2d& layer,
                                                              const Shape& sample)
            {
                SA_THROW_IF(sample.rank() != 3,
                            "Upsample2d needs a per-sample shape of [H, W, C]; got {}",
                            sample.to_string());

                Graph::SpatialExtents extents;
                try
                {
                    extents = Graph::upsampled_extents(sample[0], sample[1], layer.scale,
                                                       layer.scale);
                }
                catch (const Error& error)
                {
                    SA_THROW_IF(true,
                                "Upsample2d needs a scale its per-sample shape can take; "
                                "got {} with scale={}: {}",
                                sample.to_string(), layer.scale, std::string_view(error.what()));
                }

                return extents;
            }
        } // namespace

        Graph::ValueID Upsample2d::forward(const Graph::GraphBuilder& builder,
                                           Graph::ValueID x) const
        {
            const Shape& image = builder.graph().value(x).shape;
            SA_THROW_IF(image.rank() != 4,
                        "Upsample2d needs a rank-4 [N, H, W, C] input; got {}",
                        image.to_string());
            (void)scale_extents(*this, Shape{image[1], image[2], image[3]});

            return builder.upsample2d(x, scale, scale);
        }

        Shape Upsample2d::output_shape(const Shape& input) const
        {
            const Graph::SpatialExtents scaled = scale_extents(*this, input);

            return Shape{scaled.height, scaled.width, input[2]};
        }

    } // namespace NN
} // namespace SushiAI
