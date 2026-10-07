/**************************************************************************/
/* unflatten.cpp                                                          */
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
 * @file unflatten.cpp
 * @brief Implements the trace and the shape rule of the Unflatten layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/unflatten.hpp>

#include <cstdint>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Throws unless the height, width and channels of @p layer are all positive. */
            void require_positive_extents(const Unflatten& layer)
            {
                SA_THROW_IF(layer.height <= 0 || layer.width <= 0 || layer.channels <= 0,
                            "Unflatten needs positive extents; got height={} width={} channels={}",
                            layer.height, layer.width, layer.channels);
            }

            /**
             * @brief Counts the elements of one sample of @p layer.
             * @param layer A layer whose extents are positive.
             * @return height * width * channels.
             */
            int64_t count_sample_elements(const Unflatten& layer)
            {
                return layer.height * layer.width * layer.channels;
            }
        } // namespace

        Graph::ValueId Unflatten::forward(const Graph::GraphBuilder& builder,
                                          Graph::ValueId x) const
        {
            require_positive_extents(*this);
            const int64_t elements = count_sample_elements(*this);

            const Graph::Value& input = builder.graph().value(x);
            SA_THROW_IF(input.shape.rank() != 2,
                        "Unflatten needs a rank-2 [N, {}] input; got {}",
                        elements, input.shape.to_string());
            SA_THROW_IF(input.shape[1] != elements,
                        "Unflatten({}, {}, {}) needs an input whose inner extent is {}; got {}",
                        height, width, channels, elements, input.shape.to_string());
            SA_THROW_IF(input.layout != Graph::TensorLayout::NHWC,
                        "Unflatten needs an NHWC input; got {} tagged NCHW. Cross with "
                        "layout_convert first (CV-3).",
                        input.shape.to_string());

            // reshape() copies its operand's layout tag, which the check above fixed at NHWC.
            return builder.reshape(x, Shape{input.shape[0], height, width, channels});
        }

        Shape Unflatten::output_shape(const Shape& input) const
        {
            require_positive_extents(*this);
            const int64_t elements = count_sample_elements(*this);

            SA_THROW_IF(input.rank() != 1 || input[0] != elements,
                        "Unflatten({}, {}, {}) needs a per-sample shape of [{}]; got {}",
                        height, width, channels, elements, input.to_string());

            return Shape{height, width, channels};
        }

    } // namespace NN
} // namespace SushiAI
