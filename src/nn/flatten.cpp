/**************************************************************************/
/* flatten.cpp                                                            */
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
 * @file flatten.cpp
 * @brief Implements the Flatten layer's tracing and shape rule.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/flatten.hpp>

#include <cstdint>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::ValueId Flatten::forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const
        {
            const Graph::Value& input = builder.graph().value(x);
            SA_THROW_IF(input.shape.rank() < 2,
                        "Flatten needs an input of rank 2 or more, [N, d1, ..., dk]; got {}",
                        input.shape.to_string());
            SA_THROW_IF(input.layout != Graph::TensorLayout::NHWC,
                        "Flatten needs an NHWC input; got {} tagged NCHW. Cross with "
                        "layout_convert first (CV-3).",
                        input.shape.to_string());

            if (input.shape.rank() == 2)
                return x;

            const int64_t batch = input.shape[0];
            return builder.reshape(x, Shape{batch, input.shape.num_elements() / batch});
        }

        Shape Flatten::output_shape(const Shape& input) const
        {
            SA_THROW_IF(input.rank() < 1,
                        "Flatten needs a per-sample shape of rank 1 or more; got {}",
                        input.to_string());

            return Shape{input.num_elements()};
        }

    } // namespace NN
} // namespace SushiAI
