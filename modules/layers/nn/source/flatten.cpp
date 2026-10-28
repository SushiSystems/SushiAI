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
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::Traced Flatten::forward(const Graph::GraphBuilder& builder,
                                       const Graph::Traced& input) const noexcept
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);

            const Graph::Value& value = builder.graph().value(x);
            SA_REFUSE_IF(builder, value.shape.rank() < 2,
                         "Flatten needs an input of rank 2 or more, [N, d1, ..., dk]; got {}",
                         value.shape.to_string());
            SA_REFUSE_IF(builder, value.layout != Graph::TensorLayout::NHWC,
                         "Flatten needs an NHWC input; got {} tagged NCHW. Cross with "
                         "layout_convert first (CV-3).",
                         value.shape.to_string());

            if (value.shape.rank() == 2)
                return x;

            const int64_t batch = value.shape[0];
            return builder.reshape(x, Shape{batch, value.shape.num_elements() / batch});
        }

        Result<Shape> Flatten::output_shape(const Shape& input) const noexcept
        {
            SA_FAIL_IF(input.rank() < 1, StatusCode::InvalidDescription,
                       "Flatten needs a per-sample shape of rank 1 or more; got {}",
                       input.to_string());

            return Shape{input.num_elements()};
        }

    } // namespace NN
} // namespace SushiAI
