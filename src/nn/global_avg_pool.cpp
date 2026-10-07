/**************************************************************************/
/* global_avg_pool.cpp                                                    */
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
 * @file global_avg_pool.cpp
 * @brief Implements the tracing and shape inference of the global average pooling layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/global_avg_pool.hpp>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::ValueId GlobalAvgPool::forward(const Graph::GraphBuilder& builder,
                                              Graph::ValueId x) const
        {
            const Shape& image = builder.graph().value(x).shape;
            SA_THROW_IF(image.rank() != 4,
                        "GlobalAvgPool needs a rank-4 [N, H, W, C] input; got {}",
                        image.to_string());

            return builder.spatial_mean(x);
        }

        Shape GlobalAvgPool::output_shape(const Shape& input) const
        {
            SA_THROW_IF(input.rank() != 3,
                        "GlobalAvgPool needs a per-sample shape of [H, W, C]; got {}",
                        input.to_string());

            return Shape{input[2]};
        }

    } // namespace NN
} // namespace SushiAI
