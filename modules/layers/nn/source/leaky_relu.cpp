/**************************************************************************/
/* leaky_relu.cpp                                                         */
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
 * @file leaky_relu.cpp
 * @brief Implements tracing and shape inference for the LeakyReLU layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/leaky_relu.hpp>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Throws unless the builder accepts the slope of @p layer. */
            void require_legal_slope(const LeakyReLU& layer)
            {
                SA_THROW_IF(!Graph::is_legal_leaky_relu_slope(layer.slope),
                            "LeakyReLU needs a slope in (0, 1); got {}", layer.slope);
            }
        } // namespace

        Graph::ValueId LeakyReLU::forward(const Graph::GraphBuilder& builder,
                                          Graph::ValueId x) const
        {
            require_legal_slope(*this);

            return builder.leaky_relu(x, slope);
        }

        Shape LeakyReLU::output_shape(const Shape& input) const
        {
            require_legal_slope(*this);

            return input;
        }

    } // namespace NN
} // namespace SushiAI
