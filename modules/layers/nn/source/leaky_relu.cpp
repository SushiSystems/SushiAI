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

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Returns InvalidDescription unless the builder takes the slope of @p layer. */
            [[nodiscard]] Status check_slope(const LeakyReLU& layer) noexcept
            {
                SA_FAIL_IF(!Graph::is_legal_leaky_relu_slope(layer.slope),
                           StatusCode::InvalidDescription,
                           "LeakyReLU needs a slope in (0, 1); got {}", layer.slope);

                return Status();
            }
        } // namespace

        Graph::Traced LeakyReLU::forward(const Graph::GraphBuilder& builder,
                                         const Graph::Traced& input) const noexcept
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);
            SA_TRY(builder.observe(check_slope(*this)));

            return builder.leaky_relu(x, slope);
        }

        Result<Shape> LeakyReLU::output_shape(const Shape& input) const noexcept
        {
            SA_TRY(check_slope(*this));

            return input;
        }

    } // namespace NN
} // namespace SushiAI
