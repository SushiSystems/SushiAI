/**************************************************************************/
/* layers.cpp                                                             */
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
 * @file layers.cpp
 * @brief Implements forward tracing logic for core neural network layers.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layers.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::ValueId Linear::forward(const Graph::GraphBuilder& builder, Graph::ValueId x)
        {
            SA_THROW_IF(in_features <= 0 || out_features <= 0,
                        "Linear needs positive feature counts; got in={} out={}",
                        in_features, out_features);

            const Shape& input_shape = builder.graph().value(x).shape;
            SA_THROW_IF(input_shape.rank() != 2,
                        "Linear needs a rank-2 input; got {}", input_shape.to_string());
            SA_THROW_IF(input_shape[1] != in_features,
                        "Linear({}, {}) was given an input of {}; its inner extent must be {}",
                        in_features, out_features, input_shape.to_string(), in_features);

            const DType dtype = builder.graph().value(x).dtype;

            const Graph::ValueId w = weight.declare(builder, Shape{out_features, in_features},
                                                    ParameterRole::WEIGHT,
                                                    in_features, out_features, "weight", dtype);

            // X [N, in] against W [out, in] transposed: the flag, never a view.
            Graph::ValueId y = builder.matmul(x, w, false, true);

            if (use_bias)
            {
                const Graph::ValueId b = bias.declare(builder, Shape{out_features},
                                                      ParameterRole::BIAS,
                                                      in_features, out_features, "bias", dtype);
                y = builder.bias_add(y, b);
            }

            return y;
        }

    } // namespace NN
} // namespace SushiAI
