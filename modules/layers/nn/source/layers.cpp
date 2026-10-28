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
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Returns InvalidDescription unless both feature counts of @p layer are > 0. */
            [[nodiscard]] Status check_features(const Linear& layer) noexcept
            {
                SA_FAIL_IF(layer.in_features <= 0 || layer.out_features <= 0,
                           StatusCode::InvalidDescription,
                           "Linear needs positive feature counts; got in={} out={}",
                           layer.in_features, layer.out_features);

                return Status();
            }
        } // namespace

        Graph::Traced Linear::forward(const Graph::GraphBuilder& builder,
                                      const Graph::Traced& input)
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);
            SA_TRY(builder.observe(check_features(*this)));

            const Shape& input_shape = builder.graph().value(x).shape;
            SA_REFUSE_IF(builder, input_shape.rank() != 2,
                         "Linear needs a rank-2 input; got {}", input_shape.to_string());
            SA_REFUSE_IF(builder, input_shape[1] != in_features,
                         "Linear({}, {}) was given an input of {}; its inner extent must be {}",
                         in_features, out_features, input_shape.to_string(), in_features);

            const DType dtype = builder.graph().value(x).dtype;

            SA_TRY_ASSIGN(const Graph::ValueID w,
                          weight.declare(builder, Shape{out_features, in_features},
                                         ParameterRole::Weight, weight_init, in_features,
                                         out_features, "weight", dtype));

            // X [N, in] against W [out, in] transposed: the flag, never a view.
            Graph::Traced y = builder.matmul(x, w, false, true);

            if (use_bias)
            {
                SA_TRY_ASSIGN(const Graph::ValueID b,
                              bias.declare(builder, Shape{out_features}, ParameterRole::Bias,
                                           in_features, out_features, "bias", dtype));
                y = builder.bias_add(y, b);
            }

            return y;
        }

        Result<Shape> Linear::output_shape(const Shape& input) const
        {
            SA_TRY(check_features(*this));
            SA_FAIL_IF(input.rank() != 1 || input[0] != in_features, StatusCode::InvalidDescription,
                       "Linear({}, {}) was given a per-sample shape of {}; it needs [{}]",
                       in_features, out_features, input.to_string(), in_features);

            return Shape{out_features};
        }

    } // namespace NN
} // namespace SushiAI
