/**************************************************************************/
/* mish.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file mish.cpp
 * @brief Implements tracing and shape inference for the Mish layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/mish.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::Traced Mish::forward(const Graph::GraphBuilder& builder,
                                    const Graph::Traced& input) const noexcept
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);

            return builder.mul(x, builder.tanh(builder.softplus(x)));
        }

        Result<Shape> Mish::output_shape(const Shape& input) const noexcept
        {
            return input;
        }

    } // namespace NN
} // namespace SushiAI
