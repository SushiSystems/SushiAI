/**************************************************************************/
/* leaky_relu.hpp                                                         */
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
 * @file leaky_relu.hpp
 * @brief Declares the LeakyReLU activation layer.
 * @author Mustafa Garip
 */

#pragma once

#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Applies x where x is positive and slope * x elsewhere, elementwise. */
        struct LeakyReLU
        {
            /** @brief Holds the factor applied to negative inputs; must lie in (0, 1). */
            double slope = 0.01;

            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the activation over an input of any rank.
             * @return The result, shaped and laid out like @p input; the failure of @p input,
             *         or InvalidDescription for an input the layer refuses.
             */
            Graph::Traced forward(const Graph::GraphBuilder& builder,
                                  const Graph::Traced& input) const noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @return The shape of @p input, which the activation keeps, or
             *         InvalidDescription for a slope the layer refuses.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(Module<LeakyReLU>, "LeakyReLU must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
