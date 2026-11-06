/**************************************************************************/
/* identity.hpp                                                           */
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
 * @file identity.hpp
 * @brief Declares the Identity layer, which stands where a block takes no activation.
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
        /** @brief Passes its input through and emits no node. */
        struct Identity
        {
            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces nothing.
             * @return @p input itself, or the failure of @p input.
             */
            Graph::Traced forward(const Graph::GraphBuilder& builder,
                                  const Graph::Traced& input) const noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @return The shape of @p input, which the layer keeps.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(Module<Identity>, "Identity must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
