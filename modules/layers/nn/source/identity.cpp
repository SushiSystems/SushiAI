/**************************************************************************/
/* identity.cpp                                                           */
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
 * @file identity.cpp
 * @brief Implements tracing and shape inference for the Identity layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/identity.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::Traced Identity::forward(const Graph::GraphBuilder&,
                                        const Graph::Traced& input) const noexcept
        {
            return input;
        }

        Result<Shape> Identity::output_shape(const Shape& input) const noexcept
        {
            return input;
        }

    } // namespace NN
} // namespace SushiAI
