/**************************************************************************/
/* head_delta_loss.cpp                                                    */
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
 * @file head_delta_loss.cpp
 * @brief Implements emit_head_delta_loss over MUL, RESHAPE, SUM_ROWS and SCALE.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/head_delta_loss.hpp>

#include <cstdint>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI::Vision::Training
{
    Graph::Traced emit_head_delta_loss(const Graph::GraphBuilder& builder, Graph::ValueID head,
                                       Graph::ValueID delta, double scale) noexcept
    {
        SA_TRY_ASSIGN(const Graph::ValueID product, builder.mul(delta, head));

        // By value: the emissions below can reallocate the graph's value vector.
        const std::int64_t elements = builder.graph().value(product).shape.num_elements();

        return builder.scale(builder.sum_rows(builder.reshape(product, Shape{elements, 1})),
                             -scale);
    }

} // namespace SushiAI::Vision::Training
