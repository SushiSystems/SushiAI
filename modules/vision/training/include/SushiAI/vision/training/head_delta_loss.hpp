/**************************************************************************/
/* head_delta_loss.hpp                                                    */
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
 * @file head_delta_loss.hpp
 * @brief Declares emit_head_delta_loss, the loss root whose gradient at a head is a given delta.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Emits the one-element value -scale * sum(delta * head).
     * @param builder Where the four nodes go: MUL, RESHAPE, SUM_ROWS and SCALE.
     * @param delta A leaf in the shape and element type of @p head; it takes no gradient.
     * @param scale The factor; the gradient at @p head is -scale * delta.
     * @return The id of the [1] result, a SCALE node's first output; InvalidDescription if
     *         @p head and @p delta differ in shape or element type.
     */
    [[nodiscard]] Graph::Traced emit_head_delta_loss(const Graph::GraphBuilder& builder,
                                                     Graph::ValueID head, Graph::ValueID delta,
                                                     double scale) noexcept;

} // namespace SushiAI::Vision::Training
