/**************************************************************************/
/* broadcast.hpp                                                          */
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
 * @file broadcast.hpp
 * @brief Declares the lowering rows of the broadcast arithmetic and the sum along an axis.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records the sum of a tensor and a vector broadcast along the node's side. */
        [[nodiscard]] Status lower_broadcast_add(const LoweringContext& context) noexcept;

        /** @brief Records the difference of a tensor and a vector broadcast along the node's side. */
        [[nodiscard]] Status lower_broadcast_sub(const LoweringContext& context) noexcept;

        /** @brief Records the product of a tensor and a vector broadcast along the node's side. */
        [[nodiscard]] Status lower_broadcast_mul(const LoweringContext& context) noexcept;

        /** @brief Records the quotient of a tensor and a vector broadcast along the node's side. */
        [[nodiscard]] Status lower_broadcast_div(const LoweringContext& context) noexcept;

        /** @brief Records the sum of a tensor down to a prefix or a suffix of its shape. */
        [[nodiscard]] Status lower_sum_along(const LoweringContext& context) noexcept;

        /** @brief Pairs each broadcast operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 5> K_BROADCAST_LOWERINGS = {{
            {Ops::K_BROADCAST_ADD.id, &lower_broadcast_add},
            {Ops::K_BROADCAST_SUB.id, &lower_broadcast_sub},
            {Ops::K_BROADCAST_MUL.id, &lower_broadcast_mul},
            {Ops::K_BROADCAST_DIV.id, &lower_broadcast_div},
            {Ops::K_SUM_ALONG.id, &lower_sum_along},
        }};
    } // namespace Graph
} // namespace SushiAI
