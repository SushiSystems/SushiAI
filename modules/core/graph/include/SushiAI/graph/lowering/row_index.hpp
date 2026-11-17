/**************************************************************************/
/* row_index.hpp                                                          */
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
 * @file row_index.hpp
 * @brief Declares the lowering rows that move rows by an index and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records the copy of the table rows a list of indices names. */
        [[nodiscard]] Status lower_gather_rows(const LoweringContext& context) noexcept;

        /** @brief Records the sum of [N, D] rows into the table rows a list of indices names. */
        [[nodiscard]] Status lower_scatter_add_rows(const LoweringContext& context) noexcept;

        /** @brief Records the pick of one element per row, at the column an index names. */
        [[nodiscard]] Status lower_take_along(const LoweringContext& context) noexcept;

        /** @brief Records the write of one element per row, at the column an index names. */
        [[nodiscard]] Status lower_put_along(const LoweringContext& context) noexcept;

        /** @brief Records the column of the largest element of each row of a rank-2 operand. */
        [[nodiscard]] Status lower_argmax_rows(const LoweringContext& context) noexcept;

        /** @brief Pairs each row-index operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 5> K_ROW_INDEX_LOWERINGS = {{
            {Ops::K_GATHER_ROWS.id, &lower_gather_rows},
            {Ops::K_SCATTER_ADD_ROWS.id, &lower_scatter_add_rows},
            {Ops::K_TAKE_ALONG.id, &lower_take_along},
            {Ops::K_PUT_ALONG.id, &lower_put_along},
            {Ops::K_ARGMAX_ROWS.id, &lower_argmax_rows},
        }};
    } // namespace Graph
} // namespace SushiAI
