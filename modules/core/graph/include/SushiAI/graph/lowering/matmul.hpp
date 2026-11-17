/**************************************************************************/
/* matmul.hpp                                                             */
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
 * @file matmul.hpp
 * @brief Declares the lowering rows of the matrix product family and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records the product of two matrices, each transposed if its flag is set. */
        [[nodiscard]] Status lower_matmul(const LoweringContext& context) noexcept;

        /** @brief Records the sum of a rank-2 operand and a bias vector added to every row. */
        [[nodiscard]] Status lower_bias_add(const LoweringContext& context) noexcept;

        /** @brief Records the column sums of a rank-2 operand. */
        [[nodiscard]] Status lower_sum_rows(const LoweringContext& context) noexcept;

        /** @brief Pairs each matrix product operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 3> K_MATMUL_LOWERINGS = {{
            {Ops::K_MATMUL.id, &lower_matmul},
            {Ops::K_BIAS_ADD.id, &lower_bias_add},
            {Ops::K_SUM_ROWS.id, &lower_sum_rows},
        }};
    } // namespace Graph
} // namespace SushiAI
