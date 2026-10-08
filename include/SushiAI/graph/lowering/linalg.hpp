/**************************************************************************/
/* linalg.hpp                                                             */
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
 * @file linalg.hpp
 * @brief Declares the lowering rows of the linear-algebra operations and their table.
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
        [[nodiscard]] Status lower_matmul(const LoweringContext& context);

        /** @brief Records the sum of a rank-2 operand and a bias vector added to every row. */
        [[nodiscard]] Status lower_bias_add(const LoweringContext& context);

        /** @brief Records the column sums of a rank-2 operand. */
        [[nodiscard]] Status lower_sum_rows(const LoweringContext& context);

        /** @brief Pairs each linear-algebra operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 3> K_LINALG_LOWERINGS = {{
            {Ops::K_MATMUL.id, &lower_matmul},
            {Ops::K_BIAS_ADD.id, &lower_bias_add},
            {Ops::K_SUM_ROWS.id, &lower_sum_rows},
        }};
    } // namespace Graph
} // namespace SushiAI
