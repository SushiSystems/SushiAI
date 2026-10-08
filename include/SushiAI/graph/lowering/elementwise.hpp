/**************************************************************************/
/* elementwise.hpp                                                        */
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
 * @file elementwise.hpp
 * @brief Declares the lowering rows of the elementwise operations and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records the elementwise sum of two equally shaped operands. */
        [[nodiscard]] Status lower_add(const LoweringContext& context);

        /** @brief Records the elementwise difference of two equally shaped operands. */
        [[nodiscard]] Status lower_sub(const LoweringContext& context);

        /** @brief Records the elementwise product of two equally shaped operands. */
        [[nodiscard]] Status lower_mul(const LoweringContext& context);

        /** @brief Records the elementwise quotient of two equally shaped operands. */
        [[nodiscard]] Status lower_div(const LoweringContext& context);

        /** @brief Records the elementwise minimum of two equally shaped operands. */
        [[nodiscard]] Status lower_minimum(const LoweringContext& context);

        /** @brief Records the elementwise maximum of two equally shaped operands. */
        [[nodiscard]] Status lower_maximum(const LoweringContext& context);

        /** @brief Records the product of one operand and the scalar attribute. */
        [[nodiscard]] Status lower_scale(const LoweringContext& context);

        /** @brief Records the product of one operand and a one-element factor read at run time. */
        [[nodiscard]] Status lower_dynamic_scale(const LoweringContext& context);

        /** @brief Records the elementwise exponential of one operand. */
        [[nodiscard]] Status lower_exp(const LoweringContext& context);

        /** @brief Records the elementwise absolute value of one operand. */
        [[nodiscard]] Status lower_abs(const LoweringContext& context);

        /** @brief Records the elementwise arctangent of one operand. */
        [[nodiscard]] Status lower_atan(const LoweringContext& context);

        /** @brief Records the elementwise less-or-equal mask of two operands. */
        [[nodiscard]] Status lower_less_equal(const LoweringContext& context);

        /** @brief Records the selection of one of two operands by a condition operand. */
        [[nodiscard]] Status lower_select(const LoweringContext& context);

        /** @brief Records a fill of the result with zeros. */
        [[nodiscard]] Status lower_zero(const LoweringContext& context);

        /** @brief Records a fill of the result with the scalar attribute. */
        [[nodiscard]] Status lower_constant(const LoweringContext& context);

        /** @brief Records the in-place accumulation of one operand into another. */
        [[nodiscard]] Status lower_accumulate(const LoweringContext& context);

        /** @brief Records the precision conversion of one operand into the result. */
        [[nodiscard]] Status lower_cast(const LoweringContext& context);

        /** @brief Pairs each elementwise operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 17> K_ELEMENTWISE_LOWERINGS = {{
            {Ops::K_ADD.id, &lower_add},
            {Ops::K_SUB.id, &lower_sub},
            {Ops::K_MUL.id, &lower_mul},
            {Ops::K_DIV.id, &lower_div},
            {Ops::K_MINIMUM.id, &lower_minimum},
            {Ops::K_MAXIMUM.id, &lower_maximum},
            {Ops::K_SCALE.id, &lower_scale},
            {Ops::K_DYNAMIC_SCALE.id, &lower_dynamic_scale},
            {Ops::K_EXP.id, &lower_exp},
            {Ops::K_ABS.id, &lower_abs},
            {Ops::K_ATAN.id, &lower_atan},
            {Ops::K_LESS_EQUAL.id, &lower_less_equal},
            {Ops::K_SELECT.id, &lower_select},
            {Ops::K_ZERO.id, &lower_zero},
            {Ops::K_CONSTANT.id, &lower_constant},
            {Ops::K_ACCUMULATE.id, &lower_accumulate},
            {Ops::K_CAST.id, &lower_cast},
        }};
    } // namespace Graph
} // namespace SushiAI
