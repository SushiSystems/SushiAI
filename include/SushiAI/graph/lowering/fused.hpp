/**************************************************************************/
/* fused.hpp                                                              */
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
 * @file fused.hpp
 * @brief Declares the lowering rows of the fused operations and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records a GEMM with a per-column bias. */
        [[nodiscard]] Status lower_gemm_bias(const LoweringContext& context);

        /** @brief Records a GEMM with a per-column bias and a ReLU. */
        [[nodiscard]] Status lower_gemm_bias_relu(const LoweringContext& context);

        /** @brief Records a GEMM with a per-column bias and a logistic sigmoid. */
        [[nodiscard]] Status lower_gemm_bias_sigmoid(const LoweringContext& context);

        /** @brief Records a GEMM with a per-column bias and a tanh. */
        [[nodiscard]] Status lower_gemm_bias_tanh(const LoweringContext& context);

        /** @brief Records a GEMM with a per-column bias and a GELU; result 1 holds the sum. */
        [[nodiscard]] Status lower_gemm_bias_gelu(const LoweringContext& context);

        /** @brief Records the scaled difference of an operand's exponential and a target. */
        [[nodiscard]] Status lower_exp_sub_scale(const LoweringContext& context);

        /** @brief Pairs each fused operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 6> K_FUSED_LOWERINGS = {{
            {Ops::K_GEMM_BIAS.id, &lower_gemm_bias},
            {Ops::K_GEMM_BIAS_RELU.id, &lower_gemm_bias_relu},
            {Ops::K_GEMM_BIAS_SIGMOID.id, &lower_gemm_bias_sigmoid},
            {Ops::K_GEMM_BIAS_TANH.id, &lower_gemm_bias_tanh},
            {Ops::K_GEMM_BIAS_GELU.id, &lower_gemm_bias_gelu},
            {Ops::K_EXP_SUB_SCALE.id, &lower_exp_sub_scale},
        }};
    } // namespace Graph
} // namespace SushiAI
