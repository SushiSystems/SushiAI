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
 * @brief Declares backward differentiation rules for linear algebra operations.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief Differentiates 2D matrix multiplication across transposition configurations. */
        void matmul_backward(const GradientTape& tape);
        /** @brief Differentiates row-vector bias addition: identity in X, row-summed in bias. */
        void bias_add_backward(const GradientTape& tape);
        /** @brief Differentiates matrix row summation by broadcasting incoming gradients. */
        void sum_rows_backward(const GradientTape& tape);

        /** @brief Differentiates a row gather by scatter-adding its gradient into the table's. */
        void gather_rows_backward(const GradientTape& tape);
        /** @brief Differentiates a row scatter-add by gathering its gradient's rows back. */
        void scatter_add_rows_backward(const GradientTape& tape);

        /** @brief Differentiates the causal mask by zeroing the gradient above the diagonal. */
        void causal_mask_backward(const GradientTape& tape);

        /** @brief Holds backward rule entries for linear algebra operations. */
        inline constexpr std::array<RuleEntry, 6> K_LINALG_RULES = {{
            {Ops::K_MATMUL.id,           &matmul_backward},
            {Ops::K_BIAS_ADD.id,         &bias_add_backward},
            {Ops::K_SUM_ROWS.id,         &sum_rows_backward},
            {Ops::K_GATHER_ROWS.id,      &gather_rows_backward},
            {Ops::K_SCATTER_ADD_ROWS.id, &scatter_add_rows_backward},
            {Ops::K_CAUSAL_MASK.id,      &causal_mask_backward},
        }};
    }
}
