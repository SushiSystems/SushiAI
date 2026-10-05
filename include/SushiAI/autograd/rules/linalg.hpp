/****************************************************************/
/* linalg.hpp                                                   */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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

        /** @brief Holds backward rule entries for linear algebra operations. */
        inline constexpr std::array<RuleEntry, 3> LINALG_RULES = {{
            {Ops::MATMUL.id,   &matmul_backward},
            {Ops::BIAS_ADD.id, &bias_add_backward},
            {Ops::SUM_ROWS.id, &sum_rows_backward},
        }};
    }
}
