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
 * @brief Declares backward differentiation rules for elementwise operations.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief Differentiates binary addition: d(a+b) = [dy, dy]. */
        void add_backward(const GradientTape& tape);
        /** @brief Differentiates binary subtraction: d(a-b) = [dy, -dy]. */
        void sub_backward(const GradientTape& tape);
        /** @brief Differentiates binary multiplication: d(a*b) = [dy*b, dy*a]. */
        void mul_backward(const GradientTape& tape);
        /** @brief Differentiates scalar scaling: d(alpha*x) = alpha*dy. */
        void scale_backward(const GradientTape& tape);
        /** @brief Differentiates elementwise exponential: d(exp x) = dy * exp(x). */
        void exp_backward(const GradientTape& tape);
        /** @brief Propagates accumulation gradients to both operands. */
        void accumulate_backward(const GradientTape& tape);
        /** @brief Differentiates division: d(a/b) = [dy/b, -(dy/b)*(a/b)]. */
        void div_backward(const GradientTape& tape);
        /** @brief Differentiates the minimum: dy goes to a where a <= b, to b elsewhere. */
        void minimum_backward(const GradientTape& tape);
        /** @brief Differentiates the maximum: dy goes to a where a >= b, to b elsewhere. */
        void maximum_backward(const GradientTape& tape);
        /** @brief Differentiates the absolute value: d|x| = dy * sign(x), zero at x = 0. */
        void abs_backward(const GradientTape& tape);
        /** @brief Differentiates the arctangent: d(atan x) = dy / (1 + x*x). */
        void atan_backward(const GradientTape& tape);
        /** @brief Differentiates selection: dy goes to a where the condition holds, else to b. */
        void select_backward(const GradientTape& tape);

        /** @brief Holds backward rule entries for elementwise operations. */
        inline constexpr std::array<RuleEntry, 12> ELEMENTWISE_RULES = {{
            {Ops::ADD.id,        &add_backward},
            {Ops::SUB.id,        &sub_backward},
            {Ops::MUL.id,        &mul_backward},
            {Ops::SCALE.id,      &scale_backward},
            {Ops::EXP.id,        &exp_backward},
            {Ops::ACCUMULATE.id, &accumulate_backward},
            {Ops::DIV.id,        &div_backward},
            {Ops::MINIMUM.id,    &minimum_backward},
            {Ops::MAXIMUM.id,    &maximum_backward},
            {Ops::ABS.id,        &abs_backward},
            {Ops::ATAN.id,       &atan_backward},
            {Ops::SELECT.id,     &select_backward},
        }};
    }
}
