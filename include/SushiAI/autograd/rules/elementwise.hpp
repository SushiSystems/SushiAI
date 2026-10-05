/****************************************************************/
/* elementwise.hpp                                              */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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

        /** @brief Holds backward rule entries for elementwise operations. */
        inline constexpr std::array<RuleEntry, 6> ELEMENTWISE_RULES = {{
            {Ops::ADD.id,        &add_backward},
            {Ops::SUB.id,        &sub_backward},
            {Ops::MUL.id,        &mul_backward},
            {Ops::SCALE.id,      &scale_backward},
            {Ops::EXP.id,        &exp_backward},
            {Ops::ACCUMULATE.id, &accumulate_backward},
        }};
    }
}
