/****************************************************************/
/* shape.hpp                                                    */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file shape.hpp
 * @brief Declares backward differentiation rules for tensor shape and layout operations.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief Differentiates reshape operations by restoring operand shape on incoming gradients. */
        void reshape_backward(const GradientTape& tape);
        /** @brief Differentiates layout conversions by restoring operand layout on incoming gradients. */
        void layout_convert_backward(const GradientTape& tape);

        /** @brief Holds backward rule entries for structural shape and layout operations. */
        inline constexpr std::array<RuleEntry, 2> SHAPE_RULES = {{
            {Ops::RESHAPE.id,        &reshape_backward},
            {Ops::LAYOUT_CONVERT.id, &layout_convert_backward},
        }};
    }
}
