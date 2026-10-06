/**************************************************************************/
/* shape.hpp                                                              */
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
