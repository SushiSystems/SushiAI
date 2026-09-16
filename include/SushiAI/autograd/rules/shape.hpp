/**************************************************************************/
/* shape.hpp                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
