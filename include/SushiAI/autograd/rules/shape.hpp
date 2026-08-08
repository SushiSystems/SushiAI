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

#pragma once

// Shape and layout operations' backward rules: RESHAPE, LAYOUT_CONVERT and
// friends.

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief Reshapes the incoming gradient back to the operand's shape. */
        void reshape_backward(const GradientTape& tape);
        /** @brief Converts the incoming gradient back to the operand's layout. */
        void layout_convert_backward(const GradientTape& tape);

        /** @brief Structural operations' backward rules. */
        inline constexpr std::array<RuleEntry, 2> SHAPE_RULES = {{
            {Ops::RESHAPE.id,        &reshape_backward},
            {Ops::LAYOUT_CONVERT.id, &layout_convert_backward},
        }};
    }
}
