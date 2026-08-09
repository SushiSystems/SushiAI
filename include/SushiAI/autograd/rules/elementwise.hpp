/**************************************************************************/
/* elementwise.hpp                                                        */
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

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief d(a+b) = [dy, dy]. */
        void add_backward(const GradientTape& tape);
        /** @brief d(a-b) = [dy, -dy]. */
        void sub_backward(const GradientTape& tape);
        /** @brief d(a*b) = [dy*b, dy*a]. */
        void mul_backward(const GradientTape& tape);
        /** @brief d(alpha*x) = alpha*dy. */
        void scale_backward(const GradientTape& tape);
        /** @brief d(exp x) = dy * exp(x), reading the forward output. */
        void exp_backward(const GradientTape& tape);
        /** @brief An in-graph accumulation passes its gradient to both operands. */
        void accumulate_backward(const GradientTape& tape);

        /** @brief Elementwise operations' backward rules. */
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
