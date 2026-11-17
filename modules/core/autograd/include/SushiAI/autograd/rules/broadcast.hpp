/**************************************************************************/
/* broadcast.hpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file broadcast.hpp
 * @brief Declares backward differentiation rules for the broadcast arithmetic and the sum along.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief BROADCAST_ADD's gradient: dy to the tensor, the sum of dy down to the vector.
         * @param tape The differentiation state for this node.
         */
        void broadcast_add_backward(const GradientTape& tape) noexcept;

        /**
         * @brief BROADCAST_SUB's gradient: dy to the tensor, minus the sum of dy to the vector.
         * @param tape The differentiation state for this node.
         */
        void broadcast_sub_backward(const GradientTape& tape) noexcept;

        /**
         * @brief BROADCAST_MUL's gradient: dy times the vector to the tensor, the sum of dy
         *        times the tensor down to the vector.
         * @param tape The differentiation state for this node.
         */
        void broadcast_mul_backward(const GradientTape& tape) noexcept;

        /**
         * @brief BROADCAST_DIV's gradient: dy over the vector to the tensor, minus the sum of dy
         *        times the result over the vector down to the vector.
         * @param tape The differentiation state for this node.
         */
        void broadcast_div_backward(const GradientTape& tape) noexcept;

        /**
         * @brief SUM_ALONG's gradient: dy added to zeros of the operand's shape along its side.
         * @param tape The differentiation state for this node.
         */
        void sum_along_backward(const GradientTape& tape) noexcept;

        /** @brief Holds backward rule entries for the broadcast operations. */
        inline constexpr std::array<RuleEntry, 5> K_BROADCAST_RULES = {{
            {Ops::K_BROADCAST_ADD.id, &broadcast_add_backward},
            {Ops::K_BROADCAST_SUB.id, &broadcast_sub_backward},
            {Ops::K_BROADCAST_MUL.id, &broadcast_mul_backward},
            {Ops::K_BROADCAST_DIV.id, &broadcast_div_backward},
            {Ops::K_SUM_ALONG.id,     &sum_along_backward},
        }};
    } // namespace Autograd
} // namespace SushiAI
