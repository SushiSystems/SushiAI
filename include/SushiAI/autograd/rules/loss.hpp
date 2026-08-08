/**************************************************************************/
/* loss.hpp                                                               */
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
        /**
         * @brief LOG_SOFTMAX's standalone gradient: not yet implemented.
         *
         * Reachable only when a LOG_SOFTMAX result is consumed by something
         * other than NLL_LOSS. Declared here and wired into @c find_rule as
         * the extra entry that only exists to produce a good error - see
         * @c differentiator.cpp - rather than placed in @ref LOSS_RULES,
         * because it does not differentiate anything and has no business
         * being counted as though it did.
         */
        void log_softmax_backward(const GradientTape& tape);

        /**
         * @brief NLL_LOSS's gradient, fused with LOG_SOFTMAX's when adjacent.
         */
        void nll_loss_backward(const GradientTape& tape);

        /** @brief Loss operations' backward rules. */
        inline constexpr std::array<RuleEntry, 1> LOSS_RULES = {{
            {Ops::NLL_LOSS.id, &nll_loss_backward},
        }};
    }
}
