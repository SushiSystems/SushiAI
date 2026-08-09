/**************************************************************************/
/* linalg.hpp                                                             */
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
        /** @brief The four-transposition-case gradient of a 2-D matrix product. */
        void matmul_backward(const GradientTape& tape);
        /** @brief Y = X + 1 b^T: identity in X, row-summed in b. */
        void bias_add_backward(const GradientTape& tape);
        /** @brief The adjoint of summing a matrix's rows: broadcast back over rows. */
        void sum_rows_backward(const GradientTape& tape);

        /** @brief Linear algebra operations' backward rules. */
        inline constexpr std::array<RuleEntry, 3> LINALG_RULES = {{
            {Ops::MATMUL.id,   &matmul_backward},
            {Ops::BIAS_ADD.id, &bias_add_backward},
            {Ops::SUM_ROWS.id, &sum_rows_backward},
        }};
    }
}
