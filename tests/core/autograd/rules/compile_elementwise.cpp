/**************************************************************************/
/* compile_elementwise.cpp                                                */
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
 * @file compile_elementwise.cpp
 * @brief Checks at compile time that the elementwise rule table names each operation once.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <SushiAI/autograd/rules/elementwise.hpp>

namespace
{
    /** @brief Reports whether no two entries of the elementwise rule table share an operation. */
    constexpr bool each_operation_appears_once() noexcept
    {
        const auto& table = SushiAI::Autograd::K_ELEMENTWISE_RULES;
        for (std::size_t i = 0; i < table.size(); ++i)
            for (std::size_t j = i + 1; j < table.size(); ++j)
                if (table[i].op == table[j].op)
                    return false;

        return true;
    }
} // namespace

static_assert(each_operation_appears_once(),
              "an operation in K_ELEMENTWISE_RULES has two backward rules");
