/**************************************************************************/
/* ranking.cpp                                                            */
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
 * @file ranking.cpp
 * @brief Implements the descending order of a row of logits.
 * @author Mustafa Garip
 */

#include "ranking.hpp"

#include <algorithm>
#include <numeric>

namespace SushiAI::Language::Generation::Detail
{
    std::vector<std::size_t> rank_descending(LogitSpan logits, std::size_t leading)
    {
        std::vector<std::size_t> order(logits.size());
        std::iota(order.begin(), order.end(), std::size_t{0});

        const auto before = [&logits](std::size_t left, std::size_t right)
        {
            if (logits[left] != logits[right])
                return logits[left] > logits[right];

            return left < right;
        };

        const std::size_t sorted = std::min(leading, order.size());
        std::partial_sort(order.begin(),
                          order.begin() + static_cast<std::ptrdiff_t>(sorted), order.end(),
                          before);
        return order;
    }

} // namespace SushiAI::Language::Generation::Detail
