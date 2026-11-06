/**************************************************************************/
/* greedy_sampler.cpp                                                     */
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
 * @file greedy_sampler.cpp
 * @brief Implements GreedySampler.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/greedy_sampler.hpp>

#include <algorithm>
#include <SushiAI/failure/check.hpp>

namespace SushiAI::Language::Generation
{
    std::string_view GreedySampler::name() const noexcept
    {
        return K_GREEDY_SAMPLER;
    }

    Text::TokenID GreedySampler::sample(LogitSpan logits, RandomGenerator&) const
    {
        SA_FATAL_IF(logits.empty(), "a sampler was given no logit; a position has one per token");

        // max_element returns the first of equal elements, which is the lowest index.
        return static_cast<Text::TokenID>(std::max_element(logits.begin(), logits.end())
                                          - logits.begin());
    }

} // namespace SushiAI::Language::Generation
