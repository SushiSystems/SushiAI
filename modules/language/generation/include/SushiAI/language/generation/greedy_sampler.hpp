/**************************************************************************/
/* greedy_sampler.hpp                                                     */
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
 * @file greedy_sampler.hpp
 * @brief Declares GreedySampler, which picks the token of the largest logit.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Names the greedy sampler in a task section. */
    inline constexpr std::string_view K_GREEDY_SAMPLER = "greedy";

    /** @brief Picks the token of the largest logit, the lowest index among equals. */
    class GreedySampler final : public Sampler
    {
        public:
            /** @brief Returns K_GREEDY_SAMPLER. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /** @brief Returns the index of the largest logit and draws nothing. */
            [[nodiscard]] Text::TokenID sample(LogitSpan logits,
                                               RandomGenerator& random) const override;
    };

} // namespace SushiAI::Language::Generation
