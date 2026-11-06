/**************************************************************************/
/* sampler.hpp                                                            */
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
 * @file sampler.hpp
 * @brief Declares Sampler, the seam that picks one token from a row of logits.
 * @author Mustafa Garip
 */

#pragma once

#include <random>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Generation
{
    /**
     * @brief Names the generator a sampler draws from, which its caller owns and seeds.
     *
     * The C++ standard fixes its output for a seed, so a run repeats on every platform.
     */
    using RandomGenerator = std::mt19937_64;

    /** @brief Views the scores of one position, one per token of the vocabulary. */
    using LogitSpan = SushiRuntime::span<const float>;

    /** @brief Picks one token from the logits of one position. */
    class Sampler
    {
        public:
            /** @brief Releases the sampler through its base. */
            virtual ~Sampler() = default;

            /** @brief Returns the name a task section spells this sampler by. */
            [[nodiscard]] virtual std::string_view name() const noexcept = 0;

            /**
             * @brief Returns the index of the token picked from @p logits.
             * @param logits One finite score per token; empty is fatal.
             * @param random Advanced by at most one draw; a sampler that draws nothing
             *        leaves it as it was.
             */
            [[nodiscard]] virtual Text::TokenID sample(LogitSpan logits,
                                                       RandomGenerator& random) const = 0;
    };

} // namespace SushiAI::Language::Generation
