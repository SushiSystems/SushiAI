/**************************************************************************/
/* dropout_mask.cpp                                                       */
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
 * @file dropout_mask.cpp
 * @brief Implements the SplitMix64 sequence that seeds the draws of a dropout site.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/dropout_mask.hpp>

#include <cstdint>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Holds the SplitMix64 increment, the golden ratio in 64 bits. */
            constexpr std::uint64_t K_INCREMENT = 0x9E3779B97F4A7C15ull;

            /** @brief Holds the first multiplier of the SplitMix64 finaliser. */
            constexpr std::uint64_t K_FIRST_MULTIPLIER = 0xBF58476D1CE4E5B9ull;

            /** @brief Holds the second multiplier of the SplitMix64 finaliser. */
            constexpr std::uint64_t K_SECOND_MULTIPLIER = 0x94D049BB133111EBull;

            /** @brief Scrambles one 64-bit value with the SplitMix64 finaliser. */
            [[nodiscard]] constexpr std::uint64_t finalise(std::uint64_t value) noexcept
            {
                value = (value ^ (value >> 30)) * K_FIRST_MULTIPLIER;
                value = (value ^ (value >> 27)) * K_SECOND_MULTIPLIER;
                return value ^ (value >> 31);
            }
        } // namespace

        std::uint64_t splitmix64_at(std::uint64_t state, std::uint64_t index) noexcept
        {
            return finalise(state + (index + 1u) * K_INCREMENT);
        }

        std::uint64_t dropout_site_seed(std::uint64_t seed, std::uint64_t site) noexcept
        {
            return splitmix64_at(seed, site);
        }

    } // namespace Graph
} // namespace SushiAI
