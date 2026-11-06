/**************************************************************************/
/* splitmix64.hpp                                                         */
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
 * @file splitmix64.hpp
 * @brief Computes the published SplitMix64 sequence, the oracle of the seeded-draw tests.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace TestSupport
{
/**
 * @brief Returns output @p index of the published SplitMix64 sequence seeded with @p state.
 * @param state The seed of the sequence.
 * @param index The zero-based position of the output in the sequence.
 * @return The output at @p index.
 */
[[nodiscard]] constexpr std::uint64_t reference_splitmix64(std::uint64_t state,
                                                           std::uint64_t index) noexcept
{
    std::uint64_t value = state + (index + 1u) * 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}
} // namespace TestSupport
