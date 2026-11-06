/**************************************************************************/
/* seeded_sequence.hpp                                                    */
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
 * @file seeded_sequence.hpp
 * @brief Declares the seeded sequence of numbers the anchor clustering draws from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

/** @brief Holds the units of the anchor part that no caller names. */
namespace SushiAI::Vision::Detect::Anchor::Detail
{
    /** @brief Yields the SplitMix64 sequence of a seed, the same on every library. */
    class SeededSequence final
    {
        public:
            /** @brief Starts the sequence of @p seed. */
            explicit SeededSequence(std::uint64_t seed) noexcept;

            /** @brief Returns the next 64-bit number of the sequence. */
            [[nodiscard]] std::uint64_t next() noexcept;

            /**
             * @brief Returns the next number of the sequence as a double in [0, 1).
             * @return The upper 53 bits of next() over 2^53.
             */
            [[nodiscard]] double next_unit() noexcept;

        private:
            /// @brief Holds the counter the next number is mixed from.
            std::uint64_t state_;
    };
} // namespace SushiAI::Vision::Detect::Anchor::Detail
