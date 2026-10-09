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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
