/**************************************************************************/
/* seeded_sequence.cpp                                                    */
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
 * @file seeded_sequence.cpp
 * @brief Implements the seeded sequence of numbers the anchor clustering draws from.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/detect/anchor/seeded_sequence.hpp>

namespace SushiAI::Vision::Detect::Anchor::Detail
{
    namespace
    {
        /** @brief Holds the step the counter advances by between two numbers. */
        constexpr std::uint64_t K_COUNTER_STEP = 0x9E3779B97F4A7C15ull;

        /** @brief Holds the multiplier of the first mixing round. */
        constexpr std::uint64_t K_FIRST_MULTIPLIER = 0xBF58476D1CE4E5B9ull;

        /** @brief Holds the multiplier of the second mixing round. */
        constexpr std::uint64_t K_SECOND_MULTIPLIER = 0x94D049BB133111EBull;

        /** @brief Holds 2^-53, the spacing of the doubles next_unit returns. */
        constexpr double K_UNIT_SPACING = 1.0 / 9007199254740992.0;
    } // namespace

    SeededSequence::SeededSequence(std::uint64_t seed) noexcept : state_(seed)
    {

    }

    std::uint64_t SeededSequence::next() noexcept
    {
        state_ += K_COUNTER_STEP;

        std::uint64_t mixed = state_;
        mixed = (mixed ^ (mixed >> 30)) * K_FIRST_MULTIPLIER;
        mixed = (mixed ^ (mixed >> 27)) * K_SECOND_MULTIPLIER;
        return mixed ^ (mixed >> 31);
    }

    double SeededSequence::next_unit() noexcept
    {
        return static_cast<double>(next() >> 11) * K_UNIT_SPACING;
    }
} // namespace SushiAI::Vision::Detect::Anchor::Detail
