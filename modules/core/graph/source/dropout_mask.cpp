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
 * @file dropout_mask.cpp
 * @brief Implements the SplitMix64 draws behind the keep mask of a dropout site.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/dropout_mask.hpp>

#include <cstddef>
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

            /** @brief Holds 2^-53, which scales the top 53 bits of a draw into [0, 1). */
            constexpr double K_UNIT_STEP = 1.0 / 9007199254740992.0;

            /** @brief Holds how many low bits of a draw the comparison leaves out. */
            constexpr int K_DISCARDED_BITS = 11;

            /** @brief Scrambles one 64-bit value with the SplitMix64 finaliser. */
            [[nodiscard]] constexpr std::uint64_t finalise(std::uint64_t value) noexcept
            {
                value = (value ^ (value >> 30)) * K_FIRST_MULTIPLIER;
                value = (value ^ (value >> 27)) * K_SECOND_MULTIPLIER;
                return value ^ (value >> 31);
            }

            /** @brief Returns whether the draw @p bits, read as a number in [0, 1), is kept. */
            [[nodiscard]] constexpr bool is_kept(std::uint64_t bits, double keep) noexcept
            {
                return static_cast<double>(bits >> K_DISCARDED_BITS) * K_UNIT_STEP < keep;
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

        std::uint64_t dropout_bits(std::uint64_t seed, std::uint64_t run,
                                   std::uint64_t element) noexcept
        {
            return splitmix64_at(splitmix64_at(seed, run), element);
        }

        bool dropout_keeps(std::uint64_t seed, std::uint64_t run, std::uint64_t element,
                           double keep) noexcept
        {
            return is_kept(dropout_bits(seed, run, element), keep);
        }

        void fill_keep_mask(float* mask, std::size_t count, std::uint64_t seed,
                            std::uint64_t run, double keep) noexcept
        {
            // The state after i + 1 increments is the argument splitmix64_at finalises for i.
            std::uint64_t state = splitmix64_at(seed, run);
            for (std::size_t element = 0; element < count; ++element)
            {
                state += K_INCREMENT;
                mask[element] = is_kept(finalise(state), keep) ? 1.0f : 0.0f;
            }
        }

    } // namespace Graph
} // namespace SushiAI
