/**************************************************************************/
/* epoch_permutation.cpp                                                  */
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
 * @file epoch_permutation.cpp
 * @brief Implements the order of one epoch as a keyed permutation with no stored table.
 * @author Mustafa Garip
 */

#include <SushiAI/data/epoch_permutation.hpp>

#include <SushiAI/core/check.hpp>

namespace SushiAI
{
    namespace Data
    {
        namespace
        {
            /** @brief Holds the odd constant that separates two keys of one seed. */
            constexpr std::uint64_t K_INCREMENT = 0x9E3779B97F4A7C15ull;

            /** @brief Holds the first multiplier of mix(). */
            constexpr std::uint64_t K_FIRST_MULTIPLIER = 0xBF58476D1CE4E5B9ull;

            /** @brief Holds the second multiplier of mix(). */
            constexpr std::uint64_t K_SECOND_MULTIPLIER = 0x94D049BB133111EBull;

            /** @brief Counts the bits of the widest value the network permutes. */
            constexpr std::uint32_t K_VALUE_BITS = 64;

            /** @brief Scrambles one 64-bit value with the SplitMix64 finaliser. */
            [[nodiscard]] constexpr std::uint64_t mix(std::uint64_t value) noexcept
            {
                value = (value ^ (value >> 30)) * K_FIRST_MULTIPLIER;
                value = (value ^ (value >> 27)) * K_SECOND_MULTIPLIER;
                return value ^ (value >> 31);
            }
        } // namespace

        EpochPermutation::EpochPermutation(std::uint64_t seed, std::int64_t epoch,
                                           std::uint64_t count) noexcept
            : count_(count)
        {
            SA_FATAL_IF(count == 0, "EpochPermutation: an order of no position has no sample");

            while (2 * half_bits_ < K_VALUE_BITS
                   && (std::uint64_t{1} << (2 * half_bits_)) < count)
                ++half_bits_;

            half_mask_ = (std::uint64_t{1} << half_bits_) - 1;

            const std::uint64_t base =
                mix(seed ^ mix(static_cast<std::uint64_t>(epoch) + K_INCREMENT));
            for (std::size_t round = 0; round < K_ROUNDS; ++round)
                keys_[round] = mix(base + (round + 1) * K_INCREMENT);
        }

        std::uint64_t EpochPermutation::at(std::uint64_t position) const noexcept
        {
            SA_FATAL_IF(position >= count_,
                        "EpochPermutation: position {} is outside an order of {}", position,
                        count_);

            // The network permutes [0, 4^half_bits_); walking its cycle returns to [0, count_).
            std::uint64_t value = position;
            do
            {
                std::uint64_t left = value >> half_bits_;
                std::uint64_t right = value & half_mask_;

                for (std::uint64_t key : keys_)
                {
                    const std::uint64_t mixed = left ^ (mix(right + key) & half_mask_);
                    left = right;
                    right = mixed;
                }

                value = (left << half_bits_) | right;
            } while (value >= count_);

            return value;
        }

    } // namespace Data
} // namespace SushiAI
