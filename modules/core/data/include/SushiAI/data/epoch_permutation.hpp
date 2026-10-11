/**************************************************************************/
/* epoch_permutation.hpp                                                  */
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
 * @file epoch_permutation.hpp
 * @brief Declares the order of one epoch as a function of a seed and the epoch.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @class EpochPermutation
         * @brief Maps each position of an epoch to one sample, every sample exactly once.
         *
         * The map depends on the seed, the epoch and the count alone and is stored nowhere:
         * at() computes it, so threads may call it at once.
         * @see modules/core/data/README.md, section "The order of an epoch".
         */
        class EpochPermutation final
        {
            public:
                /** @brief Counts the rounds of the keyed network at() runs. */
                static constexpr std::size_t K_ROUNDS = 4;

                /**
                 * @brief Derives the keys of one epoch's order over [0, count).
                 * @param epoch Any value; two epochs of one seed give two orders.
                 * @pre @p count is at least 1; zero stops the process.
                 */
                EpochPermutation(std::uint64_t seed, std::int64_t epoch,
                                 std::uint64_t count) noexcept;

                /**
                 * @brief Returns the sample read at @p position of the epoch.
                 * @pre @p position is below count(); one that is not stops the process.
                 */
                [[nodiscard]] std::uint64_t at(std::uint64_t position) const noexcept;

                /** @brief Returns how many positions the order has. */
                [[nodiscard]] std::uint64_t count() const noexcept { return count_; }

            private:
                /** @brief Holds one key per round, derived from the seed and the epoch. */
                std::array<std::uint64_t, K_ROUNDS> keys_{};

                /** @brief Holds the number of positions and of samples. */
                std::uint64_t count_ = 0;

                /** @brief Counts the bits of one half of a value the network permutes. */
                std::uint32_t half_bits_ = 1;

                /** @brief Holds the mask that keeps half_bits_ bits. */
                std::uint64_t half_mask_ = 1;
        };

    } // namespace Data
} // namespace SushiAI
