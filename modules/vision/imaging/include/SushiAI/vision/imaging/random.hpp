/**************************************************************************/
/* random.hpp                                                             */
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
 * @file random.hpp
 * @brief Declares the generator every random draw of the imaging module comes from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Generates the SplitMix64 sequence from a 64-bit counter.
     *
     * The state is one counter. Each draw adds K_INCREMENT to it modulo 2^64 and returns
     * mix() of the sum, so the n-th output of a seed is mix(seed + n * K_INCREMENT). Every
     * mapping below is integer arithmetic or one exact float operation, so a seed gives the
     * same draws on every platform. A copy replays the sequence from the point of the copy.
     */
    class Random final
    {
        public:
            /** @brief Holds the amount the counter advances by on each draw. */
            static constexpr std::uint64_t K_INCREMENT = 0x9E3779B97F4A7C15ull;

            /** @brief Holds the first multiplier of mix(). */
            static constexpr std::uint64_t K_FIRST_MULTIPLIER = 0xBF58476D1CE4E5B9ull;

            /** @brief Holds the second multiplier of mix(). */
            static constexpr std::uint64_t K_SECOND_MULTIPLIER = 0x94D049BB133111EBull;

            /**
             * @brief Constructs a generator whose counter starts at the seed.
             * @param seed Any value, zero included.
             */
            explicit constexpr Random(std::uint64_t seed) noexcept : state_(seed)
            {

            }

            /**
             * @brief Scrambles one 64-bit value with the SplitMix64 finaliser.
             *
             * With z = value: z = (z ^ (z >> 30)) * K_FIRST_MULTIPLIER, then
             * z = (z ^ (z >> 27)) * K_SECOND_MULTIPLIER, both modulo 2^64, and the result is
             * z ^ (z >> 31).
             */
            [[nodiscard]] static std::uint64_t mix(std::uint64_t value) noexcept;

            /**
             * @brief Returns a generator for one keyed stream without advancing this one.
             *
             * The new counter starts at mix(state() ^ mix(key + K_INCREMENT)). A pipeline
             * derives one generator per epoch and per sample index, so a sample's draws do not
             * depend on the order in which threads produce samples.
             */
            [[nodiscard]] Random derive(std::uint64_t key) const noexcept;

            /** @brief Advances the counter and returns the next 64 bits. */
            [[nodiscard]] std::uint64_t next_u64() noexcept;

            /** @brief Returns the high 32 bits of one next_u64() draw. */
            [[nodiscard]] std::uint32_t next_u32() noexcept;

            /**
             * @brief Returns a float in [0, 1) from one draw.
             *
             * The value is (next_u64() >> 40) * 2^-24: 24 bits, each value exact in a float,
             * the largest being 1 - 2^-24.
             */
            [[nodiscard]] float uniform() noexcept;

            /**
             * @brief Returns a float in [low, high) from one draw.
             *
             * The value is low + (high - low) * uniform(), evaluated in float in that order.
             * A result that rounds up to high is replaced by the largest float below high.
             * With low == high the draw is still made and the result is low.
             * @pre low <= high, both finite.
             */
            [[nodiscard]] float uniform(float low, float high) noexcept;

            /**
             * @brief Returns an integer in [0, count) from one draw.
             *
             * The value is (next_u32() * count) >> 32 in 64-bit arithmetic. No draw is
             * rejected, so a value's probability differs from 1 / count by at most 2^-32.
             * @pre count > 0.
             */
            [[nodiscard]] std::uint32_t index(std::uint32_t count) noexcept;

            /**
             * @brief Reports whether one draw fell below a probability.
             *
             * The result is uniform() < probability, so 0 never succeeds and 1 always does.
             * The draw is made in both of those cases.
             */
            [[nodiscard]] bool bernoulli(float probability) noexcept;

            /** @brief Returns the counter, which is the whole state. */
            [[nodiscard]] constexpr std::uint64_t state() const noexcept { return state_; }

            /** @brief Reports whether two generators will produce the same sequence. */
            [[nodiscard]] friend constexpr bool operator==(const Random& a,
                                                           const Random& b) noexcept
            {
                return a.state_ == b.state_;
            }

            /** @brief Reports whether two generators will produce different sequences. */
            [[nodiscard]] friend constexpr bool operator!=(const Random& a,
                                                           const Random& b) noexcept
            {
                return !(a == b);
            }

        private:
            /// @brief Holds the counter the next draw advances.
            std::uint64_t state_ = 0;
    };
} // namespace SushiAI::Vision::Imaging
