/**************************************************************************/
/* random.cpp                                                             */
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
 * @file random.cpp
 * @brief Implements the SplitMix64 generator and its mappings to floats and indices.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/random.hpp>

#include <cmath>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the number of low bits uniform() drops from a 64-bit draw. */
        constexpr std::int32_t K_UNIFORM_SHIFT = 40;

        /** @brief Holds 2^-24, the spacing of the values uniform() returns. */
        constexpr float K_UNIFORM_STEP = 0x1.0p-24f;
    } // namespace

    std::uint64_t Random::mix(std::uint64_t value) noexcept
    {
        std::uint64_t z = value;
        z = (z ^ (z >> 30)) * K_FIRST_MULTIPLIER;
        z = (z ^ (z >> 27)) * K_SECOND_MULTIPLIER;
        return z ^ (z >> 31);
    }

    Random Random::derive(std::uint64_t key) const noexcept
    {
        return Random(mix(state_ ^ mix(key + K_INCREMENT)));
    }

    std::uint64_t Random::next_u64() noexcept
    {
        state_ += K_INCREMENT;
        return mix(state_);
    }

    std::uint32_t Random::next_u32() noexcept
    {
        return static_cast<std::uint32_t>(next_u64() >> 32);
    }

    float Random::uniform() noexcept
    {
        // A 24-bit integer and its product with 2^-24 are both exact in a float.
        return static_cast<float>(next_u64() >> K_UNIFORM_SHIFT) * K_UNIFORM_STEP;
    }

    float Random::uniform(float low, float high) noexcept
    {
        const float unit = uniform();
        const float span = high - low;
        const float scaled = span * unit;
        const float value = low + scaled;

        if (low < high && value >= high)
            return std::nextafter(high, low);

        return value;
    }

    std::uint32_t Random::index(std::uint32_t count) noexcept
    {
        const std::uint64_t product = static_cast<std::uint64_t>(next_u32()) * count;
        return static_cast<std::uint32_t>(product >> 32);
    }

    bool Random::bernoulli(float probability) noexcept
    {
        return uniform() < probability;
    }
} // namespace SushiAI::Vision::Imaging
