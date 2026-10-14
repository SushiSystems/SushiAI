/**************************************************************************/
/* host_noise.hpp                                                         */
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
 * @file host_noise.hpp
 * @brief Declares HostNoise, a seeded stream of values for filling test tensors.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Generates a reproducible value stream for filling test tensors.
         */
        class HostNoise
        {
            public:
                /** @brief Seeds the stream. */
                explicit HostNoise(std::uint64_t seed) noexcept : state_(seed) {}

                /** @brief Returns the next value, roughly uniform in (-scale, scale). */
                double next(double scale = 1.0) noexcept
                {
                    state_ += 0x9E3779B97F4A7C15ull;
                    std::uint64_t z = state_;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                    z ^= (z >> 31);

                    const double unit = static_cast<double>(z >> 11)
                                      / static_cast<double>(1ull << 53);
                    return (unit * 2.0 - 1.0) * scale;
                }

                /** @brief Fills @p values with next(@p scale). */
                void fill(SushiRuntime::span<double> values, double scale = 1.0) noexcept
                {
                    for (double& v : values) v = next(scale);
                }

            private:
                /** @brief Holds the splitmix state. */
                std::uint64_t state_;
        };

    } // namespace Test
} // namespace SushiAI
