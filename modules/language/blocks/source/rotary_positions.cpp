/**************************************************************************/
/* rotary_positions.cpp                                                   */
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
 * @file rotary_positions.cpp
 * @brief Implements the angles of rotary positions.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/rotary_positions.hpp>

#include <cmath>
#include <cstddef>
#include <SushiAI/core/check.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        double rotary_angle(std::int64_t position, std::int64_t pair,
                            std::int64_t pairs) noexcept
        {
            const double exponent = -static_cast<double>(pair) / static_cast<double>(pairs);
            return static_cast<double>(position) * std::pow(K_ROTARY_BASE, exponent);
        }

        void write_rotary_row(std::int64_t position, SushiRuntime::span<float> cosines,
                              SushiRuntime::span<float> sines) noexcept
        {
            SA_FATAL_IF(cosines.empty() || cosines.size() != sines.size(),
                        "a rotary row was given {} cosines and {} sines; it writes one of each "
                        "per pair",
                        cosines.size(), sines.size());

            const std::int64_t pairs = static_cast<std::int64_t>(cosines.size());
            for (std::int64_t pair = 0; pair < pairs; ++pair)
            {
                const double angle = rotary_angle(position, pair, pairs);
                cosines[static_cast<std::size_t>(pair)] = static_cast<float>(std::cos(angle));
                sines[static_cast<std::size_t>(pair)] = static_cast<float>(std::sin(angle));
            }
        }

    } // namespace Language::Blocks
} // namespace SushiAI
