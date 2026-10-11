/**************************************************************************/
/* rotary_positions.hpp                                                   */
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
 * @file rotary_positions.hpp
 * @brief Declares the angles of rotary positions, computed on the host.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Holds the base of the rotary frequencies: pair i turns base^(-i / pairs). */
        inline constexpr double K_ROTARY_BASE = 10000.0;

        /**
         * @brief Returns position * K_ROTARY_BASE^(-pair / pairs), the angle in radians that
         *        pair @p pair of a group of @p pairs pairs turns by at @p position.
         * @param pair In [0, pairs).
         * @param pairs Positive.
         */
        [[nodiscard]] double rotary_angle(std::int64_t position, std::int64_t pair,
                                          std::int64_t pairs) noexcept;

        /**
         * @brief Writes the cosine and the sine of rotary_angle() of every pair at @p position.
         * @param cosines One element per pair; an empty span is fatal.
         * @param sines As many elements as @p cosines; another count is fatal.
         */
        void write_rotary_row(std::int64_t position, SushiRuntime::span<float> cosines,
                              SushiRuntime::span<float> sines) noexcept;

    } // namespace Language::Blocks
} // namespace SushiAI
