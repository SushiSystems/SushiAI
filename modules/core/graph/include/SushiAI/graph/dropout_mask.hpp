/**************************************************************************/
/* dropout_mask.hpp                                                       */
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
 * @file dropout_mask.hpp
 * @brief Declares the counter-based generator that draws the keep mask of a dropout site.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Returns output @p index of the SplitMix64 sequence seeded with @p state.
         * @return finalise(state + (index + 1) * 0x9E3779B97F4A7C15), which for one state maps
         *         distinct indices to distinct values.
         */
        [[nodiscard]] std::uint64_t splitmix64_at(std::uint64_t state,
                                                  std::uint64_t index) noexcept;

        /**
         * @brief Returns the seed of dropout site @p site of a trace seeded with @p seed.
         * @param site The order of the site among the random sites of its trace, from 0.
         */
        [[nodiscard]] std::uint64_t dropout_site_seed(std::uint64_t seed,
                                                      std::uint64_t site) noexcept;

        /**
         * @brief Returns the 64 bits a site draws for one element in one run.
         * @param seed The site seed; @p run The replays finished before this one.
         */
        [[nodiscard]] std::uint64_t dropout_bits(std::uint64_t seed, std::uint64_t run,
                                                 std::uint64_t element) noexcept;

        /**
         * @brief Returns whether a site keeps one element in one run.
         * @param keep The probability of keeping; the top 53 bits of the draw, as a number
         *        in [0, 1), are compared against it.
         */
        [[nodiscard]] bool dropout_keeps(std::uint64_t seed, std::uint64_t run,
                                         std::uint64_t element, double keep) noexcept;

        /**
         * @brief Writes 1 for each kept element of one run of a site and 0 for the others.
         * @param mask Holds @p count elements; element i is 1 when dropout_keeps() holds for i.
         */
        void fill_keep_mask(float* mask, std::size_t count, std::uint64_t seed,
                            std::uint64_t run, double keep) noexcept;

    } // namespace Graph
} // namespace SushiAI
