/**************************************************************************/
/* window_layout.hpp                                                      */
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
 * @file window_layout.hpp
 * @brief Declares where the fixed-length windows of a token file begin.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/result.hpp>

/** @brief Holds the token file of a corpus and the dataset that reads it in windows. */
namespace SushiAI::Language::Datasets
{
    /** @brief Holds the most tokens one window reads. */
    inline constexpr std::int64_t K_MAX_WINDOW_TOKENS = 1 << 20;

    /**
     * @brief States the windows of a token file: window w reads @c tokens ids from w * stride.
     * @see modules/language/datasets/README.md, section "Windows".
     */
    struct WindowLayout
    {
        /** @brief Counts the ids one window reads, in [1, K_MAX_WINDOW_TOKENS]. */
        std::int64_t tokens = 0;

        /** @brief Counts the ids between the starts of two windows; at least 1. */
        std::int64_t stride = 0;
    };

    /**
     * @brief Returns how many whole windows a file of a token count holds.
     * @return The count, 0 for a file shorter than one window; InvalidDescription for a
     *         layout out of its bounds.
     */
    [[nodiscard]] Result<std::uint64_t> count_windows(std::uint64_t token_count,
                                                      const WindowLayout& layout);

    /**
     * @brief Returns the index of the first id a window reads.
     * @pre @p layout passed count_windows(), and @p window is below what it returned.
     */
    [[nodiscard]] constexpr std::uint64_t first_token_of(std::uint64_t window,
                                                         const WindowLayout& layout) noexcept
    {
        return window * static_cast<std::uint64_t>(layout.stride);
    }
} // namespace SushiAI::Language::Datasets
