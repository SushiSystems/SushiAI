/**************************************************************************/
/* window_layout.cpp                                                      */
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
 * @file window_layout.cpp
 * @brief Implements the count of the fixed-length windows of a token file.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/window_layout.hpp>

#include <string_view>


#include <SushiAI/failure/status.hpp>

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_UNIT = "count_windows";
    } // namespace

    Result<std::uint64_t> count_windows(std::uint64_t token_count, const WindowLayout& layout)
    {
        SA_FAIL_IF(layout.tokens < 1 || layout.tokens > K_MAX_WINDOW_TOKENS,
                   StatusCode::InvalidDescription, "{}: a window of {} tokens; it has 1 to {}",
                   K_UNIT, layout.tokens, K_MAX_WINDOW_TOKENS);
        SA_FAIL_IF(layout.stride < 1, StatusCode::InvalidDescription,
                   "{}: a stride of {} tokens; it must be positive", K_UNIT, layout.stride);

        const std::uint64_t tokens = static_cast<std::uint64_t>(layout.tokens);
        if (token_count < tokens)
            return std::uint64_t{0};

        return (token_count - tokens) / static_cast<std::uint64_t>(layout.stride) + 1;
    }
} // namespace SushiAI::Language::Datasets
