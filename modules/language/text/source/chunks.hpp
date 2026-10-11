/**************************************************************************/
/* chunks.hpp                                                             */
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
 * @file chunks.hpp
 * @brief Defines how a text is cut into the chunks no merge crosses.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <string_view>

#include <SushiAI/language/text/token.hpp>

/** @brief Holds the units of the text module that no caller names. */
namespace SushiAI::Language::Text::Detail
{
    /** @brief Holds the most bytes of one chunk, so that no token outgrows its limit. */
    inline constexpr std::size_t K_MAX_CHUNK_BYTES = K_MAX_TOKEN_BYTES;

    /** @brief Reports whether a byte is a space, a tab, a line feed or a carriage return. */
    [[nodiscard]] constexpr bool is_blank(char byte) noexcept
    {
        return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
    }

    /**
     * @brief Returns the end of the chunk that begins at a position of a text.
     *
     * A chunk is a run of blank bytes and the run of other bytes after it, cut at
     * K_MAX_CHUNK_BYTES. The chunks of a text follow one another and cover it.
     * @pre @p begin is below text.size().
     * @return A position above @p begin and at most text.size().
     */
    [[nodiscard]] constexpr std::size_t chunk_end(std::string_view text,
                                                  std::size_t begin) noexcept
    {
        const std::size_t limit = std::min(text.size(), begin + K_MAX_CHUNK_BYTES);
        std::size_t end = begin;

        while (end < limit && is_blank(text[end]))
            ++end;
        while (end < limit && !is_blank(text[end]))
            ++end;

        return end;
    }
} // namespace SushiAI::Language::Text::Detail
