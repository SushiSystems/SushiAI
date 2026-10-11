/**************************************************************************/
/* stream_encoder.hpp                                                     */
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
 * @file stream_encoder.hpp
 * @brief Declares the encoder that reads a text from a stream in pieces of a bounded size.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <istream>
#include <vector>
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

namespace SushiAI::Language::Text
{
    /** @brief Holds the bytes a caller of encode_stream() reads at a time by default. */
    inline constexpr std::size_t K_STREAM_PIECE_BYTES = std::size_t{1} << 20;

    /**
     * @brief Appends the tokens encode_text() gives for every byte left in @p in.
     * @param piece_bytes The bytes read at a time, at least one. The bytes held grow past
     *        it only while no blank byte follows another byte in what was read.
     * @param ids Appended to; unspecified after a failure.
     * @return Success, InvalidFile when the stream fails, or the failure of encode_text().
     */
    [[nodiscard]] Status encode_stream(const Vocabulary& vocabulary, std::istream& in,
                                       std::size_t piece_bytes, std::vector<TokenID>& ids);

} // namespace SushiAI::Language::Text
