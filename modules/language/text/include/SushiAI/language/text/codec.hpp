/**************************************************************************/
/* codec.hpp                                                              */
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
 * @file codec.hpp
 * @brief Declares the encoder from text bytes to token ids and the decoder back.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

/** @brief Holds the tokens of a text, the vocabulary and the byte-pair tokenizer. */
namespace SushiAI::Language::Text
{
    /**
     * @brief Appends the tokens of a text, applying the merges in the order they were learned.
     *
     * Any byte string encodes, valid UTF-8 or not, and no special token is ever written.
     * @param ids Appended to; unspecified after a failure.
     * @return Success, or OutOfMemory when @p ids cannot grow.
     * @see modules/language/text/README.md, section "Encoding".
     */
    [[nodiscard]] Status encode_text(const Vocabulary& vocabulary, std::string_view text,
                                     std::vector<TokenID>& ids) noexcept;

    /**
     * @brief Appends the bytes of each token; a special token appends none.
     *
     * decode_tokens over what encode_text wrote gives the text back byte for byte.
     * @param text Appended to; unspecified after a failure.
     * @return Success; InvalidDescription, naming the index, for an id the vocabulary does
     *         not hold; OutOfMemory when @p text cannot grow.
     */
    [[nodiscard]] Status decode_tokens(const Vocabulary& vocabulary,
                                       SushiRuntime::span<const TokenID> ids, std::string& text) noexcept;
} // namespace SushiAI::Language::Text
