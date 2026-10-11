/**************************************************************************/
/* token.hpp                                                              */
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
 * @file token.hpp
 * @brief Declares the token id type and the limits of a vocabulary.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

/** @brief Holds the tokens of a text, the vocabulary and the byte-pair tokenizer. */
namespace SushiAI::Language::Text
{
    /** @brief Names one token; the element type of every id tensor and token file. */
    using TokenID = std::int32_t;

    /** @brief Counts the tokens that stand for one byte each; token b is the byte b. */
    inline constexpr std::int32_t K_BYTE_TOKEN_COUNT = 256;

    /** @brief Holds the id of the end-of-text token, the first special token. */
    inline constexpr TokenID K_END_OF_TEXT = K_BYTE_TOKEN_COUNT;

    /** @brief Holds the name of the end-of-text token. */
    inline constexpr std::string_view K_END_OF_TEXT_NAME = "<end_of_text>";

    /** @brief Holds the most special tokens a vocabulary has, end-of-text among them. */
    inline constexpr std::int32_t K_MAX_SPECIAL_TOKENS = 256;

    /** @brief Holds the most bytes the name of a special token has. */
    inline constexpr std::size_t K_MAX_SPECIAL_NAME_BYTES = 255;

    /** @brief Holds the most bytes one token stands for. */
    inline constexpr std::size_t K_MAX_TOKEN_BYTES = 256;

    /** @brief Holds the most tokens a vocabulary has. */
    inline constexpr std::int32_t K_MAX_VOCABULARY_SIZE = 1 << 20;
} // namespace SushiAI::Language::Text
