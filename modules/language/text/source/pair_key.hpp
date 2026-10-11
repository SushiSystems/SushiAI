/**************************************************************************/
/* pair_key.hpp                                                           */
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
 * @file pair_key.hpp
 * @brief Defines the 64-bit key of a pair of tokens.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/language/text/token.hpp>

/** @brief Holds the units of the text module that no caller names. */
namespace SushiAI::Language::Text::Detail
{
    /**
     * @brief Returns a key that is unique to a pair, the left token in its high word.
     * @pre Neither id is negative.
     */
    [[nodiscard]] constexpr std::uint64_t pair_key(TokenID left, TokenID right) noexcept
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(right));
    }
} // namespace SushiAI::Language::Text::Detail
