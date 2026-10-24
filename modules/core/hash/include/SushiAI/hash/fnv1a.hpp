/**************************************************************************/
/* fnv1a.hpp                                                              */
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
 * @file fnv1a.hpp
 * @brief Declares the 64-bit FNV-1a hash, folded a byte, a text or a word at a time.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace SushiAI
{
    /** @brief Holds the hashes every module may fold its bytes into. */
    namespace Hash
    {
        /** @brief Holds the 64-bit FNV-1a offset basis, the hash of no byte. */
        constexpr std::uint64_t K_FNV1A_OFFSET_BASIS = 0xcbf29ce484222325ull;

        /** @brief Holds the 64-bit FNV-1a prime. */
        constexpr std::uint64_t K_FNV1A_PRIME = 0x100000001b3ull;

        /** @brief Returns @p hash with @p byte folded in. */
        [[nodiscard]] constexpr std::uint64_t fnv1a_byte(std::uint64_t hash,
                                                         unsigned char byte) noexcept
        {
            return (hash ^ byte) * K_FNV1A_PRIME;
        }

        /** @brief Returns @p hash with the bytes of @p text folded in, first to last. */
        [[nodiscard]] constexpr std::uint64_t fnv1a_text(std::uint64_t hash,
                                                         std::string_view text) noexcept
        {
            for (const char byte : text)
                hash = fnv1a_byte(hash, static_cast<unsigned char>(byte));

            return hash;
        }

        /** @brief Returns @p hash with the eight bytes of @p word folded in, low byte first. */
        [[nodiscard]] constexpr std::uint64_t fnv1a_word(std::uint64_t hash,
                                                         std::uint64_t word) noexcept
        {
            for (std::size_t byte = 0; byte < sizeof(word); ++byte)
                hash = fnv1a_byte(hash, static_cast<unsigned char>((word >> (8 * byte)) & 0xffu));

            return hash;
        }

    } // namespace Hash
} // namespace SushiAI
