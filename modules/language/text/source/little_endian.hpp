/**************************************************************************/
/* little_endian.hpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file little_endian.hpp
 * @brief Defines the conversions between a 32-bit word and its four little-endian bytes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

/** @brief Holds the units of the text module that no caller names. */
namespace SushiAI::Language::Text::Detail
{
    /** @brief Holds the byte count of one 32-bit field of a file. */
    inline constexpr std::size_t K_WORD_BYTES = 4;

    /** @brief Writes a value as four bytes at an address, the least significant first. */
    inline void store_u32(char* bytes, std::uint32_t value) noexcept
    {
        for (std::size_t i = 0; i < K_WORD_BYTES; ++i)
            bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
    }

    /** @brief Returns the value of four bytes at an address, the least significant first. */
    [[nodiscard]] inline std::uint32_t load_u32(const char* bytes) noexcept
    {
        std::uint32_t value = 0;
        for (std::size_t i = 0; i < K_WORD_BYTES; ++i)
        {
            const std::uint32_t byte = static_cast<unsigned char>(bytes[i]);
            value |= byte << (8 * i);
        }

        return value;
    }
} // namespace SushiAI::Language::Text::Detail
