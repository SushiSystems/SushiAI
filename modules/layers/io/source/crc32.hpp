/**************************************************************************/
/* crc32.hpp                                                              */
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
 * @file crc32.hpp
 * @brief Defines the CRC-32 of a zip entry, the IEEE 802.3 polynomial, for the file readers.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI::IO::Detail
{
    /** @brief Holds the reflected form of the polynomial 0x04C11DB7. */
    constexpr std::uint32_t K_CRC32_POLYNOMIAL = 0xEDB88320u;

    /** @brief Holds the CRC-32 of every byte value, computed when the program is compiled. */
    struct Crc32Table
    {
        /** @brief Holds the CRC-32 remainder of each of the 256 byte values. */
        std::uint32_t entries[256];

        /** @brief Computes the 256 remainders. */
        constexpr Crc32Table() noexcept : entries()
        {
            for (std::uint32_t value = 0; value < 256; ++value)
            {
                std::uint32_t remainder = value;
                for (int bit = 0; bit < 8; ++bit)
                    remainder = (remainder & 1u) != 0 ? K_CRC32_POLYNOMIAL ^ (remainder >> 1u)
                                                      : remainder >> 1u;
                entries[value] = remainder;
            }
        }
    };

    /** @brief Holds the table crc32 reads. */
    inline constexpr Crc32Table K_CRC32_TABLE{};

    /** @brief Returns the CRC-32 of @p bytes, as a zip header states it. */
    [[nodiscard]] inline std::uint32_t crc32(SushiRuntime::span<const std::byte> bytes) noexcept
    {
        std::uint32_t crc = 0xFFFFFFFFu;
        for (const std::byte byte : bytes)
            crc = K_CRC32_TABLE.entries[(crc ^ static_cast<std::uint32_t>(byte)) & 0xFFu] ^
                  (crc >> 8u);
        return ~crc;
    }
} // namespace SushiAI::IO::Detail
