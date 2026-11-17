/**************************************************************************/
/* float_widening.hpp                                                     */
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
 * @file float_widening.hpp
 * @brief Defines the exact widening of 16-bit floating-point values to float32 for the file readers.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace SushiAI::IO::Detail
{
    /** @brief Widens the IEEE 754 binary16 value @p bits to a float, exactly. */
    inline float widen_half(std::uint16_t bits) noexcept
    {
        const std::uint32_t sign = static_cast<std::uint32_t>(bits >> 15u) << 31u;
        const std::uint32_t exponent = (bits >> 10u) & 0x1Fu;
        const std::uint32_t mantissa = bits & 0x3FFu;
        float magnitude = 0.0f;
        if (exponent == 0)
            magnitude = std::ldexp(static_cast<float>(mantissa), -24);
        else if (exponent == 0x1Fu)
            magnitude = mantissa == 0 ? INFINITY : NAN;
        else
            magnitude = std::ldexp(static_cast<float>(mantissa | 0x400u),
                                   static_cast<int>(exponent) - 25);
        return sign != 0 ? -magnitude : magnitude;
    }

    /** @brief Widens the bfloat16 value @p bits, the high half of a float32, to a float, exactly. */
    inline float widen_bfloat16(std::uint16_t bits) noexcept
    {
        const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16u;
        float out = 0.0f;
        std::memcpy(&out, &word, sizeof(out));
        return out;
    }
} // namespace SushiAI::IO::Detail
