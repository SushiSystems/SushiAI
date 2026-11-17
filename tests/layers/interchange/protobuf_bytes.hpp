/**************************************************************************/
/* protobuf_bytes.hpp                                                     */
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
 * @file protobuf_bytes.hpp
 * @brief Declares the helpers that assemble protobuf wire-format messages by hand for the
 *        interchange tests.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Test
    {
        namespace Wire
        {
            /** @brief Returns @p value as a base-128 varint, low group first. */
            inline std::string varint(std::uint64_t value)
            {
                std::string out;
                while (value >= 0x80u)
                {
                    out.push_back(static_cast<char>((value & 0x7Fu) | 0x80u));
                    value >>= 7u;
                }
                out.push_back(static_cast<char>(value));
                return out;
            }

            /** @brief Returns the tag of field @p number with wire type @p wire_type. */
            inline std::string tag(std::uint32_t number, std::uint32_t wire_type)
            {
                return varint((static_cast<std::uint64_t>(number) << 3u) | wire_type);
            }

            /** @brief Returns a varint field; a negative value takes ten bytes. */
            inline std::string integer_field(std::uint32_t number, std::int64_t value)
            {
                return tag(number, 0) + varint(static_cast<std::uint64_t>(value));
            }

            /** @brief Returns a length-delimited field holding @p payload. */
            inline std::string bytes_field(std::uint32_t number, std::string_view payload)
            {
                return tag(number, 2) + varint(payload.size()) + std::string(payload);
            }

            /** @brief Returns the four bytes of @p value, low byte first. */
            inline std::string float_bytes(float value)
            {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                std::string out;
                for (unsigned i = 0; i < 4; ++i)
                    out.push_back(static_cast<char>((bits >> (8u * i)) & 0xFFu));
                return out;
            }

            /** @brief Returns a fixed32 field holding @p value. */
            inline std::string float_field(std::uint32_t number, float value)
            {
                return tag(number, 5) + float_bytes(value);
            }

            /** @brief Returns a packed repeated float field. */
            inline std::string packed_floats(std::uint32_t number, const std::vector<float>& values)
            {
                std::string payload;
                for (float value : values)
                    payload += float_bytes(value);
                return bytes_field(number, payload);
            }

            /** @brief Returns a packed repeated varint field. */
            inline std::string packed_integers(std::uint32_t number,
                                               const std::vector<std::int64_t>& values)
            {
                std::string payload;
                for (std::int64_t value : values)
                    payload += varint(static_cast<std::uint64_t>(value));
                return bytes_field(number, payload);
            }

            /** @brief Returns the bytes of @p text, one byte per character. */
            inline std::vector<std::byte> to_bytes(std::string_view text)
            {
                std::vector<std::byte> out(text.size());
                for (std::size_t i = 0; i < text.size(); ++i)
                    out[i] = static_cast<std::byte>(static_cast<unsigned char>(text[i]));
                return out;
            }

            /** @brief Returns a view of @p bytes. */
            inline SushiRuntime::span<const std::byte> view_of(const std::vector<std::byte>& bytes)
            {
                return SushiRuntime::span<const std::byte>(bytes.data(), bytes.size());
            }
        } // namespace Wire
    } // namespace Test
} // namespace SushiAI
