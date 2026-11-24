/**************************************************************************/
/* protobuf_wire_writer.hpp                                               */
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
 * @file protobuf_wire_writer.hpp
 * @brief Declares the writer of protobuf wire-format bytes, the inverse of ProtobufWire.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Appends fields to protobuf wire-format bytes.
         *
         * It knows no schema: a caller states each field's number and value. A sub-message is
         * written by a second writer whose bytes go in as one length-delimited field. A field
         * number must lie in 1 to 2^29 - 1. Every method may throw std::bad_alloc; a caller
         * that returns a Result runs them inside a seam guard.
         */
        class ProtobufWireWriter final
        {
            public:
                /** @brief Appends a varint field. */
                void write_varint(std::uint32_t number, std::uint64_t value);

                /** @brief Appends a varint field holding the zigzag form of a sint32 or sint64. */
                void write_zigzag(std::uint32_t number, std::int64_t value);

                /** @brief Appends a four-byte field, low byte first. */
                void write_fixed32(std::uint32_t number, std::uint32_t value);

                /** @brief Appends an eight-byte field, low byte first. */
                void write_fixed64(std::uint32_t number, std::uint64_t value);

                /** @brief Appends a float as a four-byte field of its IEEE 754 bits. */
                void write_float(std::uint32_t number, float value);

                /** @brief Appends a double as an eight-byte field of its IEEE 754 bits. */
                void write_double(std::uint32_t number, double value);

                /** @brief Appends a length-delimited field holding @p bytes. */
                void write_bytes(std::uint32_t number, SushiRuntime::span<const std::byte> bytes);

                /** @brief Appends a length-delimited field holding the bytes of @p text. */
                void write_string(std::uint32_t number, std::string_view text);

                /** @brief Appends the bytes of @p message as one length-delimited field. */
                void write_message(std::uint32_t number, const ProtobufWireWriter& message);

                /** @brief Appends a packed field of four-byte floats. */
                void write_packed_floats(std::uint32_t number,
                                         SushiRuntime::span<const float> values);

                /** @brief Appends a packed field of varints; a negative value takes ten bytes. */
                void write_packed_varints(std::uint32_t number,
                                          SushiRuntime::span<const std::int64_t> values);

                /** @brief Returns the bytes written so far. */
                [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept;

                /** @brief Moves the bytes out; the writer is empty afterwards. */
                [[nodiscard]] std::vector<std::byte> take() noexcept;

            private:
                /** @brief Appends the tag of field @p number with wire type @p type. */
                void write_tag(std::uint32_t number, std::uint8_t type);

                /** @brief Appends @p value as a base-128 varint. */
                void append_varint(std::uint64_t value);

                /** @brief Appends the low @p count bytes of @p value, low byte first. */
                void append_little(std::uint64_t value, std::size_t count);

                /** @brief Holds the bytes written so far. */
                std::vector<std::byte> bytes_;
        };

        /** @brief Returns the IEEE 754 binary32 bits of @p value. */
        [[nodiscard]] std::uint32_t float_to_bits(float value) noexcept;

        /** @brief Returns the IEEE 754 binary64 bits of @p value. */
        [[nodiscard]] std::uint64_t double_to_bits(double value) noexcept;
    } // namespace IO
} // namespace SushiAI
