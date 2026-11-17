/**************************************************************************/
/* protobuf_wire.hpp                                                      */
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
 * @file protobuf_wire.hpp
 * @brief Declares the cursor that reads the protobuf binary wire format over bytes in memory.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names how the bytes of one field are laid out, as the low three bits of a tag. */
        enum class WireType : std::uint8_t
        {
            Varint          = 0, /**< A base-128 integer of one to ten bytes. */
            Fixed64         = 1, /**< Eight bytes, low byte first. */
            LengthDelimited = 2, /**< A varint length, then that many bytes. */
            StartGroup      = 3, /**< The opening of a group, which ends at its EndGroup. */
            EndGroup        = 4, /**< The closing of a group. */
            Fixed32         = 5, /**< Four bytes, low byte first. */
        };

        /** @brief Holds the tag of one field: its number and how its bytes are laid out. */
        struct WireField
        {
            /** @brief Holds the field number, 1 to 2^29 - 1. */
            std::uint32_t number = 0;

            /** @brief Holds the wire type of the field's bytes. */
            WireType type = WireType::Varint;
        };

        /**
         * @brief Reads fields and values forward through protobuf wire-format bytes.
         *
         * It knows no schema. A caller reads a tag with next_field(), then reads the value or
         * skips it. A sub-message or a packed field is length-delimited: message() returns a
         * reader over its bytes, read until at_end(). Every failure is InvalidFile and names
         * the byte offset from the start of the outermost reader's bytes.
         */
        class ProtobufWire final
        {
            public:
                /**
                 * @brief Constructs a reader positioned at the first byte of @p bytes.
                 * @param bytes The encoded message; must outlive the reader.
                 * @param origin Names the bytes in a message; must outlive the reader.
                 */
                ProtobufWire(SushiRuntime::span<const std::byte> bytes,
                             std::string_view origin) noexcept;

                /** @brief Returns whether every byte has been read. */
                [[nodiscard]] bool at_end() const noexcept;

                /** @brief Returns the offset of the next byte from the start of the file. */
                [[nodiscard]] std::size_t offset() const noexcept;

                /**
                 * @brief Reads the tag of the next field.
                 * @pre !at_end().
                 * @return InvalidFile for a truncated or overlong tag, a field number of 0 or
                 *         above 2^29 - 1, or a wire type of 6 or 7.
                 */
                [[nodiscard]] Result<WireField> next_field() noexcept;

                /** @brief Reads a varint of up to ten bytes; InvalidFile when truncated. */
                [[nodiscard]] Result<std::uint64_t> read_varint() noexcept;

                /** @brief Reads a varint and undoes the zigzag encoding of a sint32 or sint64. */
                [[nodiscard]] Result<std::int64_t> read_zigzag() noexcept;

                /** @brief Reads four bytes, low byte first; InvalidFile when fewer remain. */
                [[nodiscard]] Result<std::uint32_t> read_fixed32() noexcept;

                /** @brief Reads eight bytes, low byte first; InvalidFile when fewer remain. */
                [[nodiscard]] Result<std::uint64_t> read_fixed64() noexcept;

                /**
                 * @brief Reads a length and returns a view of that many bytes, copying none.
                 * @return InvalidFile for a truncated length or a length past the end.
                 */
                [[nodiscard]] Result<SushiRuntime::span<const std::byte>>
                read_length_delimited() noexcept;

                /**
                 * @brief Reads a length-delimited value and returns a reader over its bytes.
                 * @return The reader, whose offsets continue those of this one; InvalidFile as
                 *         read_length_delimited().
                 */
                [[nodiscard]] Result<ProtobufWire> message() noexcept;

                /**
                 * @brief Skips the value of a field whose tag was just read.
                 * @param type The wire type next_field() returned; a group is skipped through
                 *             its EndGroup by counting depth, without comparing numbers.
                 * @return InvalidFile for a truncated value, a group that never ends, or an
                 *         EndGroup that closes nothing.
                 */
                [[nodiscard]] Status skip(WireType type) noexcept;

            private:
                /** @brief Constructs a reader over a sub-range that starts at @p base. */
                ProtobufWire(SushiRuntime::span<const std::byte> bytes, std::string_view origin,
                             std::size_t base) noexcept;

                /** @brief Holds the bytes this reader reads. */
                SushiRuntime::span<const std::byte> bytes_;

                /** @brief Names the bytes in a message. */
                std::string_view origin_;

                /** @brief Holds the offset of bytes_[0] from the start of the file. */
                std::size_t base_ = 0;

                /** @brief Holds the index in bytes_ of the next byte. */
                std::size_t at_ = 0;
        };

        /** @brief Returns the float whose IEEE 754 binary32 bits are @p bits. */
        [[nodiscard]] float float_from_bits(std::uint32_t bits) noexcept;

        /** @brief Returns the double whose IEEE 754 binary64 bits are @p bits. */
        [[nodiscard]] double double_from_bits(std::uint64_t bits) noexcept;
    } // namespace IO
} // namespace SushiAI
