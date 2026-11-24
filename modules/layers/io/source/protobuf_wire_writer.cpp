/**************************************************************************/
/* protobuf_wire_writer.cpp                                               */
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
 * @file protobuf_wire_writer.cpp
 * @brief Implements the writer of protobuf wire-format bytes.
 * @author Mustafa Garip
 */

#include <SushiAI/io/protobuf_wire_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the wire type of a varint. */
            constexpr std::uint8_t K_VARINT = 0;

            /** @brief Holds the wire type of an eight-byte value. */
            constexpr std::uint8_t K_FIXED64 = 1;

            /** @brief Holds the wire type of a length-delimited value. */
            constexpr std::uint8_t K_LENGTH_DELIMITED = 2;

            /** @brief Holds the wire type of a four-byte value. */
            constexpr std::uint8_t K_FIXED32 = 5;
        } // namespace

        void ProtobufWireWriter::write_varint(std::uint32_t number, std::uint64_t value)
        {
            write_tag(number, K_VARINT);
            append_varint(value);
        }

        void ProtobufWireWriter::write_zigzag(std::uint32_t number, std::int64_t value)
        {
            const std::uint64_t bits = static_cast<std::uint64_t>(value);
            write_varint(number, (bits << 1u) ^ (value < 0 ? ~std::uint64_t(0) : 0u));
        }

        void ProtobufWireWriter::write_fixed32(std::uint32_t number, std::uint32_t value)
        {
            write_tag(number, K_FIXED32);
            append_little(value, 4);
        }

        void ProtobufWireWriter::write_fixed64(std::uint32_t number, std::uint64_t value)
        {
            write_tag(number, K_FIXED64);
            append_little(value, 8);
        }

        void ProtobufWireWriter::write_float(std::uint32_t number, float value)
        {
            write_fixed32(number, float_to_bits(value));
        }

        void ProtobufWireWriter::write_double(std::uint32_t number, double value)
        {
            write_fixed64(number, double_to_bits(value));
        }

        void ProtobufWireWriter::write_bytes(std::uint32_t number,
                                             SushiRuntime::span<const std::byte> bytes)
        {
            write_tag(number, K_LENGTH_DELIMITED);
            append_varint(bytes.size());
            bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
        }

        void ProtobufWireWriter::write_string(std::uint32_t number, std::string_view text)
        {
            write_bytes(number, SushiRuntime::span<const std::byte>(
                                    reinterpret_cast<const std::byte*>(text.data()), text.size()));
        }

        void ProtobufWireWriter::write_message(std::uint32_t number,
                                               const ProtobufWireWriter& message)
        {
            write_bytes(number, SushiRuntime::span<const std::byte>(message.bytes_.data(),
                                                                    message.bytes_.size()));
        }

        void ProtobufWireWriter::write_packed_floats(std::uint32_t number,
                                                     SushiRuntime::span<const float> values)
        {
            write_tag(number, K_LENGTH_DELIMITED);
            append_varint(values.size() * 4u);
            bytes_.reserve(bytes_.size() + values.size() * 4u);
            for (const float value : values)
                append_little(float_to_bits(value), 4);
        }

        void ProtobufWireWriter::write_packed_varints(std::uint32_t number,
                                                      SushiRuntime::span<const std::int64_t> values)
        {
            ProtobufWireWriter packed;
            for (const std::int64_t value : values)
                packed.append_varint(static_cast<std::uint64_t>(value));
            write_message(number, packed);
        }

        const std::vector<std::byte>& ProtobufWireWriter::bytes() const noexcept
        {
            return bytes_;
        }

        std::vector<std::byte> ProtobufWireWriter::take() noexcept
        {
            return std::exchange(bytes_, std::vector<std::byte>());
        }

        void ProtobufWireWriter::write_tag(std::uint32_t number, std::uint8_t type)
        {
            append_varint((static_cast<std::uint64_t>(number) << 3u) | type);
        }

        void ProtobufWireWriter::append_varint(std::uint64_t value)
        {
            while (value >= 0x80u)
            {
                bytes_.push_back(static_cast<std::byte>((value & 0x7Fu) | 0x80u));
                value >>= 7u;
            }
            bytes_.push_back(static_cast<std::byte>(value));
        }

        void ProtobufWireWriter::append_little(std::uint64_t value, std::size_t count)
        {
            for (std::size_t i = 0; i < count; ++i)
                bytes_.push_back(static_cast<std::byte>((value >> (8u * i)) & 0xFFu));
        }

        std::uint32_t float_to_bits(float value) noexcept
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }

        std::uint64_t double_to_bits(double value) noexcept
        {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }
    } // namespace IO
} // namespace SushiAI
