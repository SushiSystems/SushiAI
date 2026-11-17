/**************************************************************************/
/* protobuf_wire.cpp                                                      */
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
 * @file protobuf_wire.cpp
 * @brief Defines the cursor that reads the protobuf binary wire format over bytes in memory.
 * @author Mustafa Garip
 */

#include <SushiAI/io/protobuf_wire.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the most bytes a varint may take. */
            constexpr std::size_t K_MAX_VARINT_BYTES = 10;

            /** @brief Holds the largest field number the wire format allows. */
            constexpr std::uint64_t K_MAX_FIELD_NUMBER = (1u << 29) - 1u;

            /** @brief Holds the bytes of a fixed32 value. */
            constexpr std::size_t K_FIXED32_BYTES = 4;

            /** @brief Holds the bytes of a fixed64 value. */
            constexpr std::size_t K_FIXED64_BYTES = 8;
        } // namespace

        ProtobufWire::ProtobufWire(SushiRuntime::span<const std::byte> bytes,
                                   std::string_view origin) noexcept
            : bytes_(bytes), origin_(origin)
        {

        }

        ProtobufWire::ProtobufWire(SushiRuntime::span<const std::byte> bytes,
                                   std::string_view origin, std::size_t base) noexcept
            : bytes_(bytes), origin_(origin), base_(base)
        {

        }

        bool ProtobufWire::at_end() const noexcept
        {
            return at_ >= bytes_.size();
        }

        std::size_t ProtobufWire::offset() const noexcept
        {
            return base_ + at_;
        }

        Result<std::uint64_t> ProtobufWire::read_varint() noexcept
        {
            const std::size_t start = at_;
            std::uint64_t value = 0;
            for (std::size_t i = 0; i < K_MAX_VARINT_BYTES; ++i)
            {
                SA_FAIL_IF(start + i >= bytes_.size(), StatusCode::InvalidFile,
                           "'{}': the varint at byte {} is truncated", origin_, base_ + start);
                const std::uint8_t byte = static_cast<std::uint8_t>(bytes_[start + i]);
                // The tenth byte holds bit 63 only; any higher bit overflows 64 bits.
                SA_FAIL_IF(i == K_MAX_VARINT_BYTES - 1 && byte > 1u, StatusCode::InvalidFile,
                           "'{}': the varint at byte {} does not fit 64 bits", origin_,
                           base_ + start);
                value |= static_cast<std::uint64_t>(byte & 0x7Fu) << (7u * i);
                if ((byte & 0x80u) == 0)
                {
                    at_ = start + i + 1;
                    return value;
                }
            }
            // Every path through the loop returns: the tenth byte carries no continuation bit.
            SA_FAIL_IF(true, StatusCode::InvalidFile,
                       "'{}': the varint at byte {} does not fit 64 bits", origin_, base_ + start);
            return std::uint64_t(0);
        }

        Result<std::int64_t> ProtobufWire::read_zigzag() noexcept
        {
            SA_TRY_ASSIGN(const std::uint64_t raw, read_varint());
            return static_cast<std::int64_t>(raw >> 1) ^ -static_cast<std::int64_t>(raw & 1u);
        }

        Result<std::uint32_t> ProtobufWire::read_fixed32() noexcept
        {
            SA_FAIL_IF(bytes_.size() - at_ < K_FIXED32_BYTES, StatusCode::InvalidFile,
                       "'{}': the fixed32 at byte {} is truncated", origin_, base_ + at_);
            std::uint32_t value = 0;
            for (std::size_t i = 0; i < K_FIXED32_BYTES; ++i)
                value |= static_cast<std::uint32_t>(bytes_[at_ + i]) << (8u * i);
            at_ += K_FIXED32_BYTES;
            return value;
        }

        Result<std::uint64_t> ProtobufWire::read_fixed64() noexcept
        {
            SA_FAIL_IF(bytes_.size() - at_ < K_FIXED64_BYTES, StatusCode::InvalidFile,
                       "'{}': the fixed64 at byte {} is truncated", origin_, base_ + at_);
            std::uint64_t value = 0;
            for (std::size_t i = 0; i < K_FIXED64_BYTES; ++i)
                value |= static_cast<std::uint64_t>(bytes_[at_ + i]) << (8u * i);
            at_ += K_FIXED64_BYTES;
            return value;
        }

        Result<SushiRuntime::span<const std::byte>> ProtobufWire::read_length_delimited() noexcept
        {
            const std::size_t start = at_;
            SA_TRY_ASSIGN(const std::uint64_t length, read_varint());
            SA_FAIL_IF(length > bytes_.size() - at_, StatusCode::InvalidFile,
                       "'{}': the length-delimited value at byte {} claims {} bytes and {} remain",
                       origin_, base_ + start, length, bytes_.size() - at_);
            const SushiRuntime::span<const std::byte> out =
                bytes_.subspan(at_, static_cast<std::size_t>(length));
            at_ += static_cast<std::size_t>(length);
            return out;
        }

        Result<ProtobufWire> ProtobufWire::message() noexcept
        {
            SA_TRY_ASSIGN(const SushiRuntime::span<const std::byte> inner,
                          read_length_delimited());
            return ProtobufWire(inner, origin_, base_ + at_ - inner.size());
        }

        Result<WireField> ProtobufWire::next_field() noexcept
        {
            const std::size_t start = base_ + at_;
            SA_TRY_ASSIGN(const std::uint64_t tag, read_varint());
            const std::uint64_t number = tag >> 3;
            const std::uint64_t type = tag & 7u;
            SA_FAIL_IF(number == 0 || number > K_MAX_FIELD_NUMBER, StatusCode::InvalidFile,
                       "'{}': the tag at byte {} has the field number {}", origin_, start, number);
            SA_FAIL_IF(type > static_cast<std::uint64_t>(WireType::Fixed32),
                       StatusCode::InvalidFile, "'{}': the tag at byte {} has the wire type {}",
                       origin_, start, type);
            WireField field;
            field.number = static_cast<std::uint32_t>(number);
            field.type = static_cast<WireType>(type);
            return field;
        }

        Status ProtobufWire::skip(WireType type) noexcept
        {
            std::size_t depth = type == WireType::StartGroup ? 1 : 0;
            while (true)
            {
                switch (type)
                {
                    case WireType::Varint:
                        SA_TRY(read_varint());
                        break;
                    case WireType::Fixed64:
                        SA_TRY(read_fixed64());
                        break;
                    case WireType::LengthDelimited:
                        SA_TRY(read_length_delimited());
                        break;
                    case WireType::Fixed32:
                        SA_TRY(read_fixed32());
                        break;
                    case WireType::StartGroup:
                        break;
                    case WireType::EndGroup:
                        SA_FAIL_IF(depth == 0, StatusCode::InvalidFile,
                                   "'{}': the end-group tag before byte {} closes no group",
                                   origin_, base_ + at_);
                        --depth;
                        break;
                }
                if (depth == 0)
                    return Status();

                SA_FAIL_IF(at_ >= bytes_.size(), StatusCode::InvalidFile,
                           "'{}': a group ends at byte {} without its end-group tag", origin_,
                           base_ + at_);
                SA_TRY_ASSIGN(const WireField inner, next_field());
                type = inner.type;
                if (type == WireType::StartGroup)
                    ++depth;
            }
        }

        float float_from_bits(std::uint32_t bits) noexcept
        {
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

        double double_from_bits(std::uint64_t bits) noexcept
        {
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
    } // namespace IO
} // namespace SushiAI
