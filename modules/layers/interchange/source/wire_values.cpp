/**************************************************************************/
/* wire_values.cpp                                                        */
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
 * @file wire_values.cpp
 * @brief Defines the readers of one typed field value that the message tables share.
 * @author Mustafa Garip
 */

#include "wire_values.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/protobuf_wire.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Returns the name of @p type for a failure message. */
                const char* wire_type_name(IO::WireType type) noexcept
                {
                    switch (type)
                    {
                        case IO::WireType::Varint:          return "varint";
                        case IO::WireType::Fixed64:         return "fixed64";
                        case IO::WireType::LengthDelimited: return "length-delimited";
                        case IO::WireType::StartGroup:      return "start-group";
                        case IO::WireType::EndGroup:        return "end-group";
                        case IO::WireType::Fixed32:         return "fixed32";
                    }
                    return "unknown";
                }

                /** @brief Fails unless @p field has the wire type @p expected. */
                Status expect_type(const IO::WireField& field, IO::WireType expected,
                                   const FieldSite& site) noexcept
                {
                    SA_FAIL_IF(field.type != expected, StatusCode::InvalidFile,
                               "'{}': {} field {} has the wire type {}, expected {}", site.origin,
                               site.message, field.number, wire_type_name(field.type),
                               wire_type_name(expected));
                    return Status();
                }
            } // namespace

            Result<std::string> read_text(IO::ProtobufWire& reader, const IO::WireField& field,
                                          const FieldSite& site) noexcept
            {
                SA_TRY(expect_type(field, IO::WireType::LengthDelimited, site));
                SA_TRY_ASSIGN(const SushiRuntime::span<const std::byte> bytes,
                              reader.read_length_delimited());
                return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            }

            Result<BytesField> read_bytes(IO::ProtobufWire& reader, const IO::WireField& field,
                                          const FieldSite& site) noexcept
            {
                SA_TRY(expect_type(field, IO::WireType::LengthDelimited, site));
                SA_TRY_ASSIGN(const SushiRuntime::span<const std::byte> bytes,
                              reader.read_length_delimited());
                BytesField out;
                out.bytes = bytes;
                out.offset = reader.offset() - bytes.size();
                return out;
            }

            Result<IO::ProtobufWire> read_submessage(IO::ProtobufWire& reader,
                                                     const IO::WireField& field,
                                                     const FieldSite& site) noexcept
            {
                SA_TRY(expect_type(field, IO::WireType::LengthDelimited, site));
                return reader.message();
            }

            Result<std::int64_t> read_integer(IO::ProtobufWire& reader, const IO::WireField& field,
                                              const FieldSite& site) noexcept
            {
                SA_TRY(expect_type(field, IO::WireType::Varint, site));
                SA_TRY_ASSIGN(const std::uint64_t value, reader.read_varint());
                return static_cast<std::int64_t>(value);
            }

            Result<float> read_real(IO::ProtobufWire& reader, const IO::WireField& field,
                                    const FieldSite& site) noexcept
            {
                SA_TRY(expect_type(field, IO::WireType::Fixed32, site));
                SA_TRY_ASSIGN(const std::uint32_t bits, reader.read_fixed32());
                return IO::float_from_bits(bits);
            }

            Result<double> read_double(IO::ProtobufWire& reader, const IO::WireField& field,
                                       const FieldSite& site) noexcept
            {
                SA_TRY(expect_type(field, IO::WireType::Fixed64, site));
                SA_TRY_ASSIGN(const std::uint64_t bits, reader.read_fixed64());
                return IO::double_from_bits(bits);
            }

            Status read_integers(IO::ProtobufWire& reader, const IO::WireField& field,
                                 const FieldSite& site, std::vector<std::int64_t>& out) noexcept
            {
                if (field.type != IO::WireType::LengthDelimited)
                {
                    SA_TRY_ASSIGN(const std::int64_t value, read_integer(reader, field, site));
                    out.push_back(value);
                    return Status();
                }
                SA_TRY_ASSIGN(IO::ProtobufWire packed, reader.message());
                while (!packed.at_end())
                {
                    SA_TRY_ASSIGN(const std::uint64_t value, packed.read_varint());
                    out.push_back(static_cast<std::int64_t>(value));
                }
                return Status();
            }

            Status read_reals(IO::ProtobufWire& reader, const IO::WireField& field,
                              const FieldSite& site, std::vector<float>& out) noexcept
            {
                if (field.type != IO::WireType::LengthDelimited)
                {
                    SA_TRY_ASSIGN(const float value, read_real(reader, field, site));
                    out.push_back(value);
                    return Status();
                }
                SA_TRY_ASSIGN(IO::ProtobufWire packed, reader.message());
                while (!packed.at_end())
                {
                    SA_TRY_ASSIGN(const std::uint32_t bits, packed.read_fixed32());
                    out.push_back(IO::float_from_bits(bits));
                }
                return Status();
            }

            Status read_doubles(IO::ProtobufWire& reader, const IO::WireField& field,
                                const FieldSite& site, std::vector<double>& out) noexcept
            {
                if (field.type != IO::WireType::LengthDelimited)
                {
                    SA_TRY_ASSIGN(const double value, read_double(reader, field, site));
                    out.push_back(value);
                    return Status();
                }
                SA_TRY_ASSIGN(IO::ProtobufWire packed, reader.message());
                while (!packed.at_end())
                {
                    SA_TRY_ASSIGN(const std::uint64_t bits, packed.read_fixed64());
                    out.push_back(IO::double_from_bits(bits));
                }
                return Status();
            }
        } // namespace Detail
    } // namespace Interchange
} // namespace SushiAI
