/**************************************************************************/
/* wire_values.hpp                                                        */
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
 * @file wire_values.hpp
 * @brief Declares the readers of one typed field value that the message tables share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
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
            /** @brief Names the file and the message a field is read from. */
            struct FieldSite
            {
                std::string_view origin;  /**< Names the file. */
                std::string_view message; /**< Names the message type, such as "NodeProto". */
            };

            /** @brief Holds the bytes of a length-delimited field and where they start. */
            struct BytesField
            {
                SushiRuntime::span<const std::byte> bytes; /**< The bytes, viewed in place. */
                std::size_t offset = 0; /**< Their offset from the start of the file. */
            };

            /** @brief Reads a length-delimited field as text. */
            [[nodiscard]] Result<std::string> read_text(IO::ProtobufWire& reader,
                                                        const IO::WireField& field,
                                                        const FieldSite& site) noexcept;

            /** @brief Reads a length-delimited field as a view of its bytes. */
            [[nodiscard]] Result<BytesField> read_bytes(IO::ProtobufWire& reader,
                                                        const IO::WireField& field,
                                                        const FieldSite& site) noexcept;

            /** @brief Reads a length-delimited field as a reader over its bytes. */
            [[nodiscard]] Result<IO::ProtobufWire> read_submessage(IO::ProtobufWire& reader,
                                                                   const IO::WireField& field,
                                                                   const FieldSite& site) noexcept;

            /** @brief Reads a varint field as a signed 64-bit integer. */
            [[nodiscard]] Result<std::int64_t> read_integer(IO::ProtobufWire& reader,
                                                            const IO::WireField& field,
                                                            const FieldSite& site) noexcept;

            /** @brief Reads a fixed32 field as a float. */
            [[nodiscard]] Result<float> read_real(IO::ProtobufWire& reader,
                                                  const IO::WireField& field,
                                                  const FieldSite& site) noexcept;

            /** @brief Reads a fixed64 field as a double. */
            [[nodiscard]] Result<double> read_double(IO::ProtobufWire& reader,
                                                     const IO::WireField& field,
                                                     const FieldSite& site) noexcept;

            /** @brief Appends the integers of a repeated varint field, packed or not, to @p out. */
            [[nodiscard]] Status read_integers(IO::ProtobufWire& reader,
                                               const IO::WireField& field, const FieldSite& site,
                                               std::vector<std::int64_t>& out) noexcept;

            /** @brief Appends the floats of a repeated fixed32 field, packed or not, to @p out. */
            [[nodiscard]] Status read_reals(IO::ProtobufWire& reader, const IO::WireField& field,
                                            const FieldSite& site,
                                            std::vector<float>& out) noexcept;

            /** @brief Appends the doubles of a repeated fixed64 field, packed or not, to @p out. */
            [[nodiscard]] Status read_doubles(IO::ProtobufWire& reader,
                                              const IO::WireField& field, const FieldSite& site,
                                              std::vector<double>& out) noexcept;
        } // namespace Detail
    } // namespace Interchange
} // namespace SushiAI
