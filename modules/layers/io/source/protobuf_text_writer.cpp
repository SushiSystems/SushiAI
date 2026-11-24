/**************************************************************************/
/* protobuf_text_writer.cpp                                               */
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
 * @file protobuf_text_writer.cpp
 * @brief Implements the writer of protobuf text format.
 * @author Mustafa Garip
 */

#include <SushiAI/io/protobuf_text_writer.hpp>

#include <cstddef>
#include <string>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the spaces one nesting level indents by. */
            constexpr std::size_t K_INDENT = 2;

            /** @brief Appends @p text quoted, its special bytes escaped. */
            void append_quoted(std::string& out, const std::string& text)
            {
                out.push_back('"');
                for (const char c : text)
                {
                    const auto byte = static_cast<unsigned char>(c);
                    switch (c)
                    {
                        case '\\': out.append("\\\\"); break;
                        case '"':  out.append("\\\""); break;
                        case '\n': out.append("\\n"); break;
                        case '\r': out.append("\\r"); break;
                        case '\t': out.append("\\t"); break;
                        default:
                            if (byte < 0x20u || byte > 0x7Eu)
                            {
                                out.push_back('\\');
                                out.push_back(static_cast<char>('0' + ((byte >> 6u) & 7u)));
                                out.push_back(static_cast<char>('0' + ((byte >> 3u) & 7u)));
                                out.push_back(static_cast<char>('0' + (byte & 7u)));
                            }
                            else
                            {
                                out.push_back(c);
                            }
                    }
                }
                out.push_back('"');
            }

            /** @brief Returns InvalidDescription unless @p field can be written. */
            [[nodiscard]] Status check_field(const TextField& field)
            {
                SA_FAIL_IF(field.name.empty(), StatusCode::InvalidDescription,
                           "a text-format field has no name");
                const bool bare = field.kind == TextKind::Number ||
                                  field.kind == TextKind::Identifier;
                SA_FAIL_IF(bare && (field.text.empty() ||
                                    field.text.find_first_of(" \t\r\n{}:\"'#") !=
                                        std::string::npos),
                           StatusCode::InvalidDescription,
                           "the field '{}' holds '{}', which is no bare token", field.name,
                           field.text);
                return Status();
            }

            /** @brief Appends the fields of @p message, indented by @p depth levels. */
            [[nodiscard]] Status append_fields(std::string& out, const TextField& message,
                                               std::size_t depth)
            {
                const std::string indent(depth * K_INDENT, ' ');
                for (const TextField& field : message.fields)
                {
                    SA_TRY(check_field(field));
                    out.append(indent).append(field.name);
                    if (field.kind == TextKind::Message)
                    {
                        out.append(" {\n");
                        SA_TRY(append_fields(out, field, depth + 1));
                        out.append(indent).append("}\n");
                        continue;
                    }

                    out.append(": ");
                    if (field.kind == TextKind::String)
                        append_quoted(out, field.text);
                    else
                        out.append(field.text);
                    out.push_back('\n');
                }

                return Status();
            }
        } // namespace

        Result<std::string> write_protobuf_text(const TextField& root) noexcept
        {
            SA_FAIL_IF(root.kind != TextKind::Message, StatusCode::InvalidDescription,
                       "the root of a text-format document must be a message");

            std::string out;
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return append_fields(out, root, 0);
            }));
            return out;
        }
    } // namespace IO
} // namespace SushiAI
