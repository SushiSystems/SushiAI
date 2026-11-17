/**************************************************************************/
/* protobuf_text.hpp                                                      */
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
 * @file protobuf_text.hpp
 * @brief Declares the field tree of a protobuf text-format document and the parser that builds it.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names what the value of one text-format field is. */
        enum class TextKind : std::uint8_t
        {
            Number,     /**< A number as written, in text; the schema says if it is real. */
            String,     /**< A quoted string with its escapes decoded, in text. */
            Identifier, /**< A bare word, such as an enum value or true, in text. */
            Message,    /**< A nested message, its fields in fields. */
        };

        /**
         * @struct TextField
         * @brief Holds one field of a text-format message, or the root message itself.
         *
         * A repeated field is several entries of the same name in fields, in document order.
         * The root holds no name; its kind is Message.
         */
        struct TextField
        {
            /** @brief Holds the field's name; empty for the root. */
            std::string name;

            /** @brief Holds what the value is. */
            TextKind kind = TextKind::Message;

            /** @brief Holds the value of a Number, a String or an Identifier. */
            std::string text;

            /** @brief Holds the fields of a Message, in document order. */
            std::vector<TextField> fields;

            /** @brief Holds the line, from 1, where the field's name starts; 0 for the root. */
            std::size_t line = 0;

            /** @brief Holds the column, from 1, where the field's name starts; 0 for the root. */
            std::size_t column = 0;
        };

        /**
         * @brief Returns the first field of @p message named @p name, or null when none is.
         * @param message A field of kind Message.
         */
        [[nodiscard]] const TextField* find_field(const TextField& message,
                                                  std::string_view name) noexcept;

        /** @brief Returns every field of @p message named @p name, in document order. */
        [[nodiscard]] std::vector<const TextField*> find_fields(const TextField& message,
                                                                std::string_view name);

        /**
         * @brief Parses a protobuf text-format document, such as a Caffe prototxt, with no schema.
         * The colon before a brace is optional and adjacent strings join. A list in brackets,
         * an extension name and the angle-bracket form of a message are refused.
         * @param origin Names the document in a message.
         * @return The root message; InvalidFile for a stray or unclosed brace, a bad token or a
         *         malformed string or number, naming the line and column.
         */
        [[nodiscard]] Result<TextField> parse_protobuf_text(std::string_view text,
                                                            std::string_view origin) noexcept;
    } // namespace IO
} // namespace SushiAI
