/**************************************************************************/
/* protobuf_text_writer.hpp                                               */
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
 * @file protobuf_text_writer.hpp
 * @brief Declares the writer of protobuf text format, the inverse of parse_protobuf_text.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <string>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/protobuf_text.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Writes the fields of @p root, a Message, as a protobuf text-format document.
         * A field is a line `name: value`, a message `name {` to `}`, two spaces a level; a
         * string is quoted, `\\` `\"` `\n` `\r` `\t` escaped, other bytes outside 0x20 to 0x7E
         * as three-digit octal. The root's own name and position are not written.
         * @return The text; InvalidDescription naming a field with no name, a root of another
         *         kind, or a Number or Identifier whose text is empty or holds a space.
         */
        [[nodiscard]] Result<std::string> write_protobuf_text(const TextField& root) noexcept;
    } // namespace IO
} // namespace SushiAI
