/**************************************************************************/
/* json_writer.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file json_writer.cpp
 * @brief Implements the writer that turns a configuration value into JSON text.
 * @author Mustafa Garip
 */

#include <SushiAI/config/json_writer.hpp>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <new>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Holds the number of spaces one level of an expanded container indents. */
            constexpr std::size_t K_INDENT_WIDTH = 4;

            /** @brief Holds a buffer size above the longest spelling of any number. */
            constexpr std::size_t K_NUMBER_CAPACITY = 32;

            /** @brief Holds the first code unit that a JSON string may carry unescaped. */
            constexpr unsigned char K_FIRST_PLAIN_CHARACTER = 0x20;

            /** @brief Lists the digits of a four-digit hexadecimal escape. */
            constexpr std::string_view K_HEX_DIGITS = "0123456789abcdef";

            /** @brief Names the value an accessor reads, for a diagnostic no call can raise. */
            constexpr std::string_view K_WRITTEN_VALUE = "a value being written";

            /** @brief Reports whether @p text is well-formed UTF-8. */
            bool is_utf8(std::string_view text) noexcept
            {
                std::size_t index = 0;
                while (index < text.size())
                {
                    const unsigned char lead = static_cast<unsigned char>(text[index]);
                    if (lead < 0x80)
                    {
                        ++index;
                        continue;
                    }

                    std::size_t length = 0;
                    unsigned char lowest = 0x80;
                    unsigned char highest = 0xBF;

                    // The narrowed second byte refuses overlongs, surrogates and U+110000 up.
                    if (lead >= 0xC2 && lead <= 0xDF)
                    {
                        length = 2;
                    }
                    else if (lead >= 0xE0 && lead <= 0xEF)
                    {
                        length = 3;
                        lowest = lead == 0xE0 ? 0xA0 : 0x80;
                        highest = lead == 0xED ? 0x9F : 0xBF;
                    }
                    else if (lead >= 0xF0 && lead <= 0xF4)
                    {
                        length = 4;
                        lowest = lead == 0xF0 ? 0x90 : 0x80;
                        highest = lead == 0xF4 ? 0x8F : 0xBF;
                    }
                    else
                    {
                        return false;
                    }

                    if (text.size() - index < length)
                        return false;

                    const unsigned char second = static_cast<unsigned char>(text[index + 1]);
                    if (second < lowest || second > highest)
                        return false;

                    for (std::size_t offset = 2; offset < length; ++offset)
                    {
                        const unsigned char next =
                            static_cast<unsigned char>(text[index + offset]);
                        if (next < 0x80 || next > 0xBF)
                            return false;
                    }

                    index += length;
                }
                return true;
            }

            /** @brief Appends a line end and the indent of a member at @p depth. */
            void append_line_start(std::string& out, std::size_t depth)
            {
                out += '\n';
                out.append(depth * K_INDENT_WIDTH, ' ');
            }

            /** @brief Appends the hexadecimal escape of the control character @p character. */
            void append_unicode_escape(std::string& out, unsigned char character)
            {
                out += "\\u00";
                out += K_HEX_DIGITS[character >> 4];
                out += K_HEX_DIGITS[character & 0x0F];
            }

            /**
             * @brief Appends @p text as a JSON string, quoted and escaped.
             * @pre is_utf8(text).
             */
            void append_string(std::string& out, std::string_view text)
            {
                out += '"';
                for (const char character : text)
                {
                    switch (character)
                    {
                        case '"':  out += "\\\""; break;
                        case '\\': out += "\\\\"; break;
                        case '\b': out += "\\b";  break;
                        case '\f': out += "\\f";  break;
                        case '\n': out += "\\n";  break;
                        case '\r': out += "\\r";  break;
                        case '\t': out += "\\t";  break;
                        default:
                            if (static_cast<unsigned char>(character) < K_FIRST_PLAIN_CHARACTER)
                                append_unicode_escape(out, static_cast<unsigned char>(character));
                            else
                                out += character;
                            break;
                    }
                }
                out += '"';
            }

            /** @brief Appends @p value in decimal digits. */
            void append_integer(std::string& out, std::int64_t value)
            {
                // K_NUMBER_CAPACITY is above the 20 characters of the longest int64.
                char digits[K_NUMBER_CAPACITY];
                const std::to_chars_result end =
                    std::to_chars(digits, digits + K_NUMBER_CAPACITY, value);
                out.append(digits, static_cast<std::size_t>(end.ptr - digits));
            }

            /**
             * @brief Appends the shortest decimal spelling that parses back to @p value.
             * @pre std::isfinite(value).
             */
            void append_real(std::string& out, double value)
            {
                // K_NUMBER_CAPACITY is above the 24 characters of the longest shortest double.
                char digits[K_NUMBER_CAPACITY];
                const std::to_chars_result end =
                    std::to_chars(digits, digits + K_NUMBER_CAPACITY, value);
                const std::string_view written(digits,
                                               static_cast<std::size_t>(end.ptr - digits));
                out.append(written);

                // A spelling with neither a point nor an exponent would read back as an integer.
                if (written.find_first_of(".e") == std::string_view::npos)
                    out += ".0";
            }

            /** @brief Declares write_value ahead of the two container writers that call it. */
            Status write_value(std::string& out, const Value& value, std::size_t depth,
                               std::size_t expanded_depth);

            /** @brief Appends the array @p value, which sits at @p depth. */
            Status write_array(std::string& out, const Value& value, std::size_t depth,
                               std::size_t expanded_depth)
            {
                SA_TRY_ASSIGN(const std::vector<Value>& entries, value.as_array(K_WRITTEN_VALUE));
                if (entries.empty())
                {
                    out += "[]";
                    return Status();
                }

                const bool expanded = depth < expanded_depth;
                out += '[';
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    if (i != 0)
                        out += expanded ? "," : ", ";
                    if (expanded)
                        append_line_start(out, depth + 1);

                    const Status written = write_value(out, entries[i], depth + 1, expanded_depth);
                    if (!written.is_ok())
                        return SA_ANNOTATE(written, "entry {}", i);
                }

                if (expanded)
                    append_line_start(out, depth);
                out += ']';
                return Status();
            }

            /** @brief Refuses a key of @p keys that is not UTF-8 or that an earlier key states. */
            Status check_keys(const std::vector<std::string>& keys)
            {
                for (std::size_t i = 0; i < keys.size(); ++i)
                {
                    SA_FAIL_IF(!is_utf8(keys[i]), StatusCode::InvalidDescription,
                               "key {} of an object holds bytes that are not UTF-8", i);

                    for (std::size_t earlier = 0; earlier < i; ++earlier)
                        SA_FAIL_IF(keys[earlier] == keys[i], StatusCode::InvalidDescription,
                                   "an object states the key '{}' twice, and a reader would "
                                   "keep one of the two values", keys[i]);
                }
                return Status();
            }

            /** @brief Appends the object @p value, which sits at @p depth. */
            Status write_object(std::string& out, const Value& value, std::size_t depth,
                                std::size_t expanded_depth)
            {
                const std::vector<std::string>& keys = value.keys();
                if (keys.empty())
                {
                    out += "{}";
                    return Status();
                }

                SA_TRY(check_keys(keys));

                const bool expanded = depth < expanded_depth;
                out += '{';
                for (std::size_t i = 0; i < keys.size(); ++i)
                {
                    if (i != 0)
                        out += ',';
                    if (expanded)
                        append_line_start(out, depth + 1);
                    else
                        out += ' ';

                    append_string(out, keys[i]);
                    out += ": ";

                    SA_TRY_ASSIGN(const Value& member, value.at(keys[i], K_WRITTEN_VALUE));
                    const Status written = write_value(out, member, depth + 1, expanded_depth);
                    if (!written.is_ok())
                        return SA_ANNOTATE(written, "key '{}'", keys[i]);
                }

                if (expanded)
                    append_line_start(out, depth);
                else
                    out += ' ';
                out += '}';
                return Status();
            }

            /** @brief Appends @p value, which sits at @p depth, in the spelling of its kind. */
            Status write_value(std::string& out, const Value& value, std::size_t depth,
                               std::size_t expanded_depth)
            {
                switch (value.kind())
                {
                    case Kind::Null:
                        out += "null";
                        return Status();

                    case Kind::Bool:
                    {
                        SA_TRY_ASSIGN(const bool flag, value.as_bool(K_WRITTEN_VALUE));
                        out += flag ? "true" : "false";
                        return Status();
                    }

                    case Kind::Int:
                    {
                        SA_TRY_ASSIGN(const std::int64_t whole, value.as_int(K_WRITTEN_VALUE));
                        append_integer(out, whole);
                        return Status();
                    }

                    case Kind::Real:
                    {
                        SA_TRY_ASSIGN(const double real, value.as_real(K_WRITTEN_VALUE));
                        SA_FAIL_IF(!std::isfinite(real), StatusCode::InvalidDescription,
                                   "a number is {}, and JSON spells finite numbers only",
                                   std::isnan(real) ? "not a number" : "infinite");
                        append_real(out, real);
                        return Status();
                    }

                    case Kind::String:
                    {
                        SA_TRY_ASSIGN(const std::string& text, value.as_string(K_WRITTEN_VALUE));
                        SA_FAIL_IF(!is_utf8(text), StatusCode::InvalidDescription,
                                   "a string holds bytes that are not UTF-8");
                        append_string(out, text);
                        return Status();
                    }

                    case Kind::Array:
                        return write_array(out, value, depth, expanded_depth);

                    case Kind::Object:
                        return write_object(out, value, depth, expanded_depth);
                }

                return Status::from_message(StatusCode::Unsupported, []()
                {
                    return std::string("a value is of a kind the JSON writer does not know");
                });
            }
        }

        Result<std::string> write_json(const Value& value, std::size_t expanded_depth) noexcept
        {
            try
            {
                std::string text;
                SA_TRY(write_value(text, value, 0, expanded_depth));

                text += '\n';
                return Result<std::string>(std::move(text));
            }
            catch (const std::bad_alloc&)
            {
                return Status::from_message(StatusCode::OutOfMemory, []()
                {
                    return std::string("the host heap refused the text of a JSON document");
                });
            }
        }

    } // namespace Config
} // namespace SushiAI
