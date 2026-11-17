/**************************************************************************/
/* protobuf_text.cpp                                                      */
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
 * @file protobuf_text.cpp
 * @brief Defines the parser of protobuf text-format documents.
 * @author Mustafa Garip
 */

#include <SushiAI/io/protobuf_text.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the deepest nesting of braces the parser accepts. */
            constexpr std::size_t K_MAX_DEPTH = 100;

            /** @brief Returns whether @p c is a decimal digit. */
            bool is_digit(char c) noexcept
            {
                return c >= '0' && c <= '9';
            }

            /** @brief Returns whether @p c is a hexadecimal digit. */
            bool is_hex_digit(char c) noexcept
            {
                return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
            }

            /** @brief Returns the value of the hexadecimal digit @p c. */
            std::uint32_t hex_value(char c) noexcept
            {
                if (is_digit(c))
                    return static_cast<std::uint32_t>(c - '0');
                if (c >= 'a' && c <= 'f')
                    return static_cast<std::uint32_t>(c - 'a' + 10);
                return static_cast<std::uint32_t>(c - 'A' + 10);
            }

            /** @brief Returns whether @p c starts an identifier. */
            bool is_identifier_start(char c) noexcept
            {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
            }

            /** @brief Returns whether @p c continues an identifier. */
            bool is_identifier_part(char c) noexcept
            {
                return is_identifier_start(c) || is_digit(c);
            }

            /** @brief Returns whether @p c is a space, tab, carriage return or newline. */
            bool is_space(char c) noexcept
            {
                return c == ' ' || c == '\t' || c == '\r' || c == '\n';
            }

            /** @brief Returns whether @p word equals @p lowered, ignoring ASCII case. */
            bool equals_ignoring_case(std::string_view word, std::string_view lowered) noexcept
            {
                if (word.size() != lowered.size())
                    return false;
                for (std::size_t i = 0; i < word.size(); ++i)
                {
                    const char c = word[i];
                    const char folded =
                        (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
                    if (folded != lowered[i])
                        return false;
                }
                return true;
            }

            /** @brief Returns whether @p word is inf, infinity or nan in any case. */
            bool is_special_real(std::string_view word) noexcept
            {
                return equals_ignoring_case(word, "inf") ||
                       equals_ignoring_case(word, "infinity") ||
                       equals_ignoring_case(word, "nan");
            }

            /** @brief Returns the length of the run of decimal digits that opens @p text. */
            std::size_t digit_run(std::string_view text) noexcept
            {
                std::size_t n = 0;
                while (n < text.size() && is_digit(text[n]))
                    ++n;
                return n;
            }

            /** @brief Returns whether @p token, without its sign, is a hex, decimal or float. */
            bool is_unsigned_number(std::string_view token) noexcept
            {
                if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X'))
                {
                    for (std::size_t i = 2; i < token.size(); ++i)
                    {
                        if (!is_hex_digit(token[i]))
                            return false;
                    }
                    return true;
                }

                std::size_t at = digit_run(token);
                std::size_t digits = at;
                if (at < token.size() && token[at] == '.')
                {
                    const std::size_t fraction = digit_run(token.substr(at + 1));
                    digits += fraction;
                    at += 1 + fraction;
                }
                if (digits == 0)
                    return false;
                if (at < token.size() && (token[at] == 'e' || token[at] == 'E'))
                {
                    ++at;
                    if (at < token.size() && (token[at] == '+' || token[at] == '-'))
                        ++at;
                    const std::size_t exponent = digit_run(token.substr(at));
                    if (exponent == 0)
                        return false;
                    at += exponent;
                }
                if (at < token.size() && (token[at] == 'f' || token[at] == 'F'))
                    ++at;
                return at == token.size();
            }

            /** @brief Returns whether @p token is a number the text format reads. */
            bool is_number(std::string_view token) noexcept
            {
                if (!token.empty() && (token[0] == '-' || token[0] == '+'))
                    token.remove_prefix(1);
                return is_unsigned_number(token) || is_special_real(token);
            }

            /** @brief Appends the UTF-8 bytes of code point @p code to @p out. */
            void append_utf8(std::string& out, std::uint32_t code)
            {
                if (code < 0x80u)
                {
                    out.push_back(static_cast<char>(code));
                }
                else if (code < 0x800u)
                {
                    out.push_back(static_cast<char>(0xC0u | (code >> 6)));
                    out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                }
                else if (code < 0x10000u)
                {
                    out.push_back(static_cast<char>(0xE0u | (code >> 12)));
                    out.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
                    out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                }
                else
                {
                    out.push_back(static_cast<char>(0xF0u | (code >> 18)));
                    out.push_back(static_cast<char>(0x80u | ((code >> 12) & 0x3Fu)));
                    out.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
                    out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                }
            }

            /** @brief Returns @p c as a quoted character, or as its hex value when unprintable. */
            std::string describe(char c)
            {
                const unsigned char byte = static_cast<unsigned char>(c);
                if (byte >= 0x20u && byte < 0x7Fu)
                    return std::string("'") + c + "'";
                const char* digits = "0123456789ABCDEF";
                std::string out = "byte 0x";
                out.push_back(digits[byte >> 4]);
                out.push_back(digits[byte & 0xFu]);
                return out;
            }

            /** @brief Reads one document, tracking the line and column of its cursor. */
            class Parser
            {
                public:
                    /** @brief Holds the document and its name. */
                    Parser(std::string_view text, std::string_view origin) noexcept
                        : text_(text), origin_(origin)
                    {

                    }

                    /** @brief Parses the whole document into its root message. */
                    Result<TextField> run()
                    {
                        TextField root;
                        SA_TRY(parse_fields(root, 0, 0, 0));
                        return root;
                    }

                private:
                    /** @brief Returns InvalidFile naming the line and column and @p what. */
                    Status fail_at(std::size_t line, std::size_t column,
                                   std::string_view what) const
                    {
                        SA_FAIL_IF(true, StatusCode::InvalidFile, "'{}': line {}, column {}: {}",
                                   origin_, line, column, what);
                        return Status();
                    }

                    /** @brief Returns InvalidFile at the cursor. */
                    Status fail_here(std::string_view what) const
                    {
                        return fail_at(line_, column_, what);
                    }

                    /** @brief Returns whether the cursor has passed the last byte. */
                    bool at_end() const noexcept
                    {
                        return at_ >= text_.size();
                    }

                    /** @brief Returns the byte under the cursor. */
                    char peek() const noexcept
                    {
                        return text_[at_];
                    }

                    /** @brief Moves the cursor one byte on, keeping line and column. */
                    void advance() noexcept
                    {
                        if (text_[at_] == '\n')
                        {
                            ++line_;
                            column_ = 1;
                        }
                        else
                        {
                            ++column_;
                        }
                        ++at_;
                    }

                    /** @brief Moves the cursor past spaces and # comments. */
                    void skip_space() noexcept
                    {
                        while (!at_end())
                        {
                            if (is_space(peek()))
                            {
                                advance();
                            }
                            else if (peek() == '#')
                            {
                                while (!at_end() && peek() != '\n')
                                    advance();
                            }
                            else
                            {
                                return;
                            }
                        }
                    }

                    /** @brief Reads the fields of a message up to its closing brace or the end. */
                    Status parse_fields(TextField& parent, std::size_t depth, std::size_t open_line,
                                        std::size_t open_column)
                    {
                        while (true)
                        {
                            skip_space();
                            if (at_end())
                            {
                                SA_FAIL_IF(depth > 0, StatusCode::InvalidFile,
                                           "'{}': line {}, column {}: the '{{' opened here is "
                                           "never closed",
                                           origin_, open_line, open_column);
                                return Status();
                            }
                            if (peek() == '}')
                            {
                                SA_FAIL_IF(depth == 0, StatusCode::InvalidFile,
                                           "'{}': line {}, column {}: stray '}}'", origin_, line_,
                                           column_);
                                advance();
                                return Status();
                            }

                            TextField field;
                            SA_TRY(parse_field(field, depth));
                            parent.fields.push_back(std::move(field));

                            skip_space();
                            if (!at_end() && (peek() == ';' || peek() == ','))
                                advance();
                        }
                    }

                    /** @brief Reads one field: its name, its optional colon and its value. */
                    Status parse_field(TextField& field, std::size_t depth)
                    {
                        field.line = line_;
                        field.column = column_;
                        SA_FAIL_IF(!is_identifier_start(peek()), StatusCode::InvalidFile,
                                   "'{}': line {}, column {}: expected a field name, found {}",
                                   origin_, line_, column_, describe(peek()));
                        const std::size_t name_start = at_;
                        while (!at_end() && is_identifier_part(peek()))
                            advance();
                        field.name = std::string(text_.substr(name_start, at_ - name_start));

                        skip_space();
                        bool has_colon = false;
                        if (!at_end() && peek() == ':')
                        {
                            has_colon = true;
                            advance();
                            skip_space();
                        }
                        SA_FAIL_IF(at_end(), StatusCode::InvalidFile,
                                   "'{}': line {}, column {}: the document ends after the field "
                                   "'{}'",
                                   origin_, line_, column_, field.name);

                        if (peek() == '{')
                        {
                            SA_FAIL_IF(depth + 1 > K_MAX_DEPTH, StatusCode::InvalidFile,
                                       "'{}': line {}, column {}: messages nest deeper than {}",
                                       origin_, line_, column_, K_MAX_DEPTH);
                            const std::size_t open_line = line_;
                            const std::size_t open_column = column_;
                            advance();
                            field.kind = TextKind::Message;
                            return parse_fields(field, depth + 1, open_line, open_column);
                        }

                        SA_FAIL_IF(!has_colon, StatusCode::InvalidFile,
                                   "'{}': line {}, column {}: expected ':' or '{{' after the "
                                   "field '{}', found {}",
                                   origin_, line_, column_, field.name, describe(peek()));
                        SA_TRY(parse_scalar(field));

                        const bool layout_before = at_ > 0 && is_space(text_[at_ - 1]);
                        SA_FAIL_IF(!at_end() && !layout_before && !is_space(peek()) &&
                                       peek() != '}' && peek() != '#' && peek() != ';' &&
                                       peek() != ',',
                                   StatusCode::InvalidFile,
                                   "'{}': line {}, column {}: unexpected {} after the value of "
                                   "'{}'",
                                   origin_, line_, column_, describe(peek()), field.name);
                        return Status();
                    }

                    /** @brief Reads a number, a string or an identifier into @p field. */
                    Status parse_scalar(TextField& field)
                    {
                        const char c = peek();
                        if (c == '"' || c == '\'')
                        {
                            field.kind = TextKind::String;
                            SA_TRY(parse_string(field.text));
                            while (true)
                            {
                                skip_space();
                                if (at_end() || (peek() != '"' && peek() != '\''))
                                    return Status();
                                SA_TRY(parse_string(field.text));
                            }
                        }
                        if (is_digit(c) || c == '-' || c == '+' || c == '.')
                        {
                            field.kind = TextKind::Number;
                            return parse_number(field.text);
                        }
                        if (is_identifier_start(c))
                        {
                            field.kind = TextKind::Identifier;
                            const std::size_t start = at_;
                            while (!at_end() && is_identifier_part(peek()))
                                advance();
                            field.text = std::string(text_.substr(start, at_ - start));
                            return Status();
                        }
                        return fail_here("expected a value, found " + describe(c));
                    }

                    /** @brief Reads a number, or a signed inf or nan, into @p out. */
                    Status parse_number(std::string& out)
                    {
                        const std::size_t line = line_;
                        const std::size_t column = column_;
                        const std::size_t start = at_;
                        if (peek() == '-' || peek() == '+')
                            advance();
                        const std::string_view prefix = text_.substr(at_, 2);
                        const bool hex = prefix == "0x" || prefix == "0X";
                        while (!at_end())
                        {
                            const char c = peek();
                            const bool exponent_sign =
                                (c == '+' || c == '-') && !hex && at_ > start &&
                                (text_[at_ - 1] == 'e' || text_[at_ - 1] == 'E');
                            if (!is_identifier_part(c) && c != '.' && !exponent_sign)
                                break;
                            advance();
                        }
                        const std::string_view token = text_.substr(start, at_ - start);
                        if (!is_number(token))
                            return fail_at(line, column, "'" + std::string(token) +
                                                             "' is not a number");
                        out = std::string(token);
                        return Status();
                    }

                    /** @brief Reads one quoted string at the cursor and appends it to @p out. */
                    Status parse_string(std::string& out)
                    {
                        const std::size_t line = line_;
                        const std::size_t column = column_;
                        const char quote = peek();
                        advance();
                        while (true)
                        {
                            SA_FAIL_IF(at_end() || peek() == '\n', StatusCode::InvalidFile,
                                       "'{}': line {}, column {}: the string opened here is "
                                       "never closed",
                                       origin_, line, column);
                            const char c = peek();
                            advance();
                            if (c == quote)
                                return Status();
                            if (c != '\\')
                            {
                                out.push_back(c);
                                continue;
                            }
                            SA_TRY(parse_escape(out));
                        }
                    }

                    /** @brief Reads the escape after a backslash and appends what it stands for. */
                    Status parse_escape(std::string& out)
                    {
                        SA_FAIL_IF(at_end(), StatusCode::InvalidFile,
                                   "'{}': line {}, column {}: the document ends inside an escape",
                                   origin_, line_, column_);
                        const std::size_t line = line_;
                        const std::size_t column = column_;
                        const char c = peek();
                        advance();
                        switch (c)
                        {
                            case 'n':  out.push_back('\n'); return Status();
                            case 'r':  out.push_back('\r'); return Status();
                            case 't':  out.push_back('\t'); return Status();
                            case 'a':  out.push_back('\a'); return Status();
                            case 'b':  out.push_back('\b'); return Status();
                            case 'f':  out.push_back('\f'); return Status();
                            case 'v':  out.push_back('\v'); return Status();
                            case '\\': out.push_back('\\'); return Status();
                            case '\'': out.push_back('\''); return Status();
                            case '"':  out.push_back('"'); return Status();
                            case '?':  out.push_back('?'); return Status();
                            case 'x':
                            case 'X':
                                return parse_code(out, 2, line, column, false);
                            case 'u':
                                return parse_code(out, 4, line, column, true);
                            case 'U':
                                return parse_code(out, 8, line, column, true);
                            default:
                                break;
                        }
                        if (c >= '0' && c <= '7')
                        {
                            std::uint32_t value = static_cast<std::uint32_t>(c - '0');
                            for (std::size_t i = 0;
                                 i < 2 && !at_end() && peek() >= '0' && peek() <= '7'; ++i)
                            {
                                value = value * 8u + static_cast<std::uint32_t>(peek() - '0');
                                advance();
                            }
                            SA_FAIL_IF(value > 0xFFu, StatusCode::InvalidFile,
                                       "'{}': line {}, column {}: the octal escape is above \\377",
                                       origin_, line, column);
                            out.push_back(static_cast<char>(value));
                            return Status();
                        }
                        return fail_at(line, column, "unknown escape \\" + std::string(1, c));
                    }

                    /**
                     * @brief Reads the digits of a \x, \u or \U escape and appends their value.
                     * @param max_digits The digits the escape takes; \x takes one to that many.
                     * @param exact Whether the escape needs exactly @p max_digits digits, and
                     *              writes its value as a UTF-8 code point.
                     */
                    Status parse_code(std::string& out, std::size_t max_digits, std::size_t line,
                                      std::size_t column, bool exact)
                    {
                        std::uint32_t value = 0;
                        std::size_t count = 0;
                        while (count < max_digits && !at_end() && is_hex_digit(peek()))
                        {
                            value = value * 16u + hex_value(peek());
                            advance();
                            ++count;
                        }
                        SA_FAIL_IF(count == 0 || (exact && count != max_digits),
                                   StatusCode::InvalidFile,
                                   "'{}': line {}, column {}: the escape needs {} hexadecimal "
                                   "digits",
                                   origin_, line, column, exact ? max_digits : 1);
                        if (!exact)
                        {
                            out.push_back(static_cast<char>(value));
                            return Status();
                        }
                        SA_FAIL_IF(value > 0x10FFFFu || (value >= 0xD800u && value <= 0xDFFFu),
                                   StatusCode::InvalidFile,
                                   "'{}': line {}, column {}: the escape is not a Unicode scalar "
                                   "value",
                                   origin_, line, column);
                        append_utf8(out, value);
                        return Status();
                    }

                    /** @brief Holds the document. */
                    std::string_view text_;

                    /** @brief Names the document in a message. */
                    std::string_view origin_;

                    /** @brief Holds the index of the next byte. */
                    std::size_t at_ = 0;

                    /** @brief Holds the line of the next byte, from 1. */
                    std::size_t line_ = 1;

                    /** @brief Holds the column of the next byte, from 1. */
                    std::size_t column_ = 1;
            };
        } // namespace

        const TextField* find_field(const TextField& message, std::string_view name) noexcept
        {
            for (const TextField& field : message.fields)
            {
                if (field.name == name)
                    return &field;
            }
            return nullptr;
        }

        std::vector<const TextField*> find_fields(const TextField& message,
                                                  std::string_view name)
        {
            std::vector<const TextField*> out;
            for (const TextField& field : message.fields)
            {
                if (field.name == name)
                    out.push_back(&field);
            }
            return out;
        }

        Result<TextField> parse_protobuf_text(std::string_view text,
                                              std::string_view origin) noexcept
        {
            Parser parser(text, origin);
            return parser.run();
        }
    } // namespace IO
} // namespace SushiAI
