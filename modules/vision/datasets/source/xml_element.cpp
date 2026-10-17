/**************************************************************************/
/* xml_element.cpp                                                        */
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
 * @file xml_element.cpp
 * @brief Implements the parser of the XML subset: elements, text and five entities.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/xml_element.hpp>


#include <SushiAI/failure/status.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_PARSER = "parse_xml";

        /** @brief Lists the bytes that separate tokens and surround text. */
        constexpr std::string_view K_WHITE_SPACE = " \t\r\n";

        /** @brief Holds the three bytes a UTF-8 file may open with. */
        constexpr std::string_view K_BYTE_ORDER_MARK = "\xEF\xBB\xBF";

        /** @brief Pairs the text after an ampersand with the character it stands for. */
        struct Entity
        {
            std::string_view text; /**< What follows the ampersand, semicolon included. */
            char character;        /**< What the entity becomes. */
        };

        /** @brief Holds the entities the parser replaces. */
        constexpr Entity K_ENTITIES[] = {
            Entity{"amp;", '&'}, Entity{"lt;", '<'}, Entity{"gt;", '>'},
            Entity{"quot;", '"'}, Entity{"apos;", '\''},
        };

        /** @brief Reports whether a byte is an ASCII letter. */
        [[nodiscard]] constexpr bool is_letter(char c) noexcept
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }

        /** @brief Reports whether a byte may open a name. */
        [[nodiscard]] constexpr bool is_name_start(char c) noexcept
        {
            return is_letter(c) || c == '_' || c == ':';
        }

        /** @brief Reports whether a byte may follow the first byte of a name. */
        [[nodiscard]] constexpr bool is_name_part(char c) noexcept
        {
            return is_name_start(c) || (c >= '0' && c <= '9') || c == '-' || c == '.';
        }

        /** @brief Reads one document from its first byte to its last, counting lines. */
        class XMLParser final
        {
            public:
                /** @brief Starts at the first byte of @p text, on line 1. */
                explicit XMLParser(std::string_view text) noexcept : text_(text)
                {

                }

                /** @brief Parses the declaration, the root element and what may follow it. */
                [[nodiscard]] Result<XMLElement> parse_document()
                {
                    if (starts_with(K_BYTE_ORDER_MARK))
                        advance(K_BYTE_ORDER_MARK.size());

                    if (starts_with("<?xml"))
                        SA_TRY(skip_past("?>", "the XML declaration"));

                    SA_TRY(skip_between_elements());
                    SA_FAIL_IF(at_end(), StatusCode::InvalidFile,
                               "line {}: {}: the document holds no element", line_, K_PARSER);
                    SA_TRY(refuse_markup());

                    XMLElement root;
                    SA_TRY(parse_element(root, 1));
                    SA_TRY(skip_between_elements());
                    SA_FAIL_IF(!at_end(), StatusCode::InvalidFile,
                               "line {}: {}: content follows the root element '{}'", line_,
                               K_PARSER, root.name);

                    return Result<XMLElement>(std::move(root));
                }

            private:
                /** @brief Reports whether every byte was read. */
                [[nodiscard]] bool at_end() const noexcept { return position_ >= text_.size(); }

                /** @brief Reports whether the unread text opens with @p prefix. */
                [[nodiscard]] bool starts_with(std::string_view prefix) const noexcept
                {
                    return text_.substr(position_, prefix.size()) == prefix;
                }

                /** @brief Reads @p count bytes, or what is left, and counts their newlines. */
                void advance(std::size_t count) noexcept
                {
                    for (; count > 0 && !at_end(); --count, ++position_)
                    {
                        if (text_[position_] == '\n')
                            ++line_;
                    }
                }

                /** @brief Reads white space, if any. */
                void skip_white_space() noexcept
                {
                    while (!at_end() && K_WHITE_SPACE.find(text_[position_]) != K_WHITE_SPACE.npos)
                        advance(1);
                }

                /** @brief Reads to the end of @p terminator; refuses a @p what left open. */
                [[nodiscard]] Status skip_past(std::string_view terminator, std::string_view what)
                {
                    const std::size_t opened = line_;
                    const std::size_t found = text_.find(terminator, position_);

                    SA_FAIL_IF(found == std::string_view::npos, StatusCode::InvalidFile,
                               "line {}: {}: {} is not closed with '{}'", opened, K_PARSER, what,
                               terminator);

                    advance(found + terminator.size() - position_);
                    return Status();
                }

                /** @brief Reads the white space and the comments between two elements. */
                [[nodiscard]] Status skip_between_elements()
                {
                    for (skip_white_space(); starts_with("<!--"); skip_white_space())
                        SA_TRY(skip_past("-->", "a comment"));

                    return Status();
                }

                /** @brief Refuses the markup the subset lacks and text where a tag must be. */
                [[nodiscard]] Status refuse_markup() const
                {
                    SA_FAIL_IF(starts_with("<![CDATA["), StatusCode::InvalidFile,
                               "line {}: {}: a CDATA section is not read", line_, K_PARSER);
                    SA_FAIL_IF(starts_with("<!"), StatusCode::InvalidFile,
                               "line {}: {}: a declaration ('<!...>') is not read", line_,
                               K_PARSER);
                    SA_FAIL_IF(starts_with("<?"), StatusCode::InvalidFile,
                               "line {}: {}: a processing instruction ('<?...?>') is not read",
                               line_, K_PARSER);
                    SA_FAIL_IF(!starts_with("<"), StatusCode::InvalidFile,
                               "line {}: {}: text stands where an element must open", line_,
                               K_PARSER);
                    return Status();
                }

                /** @brief Reads a name into @p name; refuses a byte no name opens with. */
                [[nodiscard]] Status parse_name(std::string& name)
                {
                    SA_FAIL_IF(at_end() || !is_name_start(text_[position_]),
                               StatusCode::InvalidFile,
                               "line {}: {}: a name must open with a letter, '_' or ':'", line_,
                               K_PARSER);

                    const std::size_t begin = position_;
                    while (!at_end() && is_name_part(text_[position_]))
                        ++position_;

                    name.assign(text_.substr(begin, position_ - begin));
                    return Status();
                }

                /** @brief Reads the attributes of an opening tag and keeps none of them. */
                [[nodiscard]] Status skip_attributes()
                {
                    std::string name;

                    for (skip_white_space(); !at_end() && !starts_with(">") && !starts_with("/>");
                         skip_white_space())
                    {
                        SA_TRY(parse_name(name));
                        skip_white_space();
                        SA_FAIL_IF(!starts_with("="), StatusCode::InvalidFile,
                                   "line {}: {}: the attribute '{}' has no '='", line_, K_PARSER,
                                   name);
                        advance(1);
                        skip_white_space();
                        SA_FAIL_IF(!starts_with("\"") && !starts_with("'"),
                                   StatusCode::InvalidFile,
                                   "line {}: {}: the value of the attribute '{}' is not quoted",
                                   line_, K_PARSER, name);

                        const std::string_view quote = text_.substr(position_, 1);
                        advance(1);
                        SA_TRY(skip_past(quote, "an attribute value"));
                    }

                    return Status();
                }

                /** @brief Reads one entity after its ampersand and appends its character. */
                [[nodiscard]] Status append_entity(std::string& text)
                {
                    advance(1);

                    for (const Entity& entity : K_ENTITIES)
                    {
                        if (!starts_with(entity.text))
                            continue;

                        text.push_back(entity.character);
                        advance(entity.text.size());
                        return Status();
                    }

                    SA_FAIL_IF(true, StatusCode::InvalidFile,
                               "line {}: {}: '&' opens none of &amp; &lt; &gt; &quot; &apos;",
                               line_, K_PARSER);
                    return Status();
                }

                /** @brief Reads the closing tag of @p element; refuses another element's. */
                [[nodiscard]] Status parse_closing_tag(const XMLElement& element)
                {
                    std::string name;

                    advance(2);
                    SA_TRY(parse_name(name));
                    SA_FAIL_IF(name != element.name, StatusCode::InvalidFile,
                               "line {}: {}: '</{}>' closes '<{}>', open since line {}", line_,
                               K_PARSER, name, element.name, element.line);

                    skip_white_space();
                    SA_FAIL_IF(!starts_with(">"), StatusCode::InvalidFile,
                               "line {}: {}: the closing tag of '{}' has no '>'", line_, K_PARSER,
                               name);
                    advance(1);
                    return Status();
                }

                /**
                 * @brief Reads one element from its '<' to the end of its closing tag.
                 * @param depth Counts the elements open with this one, the root being 1.
                 */
                [[nodiscard]] Status parse_element(XMLElement& element, std::int32_t depth)
                {
                    SA_FAIL_IF(depth > K_XML_MAX_DEPTH, StatusCode::InvalidFile,
                               "line {}: {}: more than {} elements are open at once", line_,
                               K_PARSER, K_XML_MAX_DEPTH);

                    element.line = line_;
                    advance(1);
                    SA_TRY(parse_name(element.name));
                    SA_TRY(skip_attributes());

                    if (starts_with("/>"))
                    {
                        advance(2);
                        return Status();
                    }

                    SA_FAIL_IF(!starts_with(">"), StatusCode::InvalidFile,
                               "line {}: {}: the tag '<{}', opened on line {}, has no '>'", line_,
                               K_PARSER, element.name, element.line);
                    advance(1);

                    SA_TRY(parse_content(element, depth));

                    const std::size_t first = element.text.find_first_not_of(K_WHITE_SPACE);
                    if (first == std::string::npos)
                        element.text.clear();
                    else
                        element.text = element.text.substr(
                            first, element.text.find_last_not_of(K_WHITE_SPACE) - first + 1);

                    return Status();
                }

                /** @brief Reads the text, the comments and the children up to the closing tag. */
                [[nodiscard]] Status parse_content(XMLElement& element, std::int32_t depth)
                {
                    for (;;)
                    {
                        SA_FAIL_IF(at_end(), StatusCode::InvalidFile,
                                   "line {}: {}: the element '{}', opened on line {}, is not "
                                   "closed",
                                   line_, K_PARSER, element.name, element.line);

                        if (starts_with("<!--"))
                        {
                            SA_TRY(skip_past("-->", "a comment"));
                        }
                        else if (starts_with("</"))
                        {
                            return parse_closing_tag(element);
                        }
                        else if (starts_with("<"))
                        {
                            SA_TRY(refuse_markup());
                            element.children.emplace_back();
                            SA_TRY(parse_element(element.children.back(), depth + 1));
                        }
                        else if (starts_with("&"))
                        {
                            SA_TRY(append_entity(element.text));
                        }
                        else
                        {
                            element.text.push_back(text_[position_]);
                            advance(1);
                        }
                    }
                }

                /// @brief Holds the whole document; not owned.
                std::string_view text_;

                /// @brief Counts the bytes read.
                std::size_t position_ = 0;

                /// @brief Holds the one-based line of the next unread byte.
                std::size_t line_ = 1;
        };
    } // namespace

    const XMLElement* XMLElement::find_child(std::string_view child_name) const noexcept
    {
        for (const XMLElement& child : children)
        {
            if (child.name == child_name)
                return &child;
        }

        return nullptr;
    }

    Result<XMLElement> parse_xml(std::string_view text)
    {
        return Detail::at_boundary(K_PARSER, [&]()
        {
            return XMLParser(text).parse_document();
        });
    }
} // namespace SushiAI::Vision::Datasets
