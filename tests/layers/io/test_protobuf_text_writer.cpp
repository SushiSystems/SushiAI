/**************************************************************************/
/* test_protobuf_text_writer.cpp                                          */
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
 * @file test_protobuf_text_writer.cpp
 * @brief Tests the protobuf text writer against the parser: every kind, an escaped string and
 *        a nested message.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include <SushiAI/io/protobuf_text_writer.hpp>

using namespace SushiAI;
using SushiAI::Test::require_value;

namespace
{
    /** @brief Returns a field named @p name of @p kind holding @p text. */
    IO::TextField make_field(const std::string& name, IO::TextKind kind, const std::string& text)
    {
        IO::TextField field;
        field.name = name;
        field.kind = kind;
        field.text = text;
        return field;
    }

    /** @brief Expects @p read to state the names, kinds, texts and fields of @p written. */
    void expect_same_tree(const IO::TextField& read, const IO::TextField& written)
    {
        EXPECT_EQ(read.name, written.name);
        EXPECT_EQ(read.kind, written.kind);
        EXPECT_EQ(read.text, written.text);
        ASSERT_EQ(read.fields.size(), written.fields.size()) << "in '" << written.name << "'";
        for (std::size_t i = 0; i < read.fields.size(); ++i)
            expect_same_tree(read.fields[i], written.fields[i]);
    }
} // namespace

/** @brief Tests that the parser reads back every kind, an escaped string and a nested message. */
TEST(Small_ProtobufTextWriter, TheParserReadsBackTheTree)
{
    IO::TextField root;
    root.fields.push_back(make_field("name", IO::TextKind::String,
                                     std::string("a \"b\"\\c\n\t\x01\xC3\xA9", 12)));
    root.fields.push_back(make_field("count", IO::TextKind::Number, "-3.5e2"));
    root.fields.push_back(make_field("pool", IO::TextKind::Identifier, "MAX"));
    IO::TextField layer = make_field("layer", IO::TextKind::Message, "");
    layer.fields.push_back(make_field("top", IO::TextKind::String, "conv1"));
    IO::TextField param = make_field("param", IO::TextKind::Message, "");
    param.fields.push_back(make_field("stride", IO::TextKind::Number, "2"));
    layer.fields.push_back(param);
    root.fields.push_back(layer);
    root.fields.push_back(layer);

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, IO::write_protobuf_text(root));
    SA_ASSERT_OK_AND_ASSIGN(const IO::TextField read, IO::parse_protobuf_text(text, "written"));

    expect_same_tree(read, root);
}

/** @brief Tests that a field with no name and a bare token with a space are refused. */
TEST(Small_ProtobufTextWriter, AFieldItCannotWriteIsRefused)
{
    IO::TextField unnamed;
    unnamed.fields.push_back(make_field("", IO::TextKind::Number, "1"));
    SA_EXPECT_FAILURE(IO::write_protobuf_text(unnamed), StatusCode::InvalidDescription,
                      "no name");

    IO::TextField spaced;
    spaced.fields.push_back(make_field("pool", IO::TextKind::Identifier, "MA X"));
    SA_EXPECT_FAILURE(IO::write_protobuf_text(spaced), StatusCode::InvalidDescription,
                      "'pool'", "no bare token");
}
