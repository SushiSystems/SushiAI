/**************************************************************************/
/* test_pickle_writer.cpp                                                 */
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
 * @file test_pickle_writer.cpp
 * @brief Tests the pickle writer against the pickle reader: every kind it takes, a call with
 *        what is built onto it, a persistent id, and the refused set.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/io/pickle_writer.hpp>

using namespace SushiAI;
using IO::PickleKind;
using IO::PickleRef;

namespace
{
    /** @brief Returns a node of @p kind. */
    PickleRef make(PickleKind kind)
    {
        PickleRef node = std::make_shared<IO::PickleValue>();
        node->kind = kind;
        return node;
    }

    /** @brief Returns an Int node of @p value. */
    PickleRef make_int(std::int64_t value)
    {
        PickleRef node = make(PickleKind::Int);
        node->integer = value;
        return node;
    }

    /** @brief Returns a node of @p kind holding @p text. */
    PickleRef make_text(PickleKind kind, std::string text)
    {
        PickleRef node = make(kind);
        node->text = std::move(text);
        return node;
    }

    /** @brief Expects @p read to hold what @p written holds, node by node. */
    void expect_same(const IO::PickleValue& read, const IO::PickleValue& written)
    {
        ASSERT_EQ(read.kind, written.kind);
        EXPECT_EQ(read.boolean, written.boolean);
        EXPECT_EQ(read.integer, written.integer);
        EXPECT_EQ(read.real, written.real);
        EXPECT_EQ(read.text, written.text);
        ASSERT_EQ(read.items.size(), written.items.size());
        for (std::size_t i = 0; i < read.items.size(); ++i)
            expect_same(*read.items[i], *written.items[i]);
        ASSERT_EQ(read.entries.size(), written.entries.size());
        for (std::size_t i = 0; i < read.entries.size(); ++i)
        {
            expect_same(*read.entries[i].first, *written.entries[i].first);
            expect_same(*read.entries[i].second, *written.entries[i].second);
        }
        ASSERT_EQ(read.appended.size(), written.appended.size());
        for (std::size_t i = 0; i < read.appended.size(); ++i)
            expect_same(*read.appended[i], *written.appended[i]);
        ASSERT_EQ(read.callable == nullptr, written.callable == nullptr);
        if (read.callable != nullptr)
            expect_same(*read.callable, *written.callable);
        ASSERT_EQ(read.state == nullptr, written.state == nullptr);
        if (read.state != nullptr)
            expect_same(*read.state, *written.state);
    }
} // namespace

/** @brief Tests that the reader gives back a tree of every kind the writer takes. */
TEST(Small_PickleWriter, TheReaderReadsBackEveryKind)
{
    PickleRef list = make(PickleKind::List);
    list->items = {make(PickleKind::None), make_int(7), make_int(-70000), make_int(1LL << 40),
                   make_int(300)};
    PickleRef truth = make(PickleKind::Bool);
    truth->boolean = true;
    PickleRef real = make(PickleKind::Float);
    real->real = -0.15625;
    list->items.push_back(truth);
    list->items.push_back(make(PickleKind::Bool));
    list->items.push_back(real);
    list->items.push_back(make_text(PickleKind::Bytes, std::string(300, 'b')));
    list->items.push_back(make_text(PickleKind::Bytes, "short"));

    PickleRef persistent = make(PickleKind::PersistentID);
    PickleRef id = make(PickleKind::Tuple);
    id->items = {make_text(PickleKind::String, "storage"),
                 make_text(PickleKind::Global, "torch FloatStorage"),
                 make_text(PickleKind::String, "0")};
    persistent->items = {id};

    PickleRef call = make(PickleKind::Call);
    call->callable = make_text(PickleKind::Global, "collections OrderedDict");
    call->entries = {{make_text(PickleKind::String, "w"), persistent}};
    call->appended = {make_int(1)};
    call->state = make(PickleKind::Dict);
    call->state->entries = {{make_text(PickleKind::String, "k"), make(PickleKind::Tuple)}};

    PickleRef root = make(PickleKind::Dict);
    root->entries = {{make_text(PickleKind::String, "list"), list},
                     {make_text(PickleKind::String, "call"), call}};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::byte> stream, IO::write_pickle(*root));
    const std::string_view accepted[] = {"collections OrderedDict", "torch FloatStorage"};
    SA_ASSERT_OK_AND_ASSIGN(
        const PickleRef read,
        IO::read_pickle(SushiRuntime::span<const std::byte>(stream.data(), stream.size()),
                        SushiRuntime::span<const std::string_view>(accepted, 2), "written"));

    expect_same(*read, *root);
}

/** @brief Tests that a set is refused by its path, since protocol 2 cannot state one. */
TEST(Small_PickleWriter, ASetIsRefused)
{
    PickleRef root = make(PickleKind::List);
    root->items = {make(PickleKind::Set)};

    SA_EXPECT_FAILURE(IO::write_pickle(*root), StatusCode::Unsupported, "root[0]", "set");
}
