/**************************************************************************/
/* test_pickle_reader.cpp                                                 */
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
 * @file test_pickle_reader.cpp
 * @brief Tests the pickle reader on streams written by hand from the opcodes of the pickle format.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;
using namespace std::string_view_literals;

namespace
{
    /** @brief Lists the globals the tests accept: one whole name and one module prefix. */
    constexpr std::string_view K_ACCEPTED[] = {"torch Size", "ultralytics."};

    /** @brief Returns read_pickle of @p stream under K_ACCEPTED. */
    Result<IO::PickleRef> read_stream(std::string_view stream)
    {
        const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(stream);
        return IO::read_pickle(SushiAI::Test::view_of(bytes), K_ACCEPTED, "fixture.pkl");
    }

    /** @brief Names one stream that pushes one whole number and the number it pushes. */
    struct IntegerRow
    {
        std::string_view name;   /**< The row's name. */
        std::string_view stream; /**< The opcodes, ending in STOP. */
        std::int64_t expected;   /**< The number, decoded from the pickle format's rules. */
    };

    /** @brief Lists one row per integer opcode, with a sign the opcode can carry. */
    const IntegerRow K_INTEGER_ROWS[] = {
        {"BinInt1", "K\x05."sv, 5},
        {"BinInt2", "M\x00\x01."sv, 256},
        {"NegativeBinInt", "J\xfe\xff\xff\xff."sv, -2},
        {"NegativeLong1", "\x8a\x01\xff."sv, -1},
        {"PositiveLong1", "\x8a\x02\x00\x01."sv, 256},
        {"EmptyLong1", "\x8a\x00."sv, 0},
    };

    /** @brief Runs one row of K_INTEGER_ROWS. */
    class Small_PickleReaderInteger : public ::testing::TestWithParam<IntegerRow>
    {

    };

    /** @brief Returns the name of the row @p info runs. */
    std::string integer_name(const ::testing::TestParamInfo<IntegerRow>& info)
    {
        return std::string(info.param.name);
    }
} // namespace

/** @brief Tests that each integer opcode pushes the number the pickle format defines for it. */
TEST_P(Small_PickleReaderInteger, PushesTheNumberItEncodes)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::PickleRef value, read_stream(GetParam().stream));

    EXPECT_EQ(value->kind, IO::PickleKind::Int);
    EXPECT_EQ(value->integer, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(Opcodes, Small_PickleReaderInteger, ::testing::ValuesIn(K_INTEGER_ROWS),
                         integer_name);

/** @brief Tests that a protocol 2 dict of a whole number, a float and a text reads back whole. */
TEST(Small_PickleReader, ReadsADictOfANumberAFloatAndAText)
{
    // {'a': 7, 'b': 1.5, 'c': 'hi'}; 1.5 is 0x3FF8000000000000 big-endian.
    const std::string_view stream = "\x80\x02}q\x00("
                                    "X\x01\x00\x00\x00" "a" "K\x07"
                                    "X\x01\x00\x00\x00" "b" "G\x3f\xf8\x00\x00\x00\x00\x00\x00"
                                    "X\x01\x00\x00\x00" "c" "X\x02\x00\x00\x00" "hi"
                                    "u."sv;

    SA_ASSERT_OK_AND_ASSIGN(const IO::PickleRef value, read_stream(stream));

    ASSERT_EQ(value->kind, IO::PickleKind::Dict);
    ASSERT_NE(IO::find_entry(*value, "a"), nullptr);
    EXPECT_EQ(IO::find_entry(*value, "a")->integer, 7);
    ASSERT_NE(IO::find_entry(*value, "b"), nullptr);
    EXPECT_EQ(IO::find_entry(*value, "b")->real, 1.5);
    ASSERT_NE(IO::find_entry(*value, "c"), nullptr);
    EXPECT_EQ(IO::find_entry(*value, "c")->text, "hi");
}

/** @brief Tests that a memo fetch hands out the node it stored, not a copy of it. */
TEST(Small_PickleReader, AMemoFetchHandsOutTheStoredNode)
{
    // [d, d] for one dict d: MARK, EMPTY_DICT, BINPUT 0, BINGET 0, LIST, STOP.
    SA_ASSERT_OK_AND_ASSIGN(const IO::PickleRef value, read_stream("(}q\x00h\x00l."sv));

    ASSERT_EQ(value->kind, IO::PickleKind::List);
    ASSERT_EQ(value->items.size(), 2u);
    EXPECT_EQ(value->items[0].get(), value->items[1].get());
}

/** @brief Tests that REDUCE builds a call of the global with the elements of its tuple. */
TEST(Small_PickleReader, ReduceBuildsACallWithTheArgumentsOfItsTuple)
{
    // torch.Size((2, 3)): GLOBAL, BININT1 2, BININT1 3, TUPLE2, TUPLE1, REDUCE.
    SA_ASSERT_OK_AND_ASSIGN(const IO::PickleRef value,
                            read_stream("ctorch\nSize\nK\x02K\x03\x86\x85R."sv));

    EXPECT_TRUE(IO::is_call_of(value.get(), "torch Size"));
    ASSERT_EQ(value->items.size(), 1u);
    ASSERT_EQ(value->items[0]->items.size(), 2u);
    EXPECT_EQ(value->items[0]->items[1]->integer, 3);
}

/** @brief Tests that BUILD gives a NEWOBJ call of an accepted module prefix its state. */
TEST(Small_PickleReader, BuildGivesANewObjectOfAnAcceptedModuleItsState)
{
    // DetectionModel.__new__(), then BUILD with {'a': 1}.
    const std::string_view stream = "\x80\x02" "cultralytics.nn.tasks\nDetectionModel\n)\x81"
                                    "}X\x01\x00\x00\x00" "a" "K\x01" "sb."sv;

    SA_ASSERT_OK_AND_ASSIGN(const IO::PickleRef value, read_stream(stream));

    EXPECT_TRUE(IO::is_call_of(value.get(), "ultralytics.nn.tasks DetectionModel"));
    ASSERT_NE(value->state, nullptr);
    ASSERT_NE(IO::find_entry(*value->state, "a"), nullptr);
    EXPECT_EQ(IO::find_entry(*value->state, "a")->integer, 1);
}

/** @brief Tests that a global no rule accepts is refused by name and never resolved. */
TEST(Small_PickleReader, RefusesAGlobalNoRuleAccepts)
{
    SA_EXPECT_FAILURE(read_stream("\x80\x02" "cos\nsystem\n."sv), StatusCode::Unsupported,
                      "'os system'", "does not accept");
}

/** @brief Tests that a module prefix does not accept a module that only begins with its text. */
TEST(Small_PickleReader, APrefixRefusesAModuleThatOnlySharesItsText)
{
    SA_EXPECT_FAILURE(read_stream("cultralyticsx\nModel\n."sv), StatusCode::Unsupported,
                      "'ultralyticsx Model'");
}

/** @brief Tests that a stream that ends before STOP is refused as a malformed file. */
TEST(Small_PickleReader, RefusesAStreamThatEndsBeforeStop)
{
    SA_EXPECT_FAILURE(read_stream("\x80\x02K\x01"sv), StatusCode::InvalidFile,
                      "before its STOP");
}

/** @brief Tests that an opcode the reader does not run is refused, naming its number. */
TEST(Small_PickleReader, RefusesAnOpcodeItDoesNotRun)
{
    // 'P' is PERSID, the text form of a persistent id, which protocol 2 writers never emit.
    SA_EXPECT_FAILURE(read_stream("P0\n."sv), StatusCode::Unsupported, "opcode 80");
}
