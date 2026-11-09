/**************************************************************************/
/* test_byte_sink.cpp                                                     */
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
 * @file test_byte_sink.cpp
 * @brief Tests that MemorySink puts the files written to it where a MemorySource reads them.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_sink.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>

using namespace SushiAI::Vision::Datasets;

/** @brief Tests that a written file reads back from the source under the sink. */
TEST(Small_DatasetsByteSink, AWrittenFileReadsBackFromTheSource)
{
    MemorySource source;
    const MemorySink sink(source);

    SA_EXPECT_OK(sink.write("cache/a.saim", "abcd", "test"));

    SA_ASSERT_OK_AND_ASSIGN(const std::string bytes, source.read("cache/a.saim", "test"));
    EXPECT_EQ(bytes, "abcd");
    EXPECT_EQ(source.kind_of("cache"), PathKind::Directory);
}

/** @brief Tests that a second write to a path replaces the bytes of the first. */
TEST(Small_DatasetsByteSink, ASecondWriteReplacesTheFirst)
{
    MemorySource source;
    const MemorySink sink(source);

    SA_EXPECT_OK(sink.write("cache/a.saim", "abcd", "test"));
    SA_EXPECT_OK(sink.write("cache/a.saim", "xy", "test"));

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t size, source.size_of("cache/a.saim", "test"));
    EXPECT_EQ(size, 2u);
}
