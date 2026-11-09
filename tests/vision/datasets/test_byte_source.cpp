/**************************************************************************/
/* test_byte_source.cpp                                                   */
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
 * @file test_byte_source.cpp
 * @brief Tests that MemorySource serves, measures and lists the files added to it.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns a source holding data/a.bin with four bytes and data/sub/b.bin with two. */
    MemorySource two_files()
    {
        MemorySource source;
        source.add_file("data/a.bin", "abcd");
        source.add_file("data/sub/b.bin", "xy");
        return source;
    }
}

/** @brief Tests that a stored file reads back with the bytes it was added with. */
TEST(Small_DatasetsByteSource, AStoredFileReadsBack)
{
    const MemorySource source = two_files();

    SA_ASSERT_OK_AND_ASSIGN(const std::string bytes, source.read("data/a.bin", "test"));
    EXPECT_EQ(bytes, "abcd");

    SA_ASSERT_OK_AND_ASSIGN(const std::string tail, source.read_range("data/a.bin", 1, 2, "test"));
    EXPECT_EQ(tail, "bc");
}

/** @brief Tests that kind_of and size_of answer for files, folders and absent paths. */
TEST(Small_DatasetsByteSource, KindAndSizeAnswerForWhatWasAdded)
{
    MemorySource source = two_files();
    source.add_directory("empty");

    EXPECT_EQ(source.kind_of("data/a.bin"), PathKind::File);
    EXPECT_EQ(source.kind_of("data"), PathKind::Directory);
    EXPECT_EQ(source.kind_of("empty"), PathKind::Directory);
    EXPECT_EQ(source.kind_of("data/absent.bin"), PathKind::Missing);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t size, source.size_of("data/a.bin", "test"));
    EXPECT_EQ(size, 4u);
}

/** @brief Tests that list names the entries of a folder, one level or the whole tree. */
TEST(Small_DatasetsByteSource, ListNamesTheEntriesOfAFolder)
{
    const MemorySource source = two_files();

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::filesystem::path> children,
                            source.list("data", EntryKind::File, ListDepth::Children, "test"));
    EXPECT_EQ(children, (std::vector<std::filesystem::path>{"data/a.bin"}));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::filesystem::path> tree,
                            source.list("data", EntryKind::File, ListDepth::Tree, "test"));
    EXPECT_EQ(tree, (std::vector<std::filesystem::path>{"data/a.bin", "data/sub/b.bin"}));
}

/** @brief Tests that remove drops one file, reports whether it held it, and leaves the others. */
TEST(Small_DatasetsByteSource, RemoveDropsTheFileItNames)
{
    MemorySource source = two_files();

    EXPECT_TRUE(source.remove("data/a.bin"));
    EXPECT_EQ(source.kind_of("data/a.bin"), PathKind::Missing);
    EXPECT_EQ(source.kind_of("data/sub/b.bin"), PathKind::File);
    EXPECT_FALSE(source.remove("data/a.bin"));
}

/** @brief Tests that an absent path is refused with the reader and the path in the message. */
TEST(Small_DatasetsByteSource, AnAbsentPathIsRefused)
{
    const MemorySource source = two_files();

    SA_EXPECT_FAILURE(source.read("data/absent.bin", "probe"), StatusCode::FileNotFound,
                      "absent.bin", "probe");
    SA_EXPECT_FAILURE(source.size_of("data", "probe"), StatusCode::InvalidFile, "probe");
    SA_EXPECT_FAILURE(source.list("nowhere", EntryKind::File, ListDepth::Children, "probe"),
                      StatusCode::FileNotFound, "nowhere", "probe");
}
