/**************************************************************************/
/* test_zip_archive.cpp                                                   */
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
 * @file test_zip_archive.cpp
 * @brief Tests the zip reader: the entries it lists, their bytes and the archives it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/zip_archive.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;

namespace
{
    /** @brief Returns the bytes of @p entry of @p archive as text. */
    std::string text_of(const IO::ZipArchive& archive, const IO::ZipEntry& entry)
    {
        const SushiRuntime::span<const std::byte> bytes = archive.contents(entry);
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
} // namespace

/** @brief Tests that read lists the stored entries in directory order with their bytes. */
TEST(Small_ZipArchive, ReadListsTheStoredEntriesWithTheirBytes)
{
    const std::vector<std::byte> bytes =
        SushiAI::Test::zip_of({{"a/data.pkl", "pickle"}, {"a/data/0", ""}, {"a/version", "3\n"}});

    SA_ASSERT_OK_AND_ASSIGN(const IO::ZipArchive archive,
                            IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.pt"));

    ASSERT_EQ(archive.entries().size(), 3u);
    EXPECT_EQ(archive.entries()[0].name, "a/data.pkl");
    EXPECT_EQ(archive.entries()[1].name, "a/data/0");
    EXPECT_EQ(text_of(archive, archive.entries()[0]), "pickle");
    EXPECT_EQ(text_of(archive, archive.entries()[1]), "");
    EXPECT_EQ(text_of(archive, archive.entries()[2]), "3\n");
}

/** @brief Tests that find returns the entry of a stored name and null for any other. */
TEST(Small_ZipArchive, FindReturnsTheEntryOfAStoredNameOnly)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of({{"x/byteorder", "little"}});
    SA_ASSERT_OK_AND_ASSIGN(const IO::ZipArchive archive,
                            IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.pt"));

    const IO::ZipEntry* const found = archive.find("x/byteorder");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(text_of(archive, *found), "little");
    EXPECT_EQ(archive.find("x/version"), nullptr);
}

/** @brief Tests that read refuses a compressed entry and names it. */
TEST(Small_ZipArchive, ReadRefusesACompressedEntryByName)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of({{"a/data.pkl", "zz", 8}});

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.pt"),
                      StatusCode::Unsupported, "'a/data.pkl'", "method 8");
}

/** @brief Tests that read refuses bytes that hold no end of central directory record. */
TEST(Small_ZipArchive, ReadRefusesBytesWithoutAnEndRecord)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(std::string(40, 'x'));

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.pt"),
                      StatusCode::InvalidFile, "'fixture.pt'", "not a zip archive");
}

/** @brief Tests that read refuses an archive cut inside an entry's bytes. */
TEST(Small_ZipArchive, ReadRefusesAnEntryThatRunsPastTheFile)
{
    std::vector<std::byte> bytes = SushiAI::Test::zip_of({{"a/data/0", std::string(64, 'w')}});
    // The central header follows the 30-byte local header, the 8-byte name and 64 bytes.
    const std::size_t central = 30 + 8 + 64;
    bytes[central + 21] = std::byte{0x7F};
    bytes[central + 25] = std::byte{0x7F};

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.pt"),
                      StatusCode::InvalidFile, "'a/data/0'");
}
