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
 * @brief Tests the zip reader: the entries it lists, their bytes, deflated entries and refusals.
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
#include <SushiAI/io/zip_archive.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;

namespace
{
    /** @brief Returns the bytes whose values are @p values as a string. */
    std::string string_of(const std::vector<unsigned>& values)
    {
        std::string out;
        for (const unsigned value : values)
            out.push_back(static_cast<char>(value));
        return out;
    }

    /** @brief Holds the text of the deflated entries below. */
    const std::string K_HELLO = "hello hello hello hello";

    /**
     * @brief Holds the raw DEFLATE stream of K_HELLO, one fixed-Huffman block, made by
     *        `c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_FIXED);
     *        c.compress(b"hello hello hello hello") + c.flush()`.
     */
    const std::string K_HELLO_DEFLATED =
        string_of({0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x57, 0xC8, 0x40, 0x27, 0x01});

    /** @brief Holds `zlib.crc32(b"hello hello hello hello")`. */
    constexpr std::uint32_t K_HELLO_CRC = 0x8D3D51E3u;

    /**
     * @brief Holds the raw DEFLATE stream of "stored block bytes", one stored block, made by
     *        `c = zlib.compressobj(0, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY);
     *        c.compress(b"stored block bytes") + c.flush()`.
     */
    const std::string K_BLOCK_DEFLATED =
        string_of({0x01, 0x12, 0x00, 0xED, 0xFF, 0x73, 0x74, 0x6F, 0x72, 0x65, 0x64, 0x20,
                   0x62, 0x6C, 0x6F, 0x63, 0x6B, 0x20, 0x62, 0x79, 0x74, 0x65, 0x73});

    /** @brief Holds `zlib.crc32(b"stored block bytes")`. */
    constexpr std::uint32_t K_BLOCK_CRC = 0x7346EFFFu;

    /** @brief Returns a deflated entry named @p name that states the size and the CRC-32 given. */
    SushiAI::Test::ZipFixtureEntry deflated_entry(const std::string& name,
                                                  const std::string& compressed,
                                                  std::size_t inflated_size, std::uint32_t crc)
    {
        return {name, compressed, 8, inflated_size, crc};
    }

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

/** @brief Tests that read refuses an entry of a method other than stored and deflated, by name. */
TEST(Small_ZipArchive, ReadRefusesAnEntryOfAnotherMethodByName)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of({{"a/data.pkl", "zz", 12}});

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.pt"),
                      StatusCode::Unsupported, "'a/data.pkl'", "method 12");
}

/** @brief Tests that read decodes a deflated entry whose CRC-32 matches. */
TEST(Small_ZipArchive, ReadDecodesADeflatedEntryWithItsCrc)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of(
        {deflated_entry("a/data.bin", K_HELLO_DEFLATED, K_HELLO.size(), K_HELLO_CRC)});

    SA_ASSERT_OK_AND_ASSIGN(const IO::ZipArchive archive,
                            IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.zip"));

    ASSERT_EQ(archive.entries().size(), 1u);
    EXPECT_TRUE(archive.entries()[0].deflated);
    EXPECT_EQ(archive.entries()[0].size, K_HELLO.size());
    EXPECT_EQ(text_of(archive, archive.entries()[0]), K_HELLO);
}

/** @brief Tests that each entry of a mixed archive keeps its own bytes. */
TEST(Small_ZipArchive, ContentsOfAMixedArchiveStayWithTheirEntries)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of(
        {{"first", "plain"},
         deflated_entry("second", K_HELLO_DEFLATED, K_HELLO.size(), K_HELLO_CRC),
         {"third", "tail"},
         deflated_entry("fourth", K_BLOCK_DEFLATED, 18, K_BLOCK_CRC)});

    SA_ASSERT_OK_AND_ASSIGN(const IO::ZipArchive archive,
                            IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.zip"));

    ASSERT_EQ(archive.entries().size(), 4u);
    EXPECT_EQ(text_of(archive, archive.entries()[0]), "plain");
    EXPECT_EQ(text_of(archive, archive.entries()[1]), K_HELLO);
    EXPECT_EQ(text_of(archive, archive.entries()[2]), "tail");
    EXPECT_EQ(text_of(archive, archive.entries()[3]), "stored block bytes");
}

/** @brief Tests that read refuses a deflated entry whose CRC-32 differs, naming the entry. */
TEST(Small_ZipArchive, ReadRefusesADeflatedEntryWithAWrongCrcByName)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of(
        {deflated_entry("a/data.bin", K_HELLO_DEFLATED, K_HELLO.size(), K_HELLO_CRC ^ 1u)});

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.zip"),
                      StatusCode::InvalidFile, "'a/data.bin'", "CRC-32");
}

/** @brief Tests that read refuses a deflated entry whose stream is corrupt, naming the entry. */
TEST(Small_ZipArchive, ReadRefusesACorruptDeflateStreamByEntryName)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of(
        {deflated_entry("a/data.bin", string_of({0x07}), 4, 0)});

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.zip"),
                      StatusCode::InvalidFile, "'a/data.bin'", "reserved type 3");
}

/** @brief Tests that read refuses a deflated entry that decodes to other than its stated size. */
TEST(Small_ZipArchive, ReadRefusesADeflatedEntryOfAnotherSizeThanItStates)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of(
        {deflated_entry("a/data.bin", K_HELLO_DEFLATED, K_HELLO.size() + 1, K_HELLO_CRC)});

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.zip"),
                      StatusCode::InvalidFile, "'a/data.bin'", "states 24 bytes", "holds 23");
}

/** @brief Tests that read refuses a size no DEFLATE stream of that length can reach. */
TEST(Small_ZipArchive, ReadRefusesADeflatedSizeBeyondTheMostDeflateCanMake)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of(
        {deflated_entry("a/data.bin", K_HELLO_DEFLATED, 1000000, K_HELLO_CRC)});

    SA_EXPECT_FAILURE(IO::ZipArchive::read(SushiAI::Test::view_of(bytes), "fixture.zip"),
                      StatusCode::InvalidFile, "'a/data.bin'", "more than DEFLATE can make");
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
