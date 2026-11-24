/**************************************************************************/
/* test_zip_writer.cpp                                                    */
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
 * @file test_zip_writer.cpp
 * @brief Tests the zip writer against the zip reader: names, bytes, CRC-32 and alignment.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/zip_archive.hpp>
#include <SushiAI/io/zip_writer.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Returns the bytes of @p text. */
    std::vector<std::byte> bytes_of(const std::string& text)
    {
        std::vector<std::byte> out;
        for (const char c : text)
            out.push_back(static_cast<std::byte>(c));
        return out;
    }

    /** @brief Returns the CRC-32 of @p bytes, bit by bit, as zip states it. */
    std::uint32_t reference_crc(const std::vector<std::byte>& bytes)
    {
        std::uint32_t crc = 0xFFFFFFFFu;
        for (const std::byte byte : bytes)
        {
            crc ^= static_cast<std::uint32_t>(byte);
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc & 1u) != 0 ? 0xEDB88320u ^ (crc >> 1u) : crc >> 1u;
        }
        return ~crc;
    }

    /** @brief Returns the four bytes at @p at, low byte first. */
    std::uint32_t read_u32(const std::vector<std::byte>& bytes, std::size_t at)
    {
        std::uint32_t value = 0;
        for (std::size_t i = 0; i < 4; ++i)
            value |= static_cast<std::uint32_t>(bytes[at + i]) << (8u * i);
        return value;
    }
} // namespace

/** @brief Tests that the reader finds every entry with its bytes, its CRC-32 and its alignment. */
TEST(Small_ZipWriter, TheReaderReadsBackEveryEntry)
{
    const std::vector<std::string> names = {"archive/data.pkl", "archive/data/0",
                                            "archive/version"};
    const std::vector<std::vector<std::byte>> contents = {
        bytes_of("pickle"), bytes_of(std::string(100, 'x')), bytes_of("3\n")};
    const std::vector<std::size_t> alignments = {1, 64, 64};

    IO::ZipWriter writer;
    for (std::size_t i = 0; i < names.size(); ++i)
        writer.add(names[i],
                   SushiRuntime::span<const std::byte>(contents[i].data(), contents[i].size()),
                   alignments[i]);
    const std::vector<std::byte> archive_bytes = writer.finish();

    SA_ASSERT_OK_AND_ASSIGN(const IO::ZipArchive archive,
                            IO::ZipArchive::read(SushiRuntime::span<const std::byte>(
                                                     archive_bytes.data(), archive_bytes.size()),
                                                 "written"));
    ASSERT_EQ(archive.entries().size(), names.size());
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        const IO::ZipEntry& entry = archive.entries()[i];
        EXPECT_EQ(entry.name, names[i]);
        EXPECT_FALSE(entry.deflated);
        const SushiRuntime::span<const std::byte> read = archive.contents(entry);
        EXPECT_EQ(std::vector<std::byte>(read.begin(), read.end()), contents[i]);
        EXPECT_EQ(entry.offset % alignments[i], 0u) << names[i];
    }

    // The CRC-32 of each local header sits 14 bytes after its signature.
    std::size_t local = 0;
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        ASSERT_EQ(read_u32(archive_bytes, local), 0x04034b50u);
        EXPECT_EQ(read_u32(archive_bytes, local + 14), reference_crc(contents[i]));
        local = archive.entries()[i].offset + contents[i].size();
    }
}
