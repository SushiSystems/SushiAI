/**************************************************************************/
/* test_idx_file.cpp                                                      */
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
 * @file test_idx_file.cpp
 * @brief Tests the readers of an IDX file's header and payload.
 * @author Mustafa Garip
 */
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/idx_file.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns a source holding the bytes as the file row.idx. */
    MemorySource source_of(const std::vector<std::uint8_t>& bytes)
    {
        MemorySource source;
        source.add_file("row.idx", Support::text_of(bytes));
        return source;
    }

    /** @brief Holds an element type and the byte count the format gives it. */
    struct ElementRow
    {
        std::string name;       /**< The row's name. */
        IDXElement element;     /**< The element type handed to idx_element_bytes. */
        std::size_t bytes;      /**< The byte count of one element. */
    };

    /** @brief Holds a malformed header and the cause its message names. */
    struct MalformedRow
    {
        std::string name;                 /**< The row's name. */
        std::vector<std::uint8_t> bytes;  /**< The whole file. */
        std::string cause;                /**< The text the message holds besides the path. */
    };

    /** @brief Holds a file of a size its header does not declare and the two counts named. */
    struct SizeRow
    {
        std::string name;                 /**< The row's name. */
        std::vector<std::uint8_t> bytes;  /**< The whole file. */
        std::string declared;             /**< The text naming the size the header declares. */
        std::string held;                 /**< The text naming the size the file holds. */
    };

    /** @brief Runs idx_element_bytes over each row. */
    class Small_DatasetsIdxFileElementBytes : public ::testing::TestWithParam<ElementRow>
    {
    };

    /** @brief Runs the header reader over each malformed header. */
    class Small_DatasetsIdxFileMalformed : public ::testing::TestWithParam<MalformedRow>
    {
    };

    /** @brief Runs the header reader over each file of the wrong size. */
    class Small_DatasetsIdxFileWrongSize : public ::testing::TestWithParam<SizeRow>
    {
    };

    /** @brief Names a test after its element row. */
    std::string element_row_name(const ::testing::TestParamInfo<ElementRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a test after its malformed row. */
    std::string malformed_row_name(const ::testing::TestParamInfo<MalformedRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a test after its size row. */
    std::string size_row_name(const ::testing::TestParamInfo<SizeRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that an element type has the byte count the format gives it. */
TEST_P(Small_DatasetsIdxFileElementBytes, ElementBytesFollowTheFormat)
{
    EXPECT_EQ(idx_element_bytes(GetParam().element), GetParam().bytes);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsIdxFileElementBytes,
    ::testing::Values(
        ElementRow{"UInt8", IDXElement::UInt8, 1u},
        ElementRow{"Int8", IDXElement::Int8, 1u},
        ElementRow{"Int16", IDXElement::Int16, 2u},
        ElementRow{"Int32", IDXElement::Int32, 4u},
        ElementRow{"Float32", IDXElement::Float32, 4u},
        ElementRow{"Float64", IDXElement::Float64, 8u}),
    element_row_name);

/** @brief Tests that a header gives the element type, the extents and the payload size. */
TEST(Small_DatasetsIdxFile, AHeaderGivesTheElementTheExtentsAndThePayloadSize)
{
    const MemorySource source =
        source_of(Support::idx_bytes(0x08, {2, 3, 4}, Support::rising_bytes(24, 0)));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header("row.idx", source));

    EXPECT_EQ(header.element, IDXElement::UInt8);
    EXPECT_EQ(header.extents, (std::vector<std::uint32_t>{2, 3, 4}));
    EXPECT_EQ(header.header_bytes(), 16u);
    EXPECT_EQ(header.payload_bytes, 24u);
}

/** @brief Tests that an extent is read with its most significant byte first. */
TEST(Small_DatasetsIdxFile, AnExtentIsReadMostSignificantByteFirst)
{
    const MemorySource source =
        source_of(Support::idx_bytes(0x08, {258}, Support::rising_bytes(258, 0)));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header("row.idx", source));

    EXPECT_EQ(header.extents, (std::vector<std::uint32_t>{258}));
}

/** @brief Tests that the payload size counts the bytes of a wider element. */
TEST(Small_DatasetsIdxFile, ThePayloadSizeCountsTheElementBytes)
{
    const MemorySource source =
        source_of(Support::idx_bytes(0x0C, {3}, Support::rising_bytes(12, 0)));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header("row.idx", source));

    EXPECT_EQ(header.element, IDXElement::Int32);
    EXPECT_EQ(header.payload_bytes, 12u);
}

/** @brief Tests that the payload is every byte after the header, in file order. */
TEST(Small_DatasetsIdxFile, ThePayloadIsEveryByteAfterTheHeader)
{
    const MemorySource source = source_of(Support::idx_bytes(0x08, {5}, {7, 0, 9, 3, 1}));
    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header("row.idx", source));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::uint8_t> payload,
                            read_idx_payload("row.idx", header, source));

    EXPECT_EQ(payload, (std::vector<std::uint8_t>{7, 0, 9, 3, 1}));
}

/** @brief Tests that each malformed header is refused with the file and the cause. */
TEST_P(Small_DatasetsIdxFileMalformed, AMalformedHeaderIsRefused)
{
    const MemorySource source = source_of(GetParam().bytes);

    SA_EXPECT_FAILURE(read_idx_header("row.idx", source), StatusCode::InvalidFile, "row.idx",
                      GetParam().cause);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsIdxFileMalformed,
    ::testing::Values(
        MalformedRow{"CutShort", {0, 0, 8}, "leading bytes"},
        MalformedRow{"NonZeroSecondByte", {0, 1, 8, 1, 0, 0, 0, 0}, "two zero bytes"},
        MalformedRow{"UnknownElementCode", {0, 0, 10, 1, 0, 0, 0, 0}, "element code 10"},
        MalformedRow{"NoDimension", {0, 0, 8, 0}, "no dimension"},
        MalformedRow{"ExtentCutShort", {0, 0, 8, 2, 0, 0, 0, 1, 0}, "dimension 1 of 2"}),
    malformed_row_name);

/** @brief Tests that a file shorter or longer than its header declares is refused. */
TEST_P(Small_DatasetsIdxFileWrongSize, AFileOfAnotherSizeThanItsHeaderDeclaresIsRefused)
{
    const MemorySource source = source_of(GetParam().bytes);

    SA_EXPECT_FAILURE(read_idx_header("row.idx", source), StatusCode::InvalidFile, "row.idx",
                      GetParam().declared, GetParam().held);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsIdxFileWrongSize,
    ::testing::Values(
        SizeRow{"Shorter", Support::idx_bytes(0x08, {2, 2}, {1, 2, 3}), "declares 4 bytes",
                "holds 15 bytes"},
        SizeRow{"Longer", Support::idx_bytes(0x08, {2}, {1, 2, 3}), "declares 2 bytes",
                "holds 11 bytes"}),
    size_row_name);

/** @brief Tests that extents whose product passes 64 bits are refused, not wrapped. */
TEST(Small_DatasetsIdxFile, ExtentsThatOverflowAreRefused)
{
    const std::uint32_t largest = 0xFFFFFFFFu;
    const MemorySource source =
        source_of(Support::idx_bytes(0x0E, {largest, largest, largest}, {}));

    SA_EXPECT_FAILURE(read_idx_header("row.idx", source), StatusCode::InvalidFile, "row.idx",
                      "overflow");
}

/** @brief Tests that a path that names nothing is reported as absent, with the path. */
TEST(Medium_DatasetsIdxFile, AnAbsentFileIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_absent");

    SA_EXPECT_FAILURE(read_idx_header(scratch.path() / "absent.idx"), StatusCode::FileNotFound,
                      "absent.idx");
}

/** @brief Tests that a header and a payload read from the file system as from memory. */
TEST(Medium_DatasetsIdxFile, AHeaderAndItsPayloadReadFromAFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_disk");
    const std::filesystem::path file =
        scratch.write("disk.idx", Support::idx_bytes(0x08, {2, 2}, {1, 2, 3, 4}));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header(file));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::uint8_t> payload,
                            read_idx_payload(file, header));

    EXPECT_EQ(header.extents, (std::vector<std::uint32_t>{2, 2}));
    EXPECT_EQ(payload, (std::vector<std::uint8_t>{1, 2, 3, 4}));
}
