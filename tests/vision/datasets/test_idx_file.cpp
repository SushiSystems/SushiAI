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
 * @file test_idx_file.cpp
 * @brief Tests the readers of an IDX file's header and payload.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/idx_file.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

/** @brief Tests that each element type has the byte count the format gives it. */
TEST(Unit_DatasetsIdxFile, ElementBytesFollowTheFormat)
{
    EXPECT_EQ(idx_element_bytes(IDXElement::UInt8), 1u);
    EXPECT_EQ(idx_element_bytes(IDXElement::Int8), 1u);
    EXPECT_EQ(idx_element_bytes(IDXElement::Int16), 2u);
    EXPECT_EQ(idx_element_bytes(IDXElement::Int32), 4u);
    EXPECT_EQ(idx_element_bytes(IDXElement::Float32), 4u);
    EXPECT_EQ(idx_element_bytes(IDXElement::Float64), 8u);
}

/** @brief Tests that a header gives the element type, the extents and the payload size. */
TEST(Unit_DatasetsIdxFile, AHeaderGivesTheElementTheExtentsAndThePayloadSize)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_header");
    const std::filesystem::path file = scratch.write(
        "images.idx", Support::idx_bytes(0x08, {2, 3, 4}, Support::rising_bytes(24, 0)));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header(file));

    EXPECT_EQ(header.element, IDXElement::UInt8);
    EXPECT_EQ(header.extents, (std::vector<std::uint32_t>{2, 3, 4}));
    EXPECT_EQ(header.header_bytes(), 16u);
    EXPECT_EQ(header.payload_bytes, 24u);
}

/** @brief Tests that an extent is read with its most significant byte first. */
TEST(Unit_DatasetsIdxFile, AnExtentIsReadMostSignificantByteFirst)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_extent");
    const std::filesystem::path file =
        scratch.write("wide.idx", Support::idx_bytes(0x08, {258}, Support::rising_bytes(258, 0)));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header(file));

    EXPECT_EQ(header.extents, (std::vector<std::uint32_t>{258}));
}

/** @brief Tests that the payload size counts the bytes of a wider element. */
TEST(Unit_DatasetsIdxFile, ThePayloadSizeCountsTheElementBytes)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_int32");
    const std::filesystem::path file =
        scratch.write("ints.idx", Support::idx_bytes(0x0C, {3}, Support::rising_bytes(12, 0)));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header(file));

    EXPECT_EQ(header.element, IDXElement::Int32);
    EXPECT_EQ(header.payload_bytes, 12u);
}

/** @brief Tests that the payload is every byte after the header, in file order. */
TEST(Unit_DatasetsIdxFile, ThePayloadIsEveryByteAfterTheHeader)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_payload");
    const std::filesystem::path file =
        scratch.write("labels.idx", Support::idx_bytes(0x08, {5}, {7, 0, 9, 3, 1}));

    SA_ASSERT_OK_AND_ASSIGN(const IDXHeader header, read_idx_header(file));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::uint8_t> payload,
                            read_idx_payload(file, header));

    EXPECT_EQ(payload, (std::vector<std::uint8_t>{7, 0, 9, 3, 1}));
}

/** @brief Tests that a path that names nothing is reported as absent, with the path. */
TEST(Unit_DatasetsIdxFile, AnAbsentFileIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_absent");

    SA_EXPECT_FAILURE(read_idx_header(scratch.path() / "absent.idx"), StatusCode::FileNotFound,
                      "absent.idx");
}

/** @brief Tests that each malformed header is refused with the file and the cause. */
TEST(Unit_DatasetsIdxFile, AMalformedHeaderIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_malformed");

    SA_EXPECT_FAILURE(read_idx_header(scratch.write("cut.idx", {0, 0, 8})),
                      StatusCode::InvalidFile, "cut.idx", "leading bytes");
    SA_EXPECT_FAILURE(read_idx_header(scratch.write("magic.idx", {0, 1, 8, 1, 0, 0, 0, 0})),
                      StatusCode::InvalidFile, "magic.idx", "two zero bytes");
    SA_EXPECT_FAILURE(read_idx_header(scratch.write("code.idx", {0, 0, 10, 1, 0, 0, 0, 0})),
                      StatusCode::InvalidFile, "code.idx", "element code 10");
    SA_EXPECT_FAILURE(read_idx_header(scratch.write("rank.idx", {0, 0, 8, 0})),
                      StatusCode::InvalidFile, "rank.idx", "no dimension");
    SA_EXPECT_FAILURE(read_idx_header(scratch.write("extent.idx", {0, 0, 8, 2, 0, 0, 0, 1, 0})),
                      StatusCode::InvalidFile, "extent.idx", "dimension 1 of 2");
}

/** @brief Tests that a file shorter or longer than its header declares is refused. */
TEST(Unit_DatasetsIdxFile, AFileOfAnotherSizeThanItsHeaderDeclaresIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_size");

    SA_EXPECT_FAILURE(
        read_idx_header(scratch.write("short.idx", Support::idx_bytes(0x08, {2, 2}, {1, 2, 3}))),
        StatusCode::InvalidFile, "short.idx", "declares 4 bytes", "holds 15 bytes");
    SA_EXPECT_FAILURE(
        read_idx_header(scratch.write("long.idx", Support::idx_bytes(0x08, {2}, {1, 2, 3}))),
        StatusCode::InvalidFile, "long.idx", "declares 2 bytes", "holds 11 bytes");
}

/** @brief Tests that extents whose product passes 64 bits are refused, not wrapped. */
TEST(Unit_DatasetsIdxFile, ExtentsThatOverflowAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_file_overflow");
    const std::uint32_t largest = 0xFFFFFFFFu;

    SA_EXPECT_FAILURE(
        read_idx_header(scratch.write(
            "huge.idx", Support::idx_bytes(0x0E, {largest, largest, largest}, {}))),
        StatusCode::InvalidFile, "huge.idx", "overflow");
}
