/**************************************************************************/
/* test_datasets_stored_pixels.cpp                                        */
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
 * @file test_datasets_stored_pixels.cpp
 * @brief Tests the copy of one sample's raw pixels out of a file into an image.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <common/test_status.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;

namespace
{
    /** @brief Returns a copy of an image's bytes. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns a description of pixels at an offset. */
    StoredPixels pixels_at(std::uint64_t offset, ImageSize size, std::int32_t channels,
                           PlaneLayout layout)
    {
        StoredPixels pixels;
        pixels.offset = offset;
        pixels.size = size;
        pixels.channels = channels;
        pixels.layout = layout;
        return pixels;
    }
} // namespace

/** @brief Tests that the byte count is height times width times channels. */
TEST(Unit_DatasetsStoredPixels, ByteCountIsTheProductOfTheExtents)
{
    EXPECT_EQ(pixels_at(0, ImageSize{4, 5}, 3, PlaneLayout::Planar).byte_count(), 60u);
}

/** @brief Tests that interleaved bytes are copied as they are, starting at the offset. */
TEST(Unit_DatasetsStoredPixels, InterleavedBytesAreCopiedFromTheOffset)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_stored_interleaved");
    const std::filesystem::path file = scratch.write("pixels.bin", Support::rising_bytes(20, 0));

    Image image;
    SA_EXPECT_OK(read_stored_pixels(
        file, pixels_at(5, ImageSize{2, 3}, 2, PlaneLayout::Interleaved), image));

    EXPECT_EQ(image.height(), 2);
    EXPECT_EQ(image.width(), 3);
    EXPECT_EQ(image.channels(), 2);
    EXPECT_EQ(bytes_of(image), Support::rising_bytes(12, 5));
}

/** @brief Tests that plane c of a planar image becomes channel c of every pixel. */
TEST(Unit_DatasetsStoredPixels, PlanarBytesAreInterleaved)
{
    const std::vector<std::uint8_t> stored = {9, 1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24};
    std::istringstream in(std::string(stored.begin(), stored.end()), std::ios::binary);

    Image image;
    SA_EXPECT_OK(
        read_stored_pixels(in, pixels_at(1, ImageSize{2, 2}, 3, PlaneLayout::Planar), image));

    const std::vector<std::uint8_t> expected = {1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24};
    EXPECT_EQ(bytes_of(image), expected);
}

/** @brief Tests that a plane longer than one read is copied whole. */
TEST(Unit_DatasetsStoredPixels, APlaneLongerThanOneChunkIsCopiedWhole)
{
    const std::size_t plane_bytes = 100 * 100;
    const std::vector<std::uint8_t> stored = Support::rising_bytes(2 * plane_bytes, 0);
    std::istringstream in(std::string(stored.begin(), stored.end()), std::ios::binary);

    Image image;
    SA_EXPECT_OK(
        read_stored_pixels(in, pixels_at(0, ImageSize{100, 100}, 2, PlaneLayout::Planar), image));

    for (std::size_t pixel = 0; pixel < plane_bytes; pixel += 997)
    {
        EXPECT_EQ(image.data()[pixel * 2], stored[pixel]) << "pixel " << pixel;
        EXPECT_EQ(image.data()[pixel * 2 + 1], stored[plane_bytes + pixel]) << "pixel " << pixel;
    }
}

/** @brief Tests that a file that ends inside the pixels is invalid and named. */
TEST(Unit_DatasetsStoredPixels, AFileThatEndsEarlyIsInvalid)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_stored_short");
    const std::filesystem::path file = scratch.write("short.bin", Support::rising_bytes(10, 0));

    Image image;
    SA_EXPECT_FAILURE(
        read_stored_pixels(file, pixels_at(4, ImageSize{2, 2}, 3, PlaneLayout::Interleaved),
                           image),
        StatusCode::InvalidFile, "short.bin", "6 of 12");
}

/** @brief Tests that a path that names no file is invalid and named. */
TEST(Unit_DatasetsStoredPixels, AFileThatCannotBeOpenedIsInvalid)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_stored_absent");

    Image image;
    SA_EXPECT_FAILURE(
        read_stored_pixels(scratch.path() / "absent.bin",
                           pixels_at(0, ImageSize{1, 1}, 1, PlaneLayout::Interleaved), image),
        StatusCode::InvalidFile, "absent.bin");
}

/** @brief Tests that a size or a channel count no image can have is refused. */
TEST(Unit_DatasetsStoredPixels, ExtentsOutsideAnImageAreRefused)
{
    std::istringstream in(std::string(16, 'x'), std::ios::binary);
    Image image;

    SA_EXPECT_FAILURE(
        read_stored_pixels(in, pixels_at(0, ImageSize{0, 2}, 3, PlaneLayout::Planar), image),
        StatusCode::InvalidDescription, "0 by 2");
    SA_EXPECT_FAILURE(
        read_stored_pixels(in, pixels_at(0, ImageSize{2, 2}, 5, PlaneLayout::Planar), image),
        StatusCode::InvalidDescription, "5 channels");
}
