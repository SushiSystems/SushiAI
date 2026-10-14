/**************************************************************************/
/* test_load_sample_image.cpp                                             */
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
 * @file test_load_sample_image.cpp
 * @brief Tests the loader of one sample's pixels, stored raw or as an image file.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Codecs::DecodeOptions;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;

namespace
{
    /** @brief Returns a copy of an image's bytes. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns a list of one file and one sample that names it. */
    SampleList list_of(const std::filesystem::path& file)
    {
        SampleList list;
        list.files.push_back(file);
        list.samples.emplace_back();
        return list;
    }
} // namespace

/** @brief Tests that a sample with stored pixels is read from its offset in the file. */
TEST(Unit_DatasetsLoadSampleImage, StoredPixelsAreReadFromTheirOffset)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_stored");
    SampleList list = list_of(scratch.write("pixels.bin", Support::rising_bytes(10, 0)));

    StoredPixels stored;
    stored.offset = 2;
    stored.size = ImageSize{2, 2};
    stored.channels = 1;
    list.samples[0].stored = stored;

    Image image;
    SA_EXPECT_OK(load_sample_image(list, 0, DecodeOptions{}, image));

    EXPECT_EQ(image.size(), (ImageSize{2, 2}));
    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(bytes_of(image), Support::rising_bytes(4, 2));
}

/** @brief Tests that a sample without stored pixels is decoded from its image file. */
TEST(Unit_DatasetsLoadSampleImage, AnImageFileIsDecoded)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_file");
    const std::filesystem::path file = scratch.path() / "image.saim";

    Image written(ImageSize{2, 3}, 3);
    const std::vector<std::uint8_t> pixels = Support::rising_bytes(written.byte_count(), 7);
    std::copy(pixels.begin(), pixels.end(), written.data());
    SushiAI::Vision::Imaging::write_raw_image_file(
        file, written.view(), SushiAI::Vision::Imaging::ChannelOrder::RGB);

    Image image;
    SA_EXPECT_OK(load_sample_image(list_of(file), 0, DecodeOptions{}, image));

    EXPECT_EQ(image.size(), (ImageSize{2, 3}));
    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image), pixels);
}

/** @brief Tests that a sample index past the list is refused with its number. */
TEST(Unit_DatasetsLoadSampleImage, ASampleOutsideTheListIsRefused)
{
    Image image;
    SA_EXPECT_FAILURE(load_sample_image(list_of("unread.png"), 3, DecodeOptions{}, image),
                      StatusCode::InvalidDescription, "load_sample_image", "sample 3",
                      "1 samples");
}

/** @brief Tests that a sample naming a file the list lacks is refused. */
TEST(Unit_DatasetsLoadSampleImage, AFileIndexOutsideTheListIsRefused)
{
    SampleList list = list_of("unread.png");
    list.samples[0].file = 5;

    Image image;
    SA_EXPECT_FAILURE(load_sample_image(list, 0, DecodeOptions{}, image),
                      StatusCode::InvalidDescription, "load_sample_image", "file 5");
}

/** @brief Tests that an absent image file is a failure that names the file. */
TEST(Unit_DatasetsLoadSampleImage, AnAbsentFileIsNamed)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_absent");

    Image image;
    SA_EXPECT_FAILURE(
        load_sample_image(list_of(scratch.path() / "absent.saim"), 0, DecodeOptions{}, image),
        StatusCode::FileNotFound, "absent.saim");
}
