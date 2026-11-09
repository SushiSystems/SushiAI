/**************************************************************************/
/* test_load_sample_mask.cpp                                              */
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
 * @file test_load_sample_mask.cpp
 * @brief Tests that the loader of a sample's mask keeps the channels a mask file stores.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/load_sample_mask.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI::Vision;
using namespace SushiAI::Vision::Datasets;

namespace
{
    /** @brief Returns an image of some channels that holds the given bytes, row by row. */
    Imaging::Image image_with_bytes(Imaging::ImageSize size, std::int32_t channels,
                                    const std::vector<std::uint8_t>& bytes)
    {
        Imaging::Image image(size, channels);
        std::copy(bytes.begin(), bytes.end(), image.data());
        return image;
    }

    /** @brief Returns a list of one sample whose mask is the raw image file written from stored. */
    SampleList list_with_mask(const Support::ScratchDirectory& scratch,
                              const Imaging::Image& stored)
    {
        SampleList list;
        list.files.push_back(scratch.path() / "image.jpg");
        list.files.push_back(Support::write_raw_image(scratch, "mask.saim", stored));
        list.samples.emplace_back();
        list.samples[0].mask = 1u;
        return list;
    }

    /** @brief Returns the bytes of an image, row by row. */
    std::vector<std::uint8_t> bytes_of(const Imaging::Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }
} // namespace

/** @brief Tests that a mask file of three channels decodes to those channels, unchanged. */
TEST(Medium_DatasetsLoadSampleMask, AColorMaskKeepsTheChannelsItStores)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_color_mask");
    const std::vector<std::uint8_t> bytes = {1, 2, 3, 40, 50, 60};
    const SampleList list =
        list_with_mask(scratch, image_with_bytes(Imaging::ImageSize{1, 2}, 3, bytes));

    Imaging::Image mask;
    SA_EXPECT_OK(load_sample_mask(list, 0, mask));

    EXPECT_EQ(mask.size(), (Imaging::ImageSize{1, 2}));
    EXPECT_EQ(mask.channels(), 3);
    EXPECT_EQ(bytes_of(mask), bytes);
}

/** @brief Tests that a mask file of one channel keeps its class bytes, the void byte among them. */
TEST(Medium_DatasetsLoadSampleMask, AGrayMaskKeepsItsClassBytesAndTheVoidByte)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_gray_mask");
    const std::vector<std::uint8_t> classes = {0, 3, 255, 1, 2, 0};
    const SampleList list =
        list_with_mask(scratch, image_with_bytes(Imaging::ImageSize{2, 3}, 1, classes));

    Imaging::Image mask;
    SA_EXPECT_OK(load_sample_mask(list, 0, mask));

    EXPECT_EQ(mask.size(), (Imaging::ImageSize{2, 3}));
    EXPECT_EQ(mask.channels(), 1);
    EXPECT_EQ(bytes_of(mask), classes);
}
