/**************************************************************************/
/* test_fit_mask.cpp                                                      */
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
 * @file test_fit_mask.cpp
 * @brief Tests the nearest-neighbour fit of a mask and the loader of a sample's mask file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/load_sample_mask.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;

namespace
{
    /** @brief Returns the bytes of an image, row by row. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns the options of one fit to an input size. */
    SampleFitOptions options_of(SampleFit fit, ImageSize size)
    {
        SampleFitOptions options;
        options.fit = fit;
        options.size = size;
        return options;
    }

    /** @brief Holds a mask, the size of its image and the text fit_mask refuses it with. */
    struct MaskRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the channels of the mask. */
        std::int32_t channels;

        /** @brief Holds the size of the mask. */
        ImageSize mask_size;

        /** @brief Holds the size of the image the mask belongs to. */
        ImageSize image_size;

        /** @brief Holds the text the refusal carries. */
        const char* message;
    };

    /** @brief Holds one refused mask per row. */
    class Small_DatasetsFitMaskRefusal : public ::testing::TestWithParam<MaskRefusal>
    {
    };

    /** @brief Holds the call a sample makes to load its mask and the text it is refused with. */
    struct SampleRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the index of the sample asked for. */
        std::size_t sample;

        /** @brief Holds the mask index the one sample of the list names, or none. */
        std::optional<std::uint32_t> mask;

        /** @brief Holds the text the refusal carries. */
        const char* message;
    };

    /** @brief Holds one refused sample per row. */
    class Small_DatasetsLoadSampleMaskRefusal : public ::testing::TestWithParam<SampleRefusal>
    {
    };
} // namespace

/** @brief Tests that an enlarged mask repeats each class and blends none. */
TEST(Small_DatasetsFitMask, AResizeRepeatsTheNearestClass)
{
    const Image mask = Support::gray_image(ImageSize{2, 2}, {0, 1, 2, 3});

    Image fitted;
    SA_EXPECT_OK(fit_mask(mask.view(), mask.size(),
                          options_of(SampleFit::Resize, ImageSize{4, 4}), fitted));

    EXPECT_EQ(fitted.size(), (ImageSize{4, 4}));
    EXPECT_EQ(fitted.channels(), 1);
    EXPECT_EQ(bytes_of(fitted),
              (std::vector<std::uint8_t>{0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3}));
}

/** @brief Tests that a reduced mask takes the pixel each centre falls in. */
TEST(Small_DatasetsFitMask, AReductionTakesThePixelUnderEachCentre)
{
    const Image mask = Support::gray_image(ImageSize{4, 4}, Support::rising_bytes(16, 0));

    Image fitted;
    SA_EXPECT_OK(fit_mask(mask.view(), mask.size(),
                          options_of(SampleFit::Resize, ImageSize{2, 2}), fitted));

    // The centres of a 2 by 2 result fall in rows and columns 1 and 3 of the source.
    EXPECT_EQ(bytes_of(fitted), (std::vector<std::uint8_t>{5, 7, 13, 15}));
}

/** @brief Tests that a letterbox places the mask as its image and pads with K_MASK_VOID. */
TEST(Small_DatasetsFitMask, ALetterboxPadsWithTheVoidByte)
{
    const Image mask = Support::gray_image(ImageSize{2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});

    Image fitted;
    SA_EXPECT_OK(fit_mask(mask.view(), mask.size(),
                          options_of(SampleFit::Letterbox, ImageSize{4, 4}), fitted));

    // A 2 by 4 image is not scaled up, so one row of padding lies above it and one below.
    const std::uint8_t v = K_MASK_VOID;
    EXPECT_EQ(bytes_of(fitted),
              (std::vector<std::uint8_t>{v, v, v, v, 0, 1, 2, 3, 4, 5, 6, 7, v, v, v, v}));
}

/** @brief Tests that a letterbox that halves the mask samples it by nearest neighbour. */
TEST(Small_DatasetsFitMask, ALetterboxThatReducesSamplesTheNearestPixel)
{
    const Image mask = Support::gray_image(ImageSize{4, 8}, Support::rising_bytes(32, 0));

    Image fitted;
    SA_EXPECT_OK(fit_mask(mask.view(), mask.size(),
                          options_of(SampleFit::Letterbox, ImageSize{4, 4}), fitted));

    // The mask becomes 2 by 4 from source rows 1 and 3 and columns 1, 3, 5 and 7.
    const std::uint8_t v = K_MASK_VOID;
    EXPECT_EQ(bytes_of(fitted), (std::vector<std::uint8_t>{v, v, v, v, 9, 11, 13, 15, 25, 27, 29,
                                                           31, v, v, v, v}));
}

/** @brief Tests that a mask of several channels or of another size than its image fails. */
TEST_P(Small_DatasetsFitMaskRefusal, AMaskMustMatchItsImage)
{
    const MaskRefusal& row = GetParam();
    const Image mask = Support::uniform_image(row.mask_size, row.channels, 1);
    Image fitted;

    SA_EXPECT_FAILURE(fit_mask(mask.view(), row.image_size,
                               options_of(SampleFit::Resize, ImageSize{4, 4}), fitted),
                      StatusCode::Incompatible, "fit_mask", row.message);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsFitMaskRefusal,
    ::testing::Values(MaskRefusal{"ThreeChannels", 3, ImageSize{2, 2}, ImageSize{2, 2},
                                  "the mask has 3 channels"},
                      MaskRefusal{"AnotherSizeThanItsImage", 1, ImageSize{2, 2}, ImageSize{2, 4},
                                  "the mask is 2 by 2 pixels; its image is 2 by 4"}),
    [](const ::testing::TestParamInfo<MaskRefusal>& info) { return std::string(info.param.name); });

/** @brief Tests that a target size of no pixel fails. */
TEST(Small_DatasetsFitMask, ATargetSizeOfNoPixelIsRefused)
{
    const Image mask = Support::uniform_image(ImageSize{2, 2}, 1, 1);
    Image fitted;

    SA_EXPECT_FAILURE(fit_mask(mask.view(), mask.size(),
                               options_of(SampleFit::Resize, ImageSize{0, 4}), fitted),
                      StatusCode::InvalidDescription, "fit_mask", "0 by 4 pixels");
}

/** @brief Tests that a sample without a mask and indices outside the list are refused. */
TEST_P(Small_DatasetsLoadSampleMaskRefusal, ASampleMustNameAMaskOfTheList)
{
    const SampleRefusal& row = GetParam();
    SampleList list;
    list.files.push_back("image.jpg");
    list.samples.emplace_back();
    list.samples[0].mask = row.mask;
    Image mask;

    SA_EXPECT_FAILURE(load_sample_mask(list, row.sample, mask), StatusCode::InvalidDescription,
                      row.message);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsLoadSampleMaskRefusal,
    ::testing::Values(SampleRefusal{"ASampleOutsideTheList", 1, std::nullopt,
                                    "sample 1 is outside a list of 1 samples"},
                      SampleRefusal{"ASampleWithoutAMask", 0, std::nullopt,
                                    "sample 0 has no mask"},
                      SampleRefusal{"AMaskFileOutsideTheList", 0, 4u,
                                    "sample 0 names the mask file 4 of a list of 1 files"}),
    [](const ::testing::TestParamInfo<SampleRefusal>& info) { return std::string(info.param.name); });

/** @brief Tests that an absent mask file is reported with its path. */
TEST(Medium_DatasetsLoadSampleMask, AnAbsentMaskFileIsReported)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_mask_absent");

    SampleList list;
    list.files.push_back(scratch.path() / "absent.png");
    list.samples.emplace_back();
    list.samples[0].mask = 0;

    Image mask;
    SA_EXPECT_FAILURE(load_sample_mask(list, 0, mask), StatusCode::FileNotFound, "absent.png");
}
