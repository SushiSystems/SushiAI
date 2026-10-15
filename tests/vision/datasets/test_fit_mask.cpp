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
 * @file test_fit_mask.cpp
 * @brief Tests the nearest-neighbour fit of a mask and the loader of a sample's mask file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/core/status.hpp>
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
TEST(Small_DatasetsFitMask, AMaskMustMatchItsImage)
{
    const SampleFitOptions options = options_of(SampleFit::Resize, ImageSize{4, 4});
    Image fitted;

    const Image coloured = Support::uniform_image(ImageSize{2, 2}, 3, 1);
    SA_EXPECT_FAILURE(fit_mask(coloured.view(), coloured.size(), options, fitted),
                      StatusCode::Incompatible, "fit_mask", "the mask has 3 channels");

    const Image mask = Support::uniform_image(ImageSize{2, 2}, 1, 1);
    SA_EXPECT_FAILURE(fit_mask(mask.view(), ImageSize{2, 4}, options, fitted),
                      StatusCode::Incompatible, "fit_mask",
                      "the mask is 2 by 2 pixels; its image is 2 by 4");

    SA_EXPECT_FAILURE(fit_mask(mask.view(), mask.size(),
                               options_of(SampleFit::Resize, ImageSize{0, 4}), fitted),
                      StatusCode::InvalidDescription, "fit_mask", "0 by 4 pixels");
}

/** @brief Tests that the mask file of a sample is decoded as the bytes it stores. */
TEST(Small_DatasetsLoadSampleMask, TheMaskFileIsDecoded)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_mask");
    const std::vector<std::uint8_t> classes = {0, 3, 255, 1, 2, 0};

    SampleList list;
    list.files.push_back(scratch.path() / "image.jpg");
    list.files.push_back(Support::write_raw_image(
        scratch, "mask.png", Support::gray_image(ImageSize{2, 3}, classes)));
    list.samples.emplace_back();
    list.samples[0].mask = 1;

    Image mask;
    SA_EXPECT_OK(load_sample_mask(list, 0, mask));

    EXPECT_EQ(mask.size(), (ImageSize{2, 3}));
    EXPECT_EQ(mask.channels(), 1);
    EXPECT_EQ(bytes_of(mask), classes);
}

/** @brief Tests that a sample without a mask and indices outside the list are refused. */
TEST(Small_DatasetsLoadSampleMask, ASampleMustNameAMaskOfTheList)
{
    SampleList list;
    list.files.push_back("image.jpg");
    list.samples.emplace_back();

    Image mask;
    SA_EXPECT_FAILURE(load_sample_mask(list, 1, mask), StatusCode::InvalidDescription,
                      "load_sample_mask", "sample 1 is outside a list of 1 samples");
    SA_EXPECT_FAILURE(load_sample_mask(list, 0, mask), StatusCode::InvalidDescription,
                      "load_sample_mask", "sample 0 has no mask");

    list.samples[0].mask = 4;
    SA_EXPECT_FAILURE(load_sample_mask(list, 0, mask), StatusCode::InvalidDescription,
                      "sample 0 names the mask file 4 of a list of 1 files");
}

/** @brief Tests that an absent mask file is reported with its path. */
TEST(Small_DatasetsLoadSampleMask, AnAbsentMaskFileIsReported)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_mask_absent");

    SampleList list;
    list.files.push_back(scratch.path() / "absent.png");
    list.samples.emplace_back();
    list.samples[0].mask = 0;

    Image mask;
    SA_EXPECT_FAILURE(load_sample_mask(list, 0, mask), StatusCode::FileNotFound, "absent.png");
}
