/**************************************************************************/
/* test_fit_sample.cpp                                                    */
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
 * @file test_fit_sample.cpp
 * @brief Tests the fit of one loaded sample to the input size of a model.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <support/tolerance.hpp>
#include <support/test_names.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;
using SushiAI::Vision::Imaging::ImageView;
using SushiAI::Vision::Imaging::NormalizedBox;
using SushiAI::Vision::Imaging::PixelBox;

namespace
{
    /**
     * @brief Holds the absolute tolerance of a box coordinate, in pixels: 2048 times the float unit
     *        roundoff, the smallest power of two not tighter than the earlier 1e-4f.
     */
    constexpr float K_PIXEL_TOLERANCE =
        static_cast<float>(2048.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds the padding byte of the default letterbox settings. */
    constexpr std::uint8_t K_PADDING = 114;

    /** @brief Returns the options of a fit to a size. */
    SampleFitOptions options_of(ImageSize size, SampleFit fit)
    {
        SampleFitOptions options;
        options.size = size;
        options.fit = fit;
        return options;
    }

    /** @brief Expects the four corners of a box, each within K_PIXEL_TOLERANCE. */
    void expect_box(const PixelBox& box, float x1, float y1, float x2, float y2)
    {
        EXPECT_NEAR(box.x1, x1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(box.y1, y1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(box.x2, x2, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(box.y2, y2, K_PIXEL_TOLERANCE);
    }
    /** @brief Returns the bytes of an image, row by row. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns a sample of one object of class 1 with a box. */
    Sample sample_with_one_box(const NormalizedBox& box)
    {
        Sample sample;
        sample.objects.add(1, box);
        return sample;
    }

    /** @brief Holds a row of the letterboxed image and the level it holds. */
    struct LetterboxRow
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the index of the row of the fitted image. */
        std::int32_t row;

        /** @brief Holds the level every pixel of the row has. */
        std::uint8_t level;
    };

    /** @brief Holds one row of the letterboxed image per case. */
    class Small_DatasetsFitSampleLetterboxRow : public ::testing::TestWithParam<LetterboxRow>
    {
    };
} // namespace

/** @brief Tests that a resize scales each box coordinate with its axis. */
TEST(Small_DatasetsFitSample, ResizeScalesTheLabelsWithEachAxis)
{
    const Image source = Support::uniform_image(ImageSize{2, 4}, 1, 100);
    const Sample sample = sample_with_one_box(NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f});

    FittedSample fitted;
    SA_EXPECT_OK(fit_sample(sample, source.view(),
                            options_of(ImageSize{4, 4}, SampleFit::Resize), fitted));

    ASSERT_EQ(fitted.objects.size(), 1u);
    EXPECT_EQ(fitted.objects.classes()[0], 1);
    expect_box(fitted.objects.boxes()[0], 1.0f, 1.0f, 3.0f, 3.0f);
}

/** @brief Tests that a resize fills the image with the source level and carries file and class. */
TEST(Small_DatasetsFitSample, ResizeFillsTheImageAndCarriesTheFileAndTheClass)
{
    const Image source = Support::uniform_image(ImageSize{2, 4}, 1, 100);
    Sample sample = sample_with_one_box(NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f});
    sample.file = 6;
    sample.class_index = 3;

    FittedSample fitted;
    SA_EXPECT_OK(fit_sample(sample, source.view(),
                            options_of(ImageSize{4, 4}, SampleFit::Resize), fitted));

    EXPECT_EQ(fitted.image.size(), (ImageSize{4, 4}));
    EXPECT_EQ(fitted.image.channels(), 1);
    EXPECT_EQ(bytes_of(fitted.image), std::vector<std::uint8_t>(16, 100));
    EXPECT_EQ(fitted.file, 6u);
    ASSERT_TRUE(fitted.class_index.has_value());
    EXPECT_EQ(*fitted.class_index, 3);
}

/** @brief Tests that a letterbox pads the short axis and fills the rest with the source. */
TEST_P(Small_DatasetsFitSampleLetterboxRow, LetterboxPadsTheShortAxis)
{
    const LetterboxRow& row = GetParam();
    const Image source = Support::uniform_image(ImageSize{2, 4}, 1, 200);

    FittedSample fitted;
    SA_EXPECT_OK(fit_sample(Sample{}, source.view(),
                            options_of(ImageSize{4, 4}, SampleFit::Letterbox), fitted));

    ASSERT_EQ(fitted.image.size(), (ImageSize{4, 4}));
    const std::uint8_t* pixels = fitted.image.row(row.row);
    EXPECT_EQ(std::vector<std::uint8_t>(pixels, pixels + 4),
              std::vector<std::uint8_t>(4, row.level));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsFitSampleLetterboxRow,
    ::testing::Values(LetterboxRow{"TopRowIsPadding", 0, K_PADDING},
                      LetterboxRow{"SecondRowIsTheSource", 1, 200},
                      LetterboxRow{"ThirdRowIsTheSource", 2, 200},
                      LetterboxRow{"BottomRowIsPadding", 3, K_PADDING}),
    [](const ::testing::TestParamInfo<LetterboxRow>& info) { return ::SushiAI::Test::parameter_name(info.param.name); });

/** @brief Tests that a letterbox moves the boxes by the padding and invents no class. */
TEST(Small_DatasetsFitSample, LetterboxMovesTheLabelsByThePadding)
{
    const Image source = Support::uniform_image(ImageSize{2, 4}, 1, 200);
    Sample sample;
    sample.objects.add(0, NormalizedBox{0.5f, 0.5f, 0.5f, 1.0f});

    FittedSample fitted;
    SA_EXPECT_OK(fit_sample(sample, source.view(),
                            options_of(ImageSize{4, 4}, SampleFit::Letterbox), fitted));

    EXPECT_FALSE(fitted.class_index.has_value());
    ASSERT_EQ(fitted.objects.size(), 1u);
    expect_box(fitted.objects.boxes()[0], 1.0f, 1.0f, 3.0f, 3.0f);
}

/** @brief Tests that an object whose box has no area is removed and the rest keep their order. */
TEST(Small_DatasetsFitSample, AnObjectWithoutAreaIsRemoved)
{
    const Image source = Support::uniform_image(ImageSize{4, 4}, 3, 9);

    Sample sample;
    sample.objects.add(2, NormalizedBox{0.5f, 0.5f, 0.0f, 0.5f});
    sample.objects.add(5, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f});

    FittedSample fitted;
    SA_EXPECT_OK(fit_sample(sample, source.view(),
                            options_of(ImageSize{4, 4}, SampleFit::Letterbox), fitted));

    ASSERT_EQ(fitted.objects.size(), 1u);
    EXPECT_EQ(fitted.objects.classes()[0], 5);
}

/** @brief Tests that an input size no image has is refused before anything is read. */
TEST(Small_DatasetsFitSample, AnInputSizeOutsideTheBoundsIsRefused)
{
    const Image source = Support::uniform_image(ImageSize{2, 2}, 1, 0);

    FittedSample fitted;
    SA_EXPECT_FAILURE(fit_sample(Sample{}, source.view(),
                                 options_of(ImageSize{0, 4}, SampleFit::Resize), fitted),
                      StatusCode::InvalidDescription, "fit_sample", "0 by 4");
}

/** @brief Tests that a source of no pixels stops the process in the resize that reads it. */
TEST(Medium_DatasetsFitSampleFatal, ASourceOfNoPixelsStopsTheProcess)
{
    FittedSample fitted;
    SA_EXPECT_FATAL(static_cast<void>(fit_sample(
                        Sample{}, ImageView(), options_of(ImageSize{4, 4}, SampleFit::Resize),
                        fitted)),
                    "resize_bilinear", "no pixels");
}

/** @brief Tests that a letterbox that leaves a resized side of zero pixels is refused. */
TEST(Small_DatasetsFitSample, ALetterboxThatLeavesNoPixelsIsRefused)
{
    const Image source = Support::uniform_image(ImageSize{1, 4000}, 1, 0);

    FittedSample fitted;
    SA_EXPECT_FAILURE(fit_sample(Sample{}, source.view(),
                                 options_of(ImageSize{4, 4}, SampleFit::Letterbox), fitted),
                      StatusCode::InvalidDescription, "plan_letterbox", "zero pixels");
}
