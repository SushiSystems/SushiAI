/**************************************************************************/
/* test_datasets_fit_sample.cpp                                           */
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
 * @file test_datasets_fit_sample.cpp
 * @brief Tests the fit of one loaded sample to the input size of a model.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <common/test_status.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;
using SushiAI::Vision::Imaging::ImageView;
using SushiAI::Vision::Imaging::NormalizedBox;
using SushiAI::Vision::Imaging::PixelBox;

namespace
{
    /** @brief Holds the absolute tolerance of a box coordinate, in pixels. */
    constexpr float K_PIXEL_TOLERANCE = 1e-4f;

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
} // namespace

/** @brief Tests that a resize scales the image and each box coordinate with its axis. */
TEST(Unit_DatasetsFitSample, ResizeScalesTheLabelsWithEachAxis)
{
    const Image source = Support::uniform_image(ImageSize{2, 4}, 1, 100);

    Sample sample;
    sample.file = 6;
    sample.class_index = 3;
    sample.objects.add(1, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f});

    FittedSample fitted;
    SA_EXPECT_OK(fit_sample(sample, source.view(),
                            options_of(ImageSize{4, 4}, SampleFit::Resize), fitted));

    EXPECT_EQ(fitted.image.size(), (ImageSize{4, 4}));
    EXPECT_EQ(fitted.image.channels(), 1);
    for (std::size_t i = 0; i < fitted.image.byte_count(); ++i)
        EXPECT_EQ(fitted.image.data()[i], 100) << "byte " << i;

    EXPECT_EQ(fitted.file, 6u);
    ASSERT_TRUE(fitted.class_index.has_value());
    EXPECT_EQ(*fitted.class_index, 3);

    ASSERT_EQ(fitted.objects.size(), 1u);
    EXPECT_EQ(fitted.objects.classes()[0], 1);
    expect_box(fitted.objects.boxes()[0], 1.0f, 1.0f, 3.0f, 3.0f);
}

/** @brief Tests that a letterbox pads the short axis and moves the boxes by the padding. */
TEST(Unit_DatasetsFitSample, LetterboxPadsAndMovesTheLabels)
{
    const Image source = Support::uniform_image(ImageSize{2, 4}, 1, 200);

    Sample sample;
    sample.objects.add(0, NormalizedBox{0.5f, 0.5f, 0.5f, 1.0f});

    FittedSample fitted;
    SA_EXPECT_OK(fit_sample(sample, source.view(),
                            options_of(ImageSize{4, 4}, SampleFit::Letterbox), fitted));

    ASSERT_EQ(fitted.image.size(), (ImageSize{4, 4}));
    for (std::int32_t x = 0; x < 4; ++x)
    {
        EXPECT_EQ(fitted.image.row(0)[x], K_PADDING) << "column " << x;
        EXPECT_EQ(fitted.image.row(1)[x], 200) << "column " << x;
        EXPECT_EQ(fitted.image.row(2)[x], 200) << "column " << x;
        EXPECT_EQ(fitted.image.row(3)[x], K_PADDING) << "column " << x;
    }

    EXPECT_FALSE(fitted.class_index.has_value());
    ASSERT_EQ(fitted.objects.size(), 1u);
    expect_box(fitted.objects.boxes()[0], 1.0f, 1.0f, 3.0f, 3.0f);
}

/** @brief Tests that an object whose box has no area is removed and the rest keep their order. */
TEST(Unit_DatasetsFitSample, AnObjectWithoutAreaIsRemoved)
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
TEST(Unit_DatasetsFitSample, AnInputSizeOutsideTheBoundsIsRefused)
{
    const Image source = Support::uniform_image(ImageSize{2, 2}, 1, 0);

    FittedSample fitted;
    SA_EXPECT_FAILURE(fit_sample(Sample{}, source.view(),
                                 options_of(ImageSize{0, 4}, SampleFit::Resize), fitted),
                      StatusCode::InvalidDescription, "fit_sample", "0 by 4");
}

/** @brief Tests that a source of no pixels is a failure and not a thrown exception. */
TEST(Unit_DatasetsFitSample, ASourceOfNoPixelsIsAFailure)
{
    FittedSample fitted;
    SA_EXPECT_FAILURE(fit_sample(Sample{}, ImageView(),
                                 options_of(ImageSize{4, 4}, SampleFit::Resize), fitted),
                      StatusCode::InvalidFile, "fit_sample");
}
