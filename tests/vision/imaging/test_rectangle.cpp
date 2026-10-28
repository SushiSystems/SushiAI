/**************************************************************************/
/* test_rectangle.cpp                                                     */
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
 * @file test_rectangle.cpp
 * @brief Tests the batch order and the batch shapes of rectangular validation.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/rectangle.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Returns seven image sizes with ratios below, at and above one, two of them equal. */
    std::vector<ImageSize> make_sizes()
    {
        return {ImageSize{480, 640}, ImageSize{640, 480}, ImageSize{500, 500}, ImageSize{427, 640},
                ImageSize{640, 427}, ImageSize{375, 500}, ImageSize{333, 500}};
    }

    /** @brief Returns the settings of a batch size, an image size and a pad at stride 32. */
    RectangleSettings make_settings(std::int32_t batch_size, std::int32_t image_size, double pad)
    {
        RectangleSettings settings;
        settings.batch_size = batch_size;
        settings.image_size = image_size;
        settings.pad = pad;
        return settings;
    }
} // namespace

/** @brief Tests that the settings default to the reference's validation run. */
TEST(Small_ImagingRectangle, SettingsDefaultToTheReferenceValidationRun)
{
    const RectangleSettings settings;

    EXPECT_EQ(settings.image_size, 640);
    EXPECT_EQ(settings.batch_size, 32);
    EXPECT_EQ(settings.stride, 32);
    EXPECT_NEAR(settings.pad, 0.5, 1e-12);
}

/** @brief Tests that images are ordered by rising height over width, ties in index order. */
TEST(Small_ImagingRectangle, OrderRisesWithTheAspectRatioAndKeepsTies)
{
    const RectangularBatches batches =
        require_value(plan_rectangular_batches(make_sizes(), make_settings(3, 640, 0.5)));

    // Images 0 and 5 both have a ratio of 0.75 and stay in dataset order.
    EXPECT_EQ(batches.order, (std::vector<std::size_t>{6, 3, 0, 5, 2, 1, 4}));
    EXPECT_EQ(batches.batch_size, 3u);
}

/** @brief Tests that the batch shapes equal the ones the reference's set_rectangle gives. */
TEST(Small_ImagingRectangle, ShapesMatchTheReferenceArithmetic)
{
    // data/base.py:405-427 run on these sizes with numpy gives the shapes below.
    const std::vector<ImageSize> sizes = make_sizes();

    const RectangularBatches padded =
        require_value(plan_rectangular_batches(sizes, make_settings(3, 640, 0.5)));
    const RectangularBatches tight =
        require_value(plan_rectangular_batches(sizes, make_settings(2, 640, 0.0)));
    const RectangularBatches single =
        require_value(plan_rectangular_batches(sizes, make_settings(7, 320, 0.5)));

    EXPECT_EQ(padded.shapes,
              (std::vector<ImageSize>{ImageSize{512, 672}, ImageSize{672, 672}, ImageSize{672,
              448}}));
    EXPECT_EQ(tight.shapes,
              (std::vector<ImageSize>{ImageSize{448, 640}, ImageSize{480, 640}, ImageSize{640, 640},
              ImageSize{640, 448}}));
    EXPECT_EQ(single.shapes, (std::vector<ImageSize>{ImageSize{352, 352}}));
}

/** @brief Tests that a position's shape is the shape of the batch that holds it. */
TEST(Small_ImagingRectangle, ShapeAtNamesTheBatchOfAPosition)
{
    const RectangularBatches batches =
        require_value(plan_rectangular_batches(make_sizes(), make_settings(3, 640, 0.5)));

    ASSERT_EQ(batches.shapes.size(), 3u);
    EXPECT_EQ(batches.shape_at(0), (ImageSize{512, 672}));
    EXPECT_EQ(batches.shape_at(2), (ImageSize{512, 672}));
    EXPECT_EQ(batches.shape_at(3), (ImageSize{672, 672}));
    EXPECT_EQ(batches.shape_at(6), (ImageSize{672, 448}));
}

/** @brief Tests that every side of every shape is a whole number of strides. */
TEST(Small_ImagingRectangle, EverySideIsAMultipleOfTheStride)
{
    RectangleSettings settings = make_settings(2, 600, 0.5);
    settings.stride = 64;

    const RectangularBatches batches =
        require_value(plan_rectangular_batches(make_sizes(), settings));

    ASSERT_EQ(batches.shapes.size(), 4u);
    for (const ImageSize shape : batches.shapes)
    {
        EXPECT_EQ(shape.height % 64, 0);
        EXPECT_EQ(shape.width % 64, 0);
        EXPECT_GE(shape.height, 64);
        EXPECT_GE(shape.width, 64);
    }
}

/** @brief Tests that one image makes one batch whose long side holds the image size. */
TEST(Small_ImagingRectangle, OneWideImageGivesOneWideShape)
{
    const RectangularBatches batches =
        require_value(plan_rectangular_batches({ImageSize{480, 640}}, make_settings(32, 640, 0.5)));

    // ceil(0.75 * 640 / 32 + 0.5) * 32 = 16 * 32 and ceil(20.5) * 32 = 21 * 32.
    EXPECT_EQ(batches.order, (std::vector<std::size_t>{0}));
    EXPECT_EQ(batches.shapes, (std::vector<ImageSize>{ImageSize{512, 672}}));
}

/** @brief Tests that the settings and the sizes are refused outside their ranges. */
TEST(Small_ImagingRectangle, RejectsSettingsAndSizesOutsideTheirRanges)
{
    const std::vector<ImageSize> sizes = make_sizes();
    RectangleSettings no_batch;
    no_batch.batch_size = 0;
    RectangleSettings no_stride;
    no_stride.stride = 0;
    RectangleSettings no_size;
    no_size.image_size = 0;
    RectangleSettings negative_pad;
    negative_pad.pad = -0.5;
    RectangleSettings undefined_pad;
    undefined_pad.pad = std::numeric_limits<double>::quiet_NaN();
    std::vector<ImageSize> flat = sizes;
    flat[2] = ImageSize{0, 500};

    SA_EXPECT_FAILURE(validate_rectangle(no_batch), StatusCode::InvalidDescription,
                      "validate_rectangle", "batch_size of 0");
    SA_EXPECT_FAILURE(plan_rectangular_batches(sizes, no_batch),
                      StatusCode::InvalidDescription, "validate_rectangle", "batch_size");
    SA_EXPECT_FAILURE(validate_rectangle(no_stride), StatusCode::InvalidDescription,
                      "validate_rectangle", "stride of 0");
    SA_EXPECT_FAILURE(validate_rectangle(no_size), StatusCode::InvalidDescription,
                      "validate_rectangle", "image_size of 0");
    SA_EXPECT_FAILURE(validate_rectangle(negative_pad), StatusCode::InvalidDescription,
                      "validate_rectangle", "pad");
    SA_EXPECT_FAILURE(validate_rectangle(undefined_pad), StatusCode::InvalidDescription,
                      "validate_rectangle", "pad");
    SA_EXPECT_FAILURE(plan_rectangular_batches({}, RectangleSettings{}),
                      StatusCode::InvalidDescription, "plan_rectangular_batches", "no image");
    SA_EXPECT_FAILURE(plan_rectangular_batches(flat, RectangleSettings{}),
                      StatusCode::InvalidDescription, "plan_rectangular_batches", "image 2",
                      "0x500");
    SA_EXPECT_OK(validate_rectangle(RectangleSettings{}));
}

/** @brief Tests that a shape whose side would pass the image bound is refused. */
TEST(Small_ImagingRectangle, RejectsAShapeAboveTheImageBound)
{
    RectangleSettings settings = make_settings(1, K_MAX_IMAGE_SIDE, 0.5);

    SA_EXPECT_FAILURE(plan_rectangular_batches({ImageSize{100, 100}}, settings),
                      StatusCode::InvalidDescription, "plan_rectangular_batches", "batch side",
                      "16384");
}
