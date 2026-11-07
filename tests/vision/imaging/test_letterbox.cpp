/**************************************************************************/
/* test_letterbox.cpp                                                     */
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
 * @file test_letterbox.cpp
 * @brief Tests the letterbox plan, its padding of an image and its mapping of boxes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>
#include <SushiAI/vision/imaging/resize.hpp>
#include <support/imaging_images.hpp>

using namespace SushiAI;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Labels;
using Vision::Imaging::LetterboxPlan;
using Vision::Imaging::LetterboxSettings;
using Vision::Imaging::PixelBox;

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::Test::expect_same_image;
    using SushiAI::Test::make_flat_image;

    /** @brief Holds the padding level of the default settings, which the cases below expect. */
    constexpr std::uint8_t K_DEFAULT_PADDING = 114;

    /** @brief Returns an image whose byte at offset i is first + i modulo 256. */
    Image make_counted_image(ImageSize size, std::int32_t channels, std::uint8_t first)
    {
        Image image(size, channels);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(first + i);

        return image;
    }

    /** @brief Returns a destination of the given side held at 255, or an empty one for side 0. */
    Image make_destination(std::int32_t side)
    {
        if (side == 0)
            return Image();

        return make_flat_image(ImageSize{side, side}, 3, 255);
    }

    /** @brief Reports whether a pixel lies inside the resized image of a plan. */
    bool is_content(const LetterboxPlan& plan, std::int32_t y, std::int32_t x)
    {
        return y >= plan.top && y < plan.top + plan.resized.height && x >= plan.left &&
               x < plan.left + plan.resized.width;
    }

    /** @brief Returns the source copied inside the padding of a plan that does not resize. */
    Image make_padded_copy(const Image& source, const LetterboxPlan& plan)
    {
        Image padded = make_flat_image(plan.target, source.channels(), plan.padding_value);
        for (std::int32_t y = 0; y < source.height(); ++y)
            for (std::int32_t x = 0; x < source.width() * source.channels(); ++x)
                padded.row(y + plan.top)[plan.left * source.channels() + x] = source.row(y)[x];

        return padded;
    }

    /** @brief Checks that every content byte is one level and every padding byte another. */
    void expect_two_levels(const Image& image, const LetterboxPlan& plan, std::uint8_t content,
                           std::uint8_t padding)
    {
        for (std::int32_t y = 0; y < image.height(); ++y)
            for (std::int32_t x = 0; x < image.width(); ++x)
                for (std::int32_t c = 0; c < image.channels(); ++c)
                {
                    const std::uint8_t expected = is_content(plan, y, x) ? content : padding;
                    ASSERT_EQ(image.row(y)[x * image.channels() + c], expected)
                        << "row " << y << ", column " << x << ", channel " << c;
                }
    }

    /**
     * @brief Checks that the content rectangle holds the bilinear resize of the source, that the
     *        rest holds the default padding, and that the content is not one flat level.
     */
    void expect_resized_content_between_padding(const Image& source, const LetterboxPlan& plan,
                                                const Image& destination)
    {
        Image expected;
        SA_EXPECT_OK(Vision::Imaging::resize_bilinear(source.view(), plan.resized, expected));
        ASSERT_EQ(destination.size(), plan.target);
        ASSERT_EQ(destination.channels(), 3);

        bool is_flat = true;
        std::size_t wrong = 0;
        for (std::int32_t y = 0; y < destination.height(); ++y)
            for (std::int32_t x = 0; x < destination.width(); ++x)
                for (std::int32_t c = 0; c < 3; ++c)
                {
                    std::uint8_t wanted = K_DEFAULT_PADDING;
                    if (is_content(plan, y, x))
                    {
                        wanted = expected.row(y - plan.top)[(x - plan.left) * 3 + c];
                        is_flat = is_flat && wanted == expected.data()[0];
                    }

                    if (destination.row(y)[x * 3 + c] != wanted)
                        ++wrong;
                }

        EXPECT_EQ(wrong, 0u) << "bytes differ from the resize placed between the padding";
        EXPECT_FALSE(is_flat) << "a flat content rectangle cannot tell a resize from a fill";
    }

    /** @brief Checks the four coordinates of a box against expected values, exactly. */
    void expect_box(const PixelBox& actual, const PixelBox& expected)
    {
        EXPECT_EQ(actual.x1, expected.x1);
        EXPECT_EQ(actual.y1, expected.y1);
        EXPECT_EQ(actual.x2, expected.x2);
        EXPECT_EQ(actual.y2, expected.y2);
    }

    /** @brief Checks the two coordinates of a point against expected values, exactly. */
    void expect_point(const Vision::Imaging::Point& actual, float x, float y)
    {
        EXPECT_EQ(actual.x, x);
        EXPECT_EQ(actual.y, y);
    }

    /** @brief Describes a source, a target and the substrings a refusal of them names. */
    struct SideRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the source size. */
        ImageSize source;

        /** @brief Holds the target size. */
        ImageSize target;

        /** @brief Holds the first substring the message names. */
        const char* first;

        /** @brief Holds the second substring the message names; empty when unused. */
        const char* second;
    };

    /** @brief Runs each side-range refusal of the plan. */
    class Small_ImagingLetterboxSideRange : public ::testing::TestWithParam<SideRow>
    {
    };

    /** @brief Describes a flat source, the padding level and the target it is fitted into. */
    struct FlatFitRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the source size. */
        ImageSize source;

        /** @brief Holds the channel count of the source. */
        std::int32_t channels;

        /** @brief Holds the level of every source byte. */
        std::uint8_t level;

        /** @brief Holds the padding level of the settings. */
        std::uint8_t padding_value;
    };

    /** @brief Runs each flat-content-and-padding case. */
    class Small_ImagingLetterboxFlatFit : public ::testing::TestWithParam<FlatFitRow>
    {
    };

    /** @brief Describes a counted source, its target and the destination it is drawn into. */
    struct ResizeRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the source size. */
        ImageSize source;

        /** @brief Holds the target size. */
        ImageSize target;

        /** @brief Holds the side of a destination held at 255 beforehand; 0 for an empty one. */
        std::int32_t destination_side;
    };

    /** @brief Runs each resized-content case. */
    class Small_ImagingLetterboxResizedContent : public ::testing::TestWithParam<ResizeRow>
    {
    };

    /** @brief Describes a destination image and the rows of it that the source view starts at. */
    struct AliasRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the height of the destination image. */
        std::int32_t image_height;

        /** @brief Holds the row of the image the source view starts at. */
        std::int32_t view_row;
    };

    /** @brief Runs each aliased-destination refusal; each starts a child process. */
    class Medium_ImagingLetterboxAliasing : public ::testing::TestWithParam<AliasRow>
    {
    };
} // namespace

TEST(Small_ImagingLetterbox, WideImageIsPaddedAboveAndBelowWithTheOddRowBelow)
{
    const LetterboxPlan plan = require_value(Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{50, 200}, ImageSize{100, 100}));

    EXPECT_DOUBLE_EQ(plan.ratio, 0.5);
    EXPECT_EQ(plan.resized, (ImageSize{25, 100}));
    EXPECT_EQ(plan.top, 37);
    EXPECT_EQ(plan.bottom, 38);
    EXPECT_EQ(plan.left, 0);
    EXPECT_EQ(plan.right, 0);
    EXPECT_EQ(plan.source, (ImageSize{50, 200}));
    EXPECT_EQ(plan.target, (ImageSize{100, 100}));
}

TEST(Small_ImagingLetterbox, TallImageIsPaddedLeftAndRightWithTheOddColumnRight)
{
    const LetterboxPlan plan = require_value(Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{200, 50}, ImageSize{100, 100}));

    EXPECT_DOUBLE_EQ(plan.ratio, 0.5);
    EXPECT_EQ(plan.resized, (ImageSize{100, 25}));
    EXPECT_EQ(plan.top, 0);
    EXPECT_EQ(plan.bottom, 0);
    EXPECT_EQ(plan.left, 37);
    EXPECT_EQ(plan.right, 38);
}

TEST(Small_ImagingLetterbox, ExactFitHasRatioOneAndNoPadding)
{
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{64, 64},
                                                      ImageSize{64, 64}));

    EXPECT_DOUBLE_EQ(plan.ratio, 1.0);
    EXPECT_EQ(plan.resized, (ImageSize{64, 64}));
    EXPECT_EQ(plan.top + plan.bottom + plan.left + plan.right, 0);
}

/** @brief Tests that a small image is only padded, not enlarged, while scale_up is off. */
TEST(Small_ImagingLetterbox, SmallImageIsOnlyPaddedWhileScaleUpIsOff)
{
    const LetterboxPlan padded =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{20, 40},
                                                      ImageSize{80, 80}));

    EXPECT_DOUBLE_EQ(padded.ratio, 1.0);
    EXPECT_EQ(padded.resized, (ImageSize{20, 40}));
    EXPECT_EQ(padded.top, 30);
    EXPECT_EQ(padded.bottom, 30);
    EXPECT_EQ(padded.left, 20);
    EXPECT_EQ(padded.right, 20);
}

/** @brief Tests that scale_up enlarges a small image to fit the target. */
TEST(Small_ImagingLetterbox, ScaleUpEnlargesASmallImage)
{
    LetterboxSettings settings;
    settings.scale_up = true;

    const LetterboxPlan enlarged =
        require_value(Vision::Imaging::plan_letterbox(settings, ImageSize{20, 40},
                                                      ImageSize{80, 80}));

    EXPECT_DOUBLE_EQ(enlarged.ratio, 2.0);
    EXPECT_EQ(enlarged.resized, (ImageSize{40, 80}));
    EXPECT_EQ(enlarged.top, 20);
    EXPECT_EQ(enlarged.bottom, 20);
    EXPECT_EQ(enlarged.left, 0);
    EXPECT_EQ(enlarged.right, 0);
}

TEST(Small_ImagingLetterbox, ResizedSideRoundsAHalfToTheEvenInteger)
{
    // 5 * 0.5 is 2.5 and 7 * 0.5 is 3.5; half to even gives 2 and 4.
    const LetterboxPlan down =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{5, 8},
                                                      ImageSize{4, 4}));
    const LetterboxPlan up =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{7, 8},
                                                      ImageSize{4, 4}));

    EXPECT_EQ(down.resized, (ImageSize{2, 4}));
    EXPECT_EQ(down.top, 1);
    EXPECT_EQ(down.bottom, 1);
    EXPECT_EQ(up.resized, (ImageSize{4, 4}));
    EXPECT_EQ(up.top, 0);
    EXPECT_EQ(up.bottom, 0);
}

TEST(Small_ImagingLetterbox, RectangularStrideTargetPadsOnlyTheShortAxis)
{
    // 512 is the stride-32 batch height of a 480x640 image: ceil(480 / 32 + 0.5) * 32.
    const LetterboxPlan plan = require_value(Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{480, 640}, ImageSize{512, 640}));

    EXPECT_DOUBLE_EQ(plan.ratio, 1.0);
    EXPECT_EQ(plan.resized, (ImageSize{480, 640}));
    EXPECT_EQ(plan.top, 16);
    EXPECT_EQ(plan.bottom, 16);
    EXPECT_EQ(plan.left, 0);
    EXPECT_EQ(plan.right, 0);
}

TEST(Small_ImagingLetterbox, PlanCopiesThePaddingValueOfTheSettings)
{
    LetterboxSettings settings;
    EXPECT_EQ(
        require_value(Vision::Imaging::plan_letterbox(settings, ImageSize{8, 8},
                                                      ImageSize{8, 8})).padding_value,
        settings.padding_value);

    settings.padding_value = 7;
    EXPECT_EQ(
        require_value(Vision::Imaging::plan_letterbox(settings, ImageSize{8, 8},
                                                      ImageSize{8, 8})).padding_value,
        7);
}

/** @brief Tests that the plan refuses a side that is not positive or is above the image bound. */
TEST_P(Small_ImagingLetterboxSideRange, PlanRejectsASideOutsideTheImageRange)
{
    const SideRow& row = GetParam();

    SA_EXPECT_FAILURE(Vision::Imaging::plan_letterbox(LetterboxSettings{}, row.source, row.target),
                      StatusCode::InvalidDescription, "plan_letterbox", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLetterboxSideRange,
    ::testing::Values(
        SideRow{"SourceSideOfZero", ImageSize{0, 8}, ImageSize{8, 8}, "source size of 0x8", ""},
        SideRow{"TargetSideBelowZero", ImageSize{8, 8}, ImageSize{8, -1}, "target size of 8x-1",
                ""},
        SideRow{"SourceSideAboveTheBound", ImageSize{16385, 8}, ImageSize{8, 8},
                "source size of 16385x8", "16384"},
        SideRow{"TargetSideAboveTheBound", ImageSize{8, 8}, ImageSize{8, 16385},
                "target size of 8x16385", "16384"}),
    [](const ::testing::TestParamInfo<SideRow>& info)
    {
        return std::string(info.param.name);
    });

TEST(Small_ImagingLetterbox, PlanRejectsAResizedSideOfZeroPixels)
{
    // 1 * (8 / 1000) is 0.008, which rounds to a height of zero.
    SA_EXPECT_FAILURE(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{1, 1000},
                                                      ImageSize{8, 8}),
                      StatusCode::InvalidDescription, "plan_letterbox", "zero pixels");
}

TEST(Small_ImagingLetterbox, ImageWithoutResizeKeepsItsPixelsInsideThePadding)
{
    const Image source = make_counted_image(ImageSize{2, 4}, 3, 1);
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(),
                                                      ImageSize{4, 6}));
    ASSERT_EQ(plan.top, 1);
    ASSERT_EQ(plan.left, 1);

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    expect_same_image(destination, make_padded_copy(source, plan));
}

TEST(Small_ImagingLetterbox, ExactFitCopiesEveryPixel)
{
    const Image source = make_counted_image(ImageSize{6, 6}, 3, 9);
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(),
                                                      source.size()));

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    expect_same_image(destination, source);
}

/** @brief Tests that a flat source is resized to one level and the rest takes the padding level. */
TEST_P(Small_ImagingLetterboxFlatFit, FlatImageIsResizedThenPaddedWithTheFillValue)
{
    // A flat image stays flat under bilinear weights that sum to one.
    const FlatFitRow& row = GetParam();
    LetterboxSettings settings;
    settings.padding_value = row.padding_value;
    const Image source = make_flat_image(row.source, row.channels, row.level);
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(settings, source.size(),
                                                      ImageSize{100, 100}));

    Image destination;
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{100, 100}));
    ASSERT_EQ(destination.channels(), row.channels);
    expect_two_levels(destination, plan, row.level, row.padding_value);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLetterboxFlatFit,
    ::testing::Values(FlatFitRow{"WideImageWithTheDefaultFill", ImageSize{50, 200}, 3, 200, 114},
                      FlatFitRow{"TallImageWithACustomFill", ImageSize{200, 50}, 1, 33, 7}),
    [](const ::testing::TestParamInfo<FlatFitRow>& info)
    {
        return std::string(info.param.name);
    });

/** @brief Tests that the content rectangle equals the bilinear resize and only it is written. */
TEST_P(Small_ImagingLetterboxResizedContent, ResizedContentEqualsTheBilinearResizeOfTheSource)
{
    const ResizeRow& row = GetParam();
    const Image source = make_counted_image(row.source, 3, 0);
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(),
                                                      row.target));

    Image destination = make_destination(row.destination_side);
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    expect_resized_content_between_padding(source, plan, destination);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLetterboxResizedContent,
    ::testing::Values(ResizeRow{"WideSourceIntoAnEmptyDestination", ImageSize{50, 200},
                                ImageSize{100, 100}, 0},
                      ResizeRow{"TallSourceIntoALargerDestination", ImageSize{200, 50},
                                ImageSize{100, 101}, 120}),
    [](const ::testing::TestParamInfo<ResizeRow>& info)
    {
        return std::string(info.param.name);
    });

TEST(Small_ImagingLetterbox, DestinationIsFullyRewrittenWhenItIsReused)
{
    const Image source = make_flat_image(ImageSize{2, 4}, 3, 50);
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(),
                                                      ImageSize{4, 6}));

    Image destination = make_flat_image(ImageSize{9, 9}, 3, 255);
    Vision::Imaging::apply_letterbox(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), (ImageSize{4, 6}));
    expect_two_levels(destination, plan, 50, 114);
}

/** @brief Tests that apply_letterbox refuses a source without pixels. */
TEST(Medium_ImagingLetterbox, SourceWithoutPixelsIsRejected)
{
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{2, 4},
                                                      ImageSize{4, 6}));

    Image destination;
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_letterbox(Vision::Imaging::ImageView{},
                                                                       plan,
                                                     destination)),
                    "apply_letterbox", "no pixels");
}

/** @brief Tests that apply_letterbox refuses a source view that lies inside the destination. */
TEST_P(Medium_ImagingLetterboxAliasing, ViewInsideTheDestinationIsRejected)
{
    const AliasRow& row = GetParam();
    Image image = make_flat_image(ImageSize{row.image_height, 4}, 3, 50);
    const Vision::Imaging::ImageView view(image.row(row.view_row), ImageSize{2, 4}, 3);
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{2, 4},
                                                      ImageSize{4, 6}));

    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_letterbox(view, plan, image)),
                    "apply_letterbox",
                    "destination holds the source's pixels");
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_ImagingLetterboxAliasing,
                         ::testing::Values(AliasRow{"DestinationThatIsTheSource", 2, 0},
                                           AliasRow{"ViewOfTheLowerRows", 3, 1}),
                         [](const ::testing::TestParamInfo<AliasRow>& info)
                         {
                             return std::string(info.param.name);
                         });

/** @brief Tests that apply_letterbox refuses a source of another size than the plan. */
TEST(Medium_ImagingLetterbox, ImageOfAnotherSizeThanThePlanIsRejected)
{
    const Image source = make_flat_image(ImageSize{2, 4}, 3, 50);
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{4, 4},
                                                      ImageSize{8, 8}));

    Image destination;
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_letterbox(source.view(), plan,
                                                                       destination)),
                    "apply_letterbox", "plan was made for");
}

/** @brief Tests that apply_letterbox refuses a plan whose paddings do not tile the target. */
TEST(Medium_ImagingLetterbox, PlanWhosePaddingsDoNotTileTheTargetIsRejected)
{
    const Image source = make_flat_image(ImageSize{2, 4}, 3, 50);
    LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, source.size(),
                                                      ImageSize{4, 6}));
    plan.right += 1;

    Image destination;
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_letterbox(source.view(), plan,
                                                                       destination)),
                    "apply_letterbox", "tile");
}

TEST(Small_ImagingLetterbox, BoxesMoveByTheRatioThenTheLeftAndTopPadding)
{
    const LetterboxPlan tall = require_value(Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{200, 50}, ImageSize{100, 100}));
    const LetterboxPlan wide = require_value(Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{50, 200}, ImageSize{100, 100}));

    const Labels labels({3, 7}, {PixelBox{10.0f, 20.0f, 30.0f, 40.0f},
                                 PixelBox{0.0f, 0.0f, 50.0f, 200.0f}});

    // x * 0.5 + 37 and y * 0.5 + 0; every value is a dyadic fraction, so the result is exact.
    const Labels in_tall = Vision::Imaging::apply_letterbox(labels, tall);
    ASSERT_EQ(in_tall.size(), 2u);
    EXPECT_EQ(in_tall.classes(), labels.classes());
    expect_box(in_tall.boxes()[0], PixelBox{42.0f, 10.0f, 52.0f, 20.0f});
    expect_box(in_tall.boxes()[1], PixelBox{37.0f, 0.0f, 62.0f, 100.0f});

    // x * 0.5 + 0 and y * 0.5 + 37.
    const Labels in_wide = Vision::Imaging::apply_letterbox(labels, wide);
    ASSERT_EQ(in_wide.size(), 2u);
    expect_box(in_wide.boxes()[0], PixelBox{5.0f, 47.0f, 15.0f, 57.0f});
}

TEST(Small_ImagingLetterbox, BoxesOutsideTheImageAreNeitherClippedNorRemoved)
{
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{20, 40},
                                                      ImageSize{80, 80}));

    const Labels labels({1, 2}, {PixelBox{-8.0f, -4.0f, 100.0f, 90.0f},
                                 PixelBox{5.0f, 5.0f, 5.0f, 5.0f}});

    // Ratio 1, left 20, top 30.
    const Labels moved = Vision::Imaging::apply_letterbox(labels, plan);
    ASSERT_EQ(moved.size(), 2u);
    EXPECT_EQ(moved.classes(), labels.classes());
    expect_box(moved.boxes()[0], PixelBox{12.0f, 26.0f, 120.0f, 120.0f});
    expect_box(moved.boxes()[1], PixelBox{25.0f, 35.0f, 25.0f, 35.0f});
}

TEST(Small_ImagingLetterbox, ExactFitLeavesBoxesWhereTheyWere)
{
    const LetterboxPlan plan =
        require_value(Vision::Imaging::plan_letterbox(LetterboxSettings{}, ImageSize{64, 64},
                                                      ImageSize{64, 64}));

    const Labels labels({0}, {PixelBox{1.5f, 2.5f, 60.25f, 63.0f}});

    const Labels moved = Vision::Imaging::apply_letterbox(labels, plan);
    ASSERT_EQ(moved.size(), 1u);
    expect_box(moved.boxes()[0], PixelBox{1.5f, 2.5f, 60.25f, 63.0f});
}

/** @brief Tests that a polygon moves by the ratio, then by the left and top padding. */
TEST(Small_ImagingLetterbox, PolygonMovesByTheRatioThenTheLeftAndTopPadding)
{
    const LetterboxPlan tall = require_value(Vision::Imaging::plan_letterbox(
        LetterboxSettings{}, ImageSize{200, 50}, ImageSize{100, 100}));
    ASSERT_EQ(tall.left, 37);
    ASSERT_EQ(tall.top, 0);
    Labels labels;
    labels.add(3, PixelBox{10.0f, 20.0f, 30.0f, 40.0f},
               Vision::Imaging::Polygon{Vision::Imaging::Point{10.0f, 20.0f},
               Vision::Imaging::Point{30.0f, 40.0f}, Vision::Imaging::Point{-4.0f, 210.0f}});
    labels.add(7, PixelBox{0.0f, 0.0f, 50.0f, 200.0f});

    const Labels moved = Vision::Imaging::apply_letterbox(labels, tall);

    ASSERT_EQ(moved.size(), 2u);
    ASSERT_EQ(moved.polygons()[0].size(), 3u);
    expect_point(moved.polygons()[0][0], 42.0f, 10.0f);
    expect_point(moved.polygons()[0][1], 52.0f, 20.0f);
    expect_point(moved.polygons()[0][2], 35.0f, 105.0f);
    EXPECT_TRUE(moved.polygons()[1].empty());
}
