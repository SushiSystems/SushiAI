/**************************************************************************/
/* test_flip.cpp                                                          */
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
 * @file test_flip.cpp
 * @brief Tests the draw of the horizontal flip and its mirroring of pixels and labels.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/flip.hpp>

using namespace SushiAI;
using Vision::Imaging::FlipPlan;
using Vision::Imaging::FlipSettings;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::Labels;
using Vision::Imaging::PixelBox;
using Vision::Imaging::Random;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Counts the plans a statistical test draws, all from one generator. */
    constexpr std::uint64_t K_SAMPLE_COUNT = 1024;

    /** @brief Holds the seed of the generator a statistical test draws from. */
    constexpr std::uint64_t K_STATISTICAL_SEED = 20261007;

    /** @brief Holds the chance that a correct run of plans still misses its bound. */
    constexpr double K_FALSE_FAILURE_RATE = 1e-9;

    /**
     * @brief Returns the distance from 0.5 that a flip frequency stays within over K_SAMPLE_COUNT.
     *
     * Hoeffding's inequality gives P(|frequency - p| >= t) <= 2 exp(-2 n t^2), so t is
     * sqrt(ln(2 / rate) / (2 n)) for n draws and a false-failure rate.
     */
    double frequency_tolerance()
    {
        return std::sqrt(std::log(2.0 / K_FALSE_FAILURE_RATE) /
                         (2.0 * static_cast<double>(K_SAMPLE_COUNT)));
    }

    /** @brief Returns an image whose byte at offset i is i modulo 256. */
    Image make_counted_image(ImageSize size, std::int32_t channels)
    {
        Image image(size, channels);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(i);

        return image;
    }

    /** @brief Checks that two images have the same size, channels and bytes. */
    void expect_same_image(const Image& actual, const Image& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        ASSERT_EQ(actual.channels(), expected.channels());
        for (std::size_t i = 0; i < expected.byte_count(); ++i)
            ASSERT_EQ(actual.data()[i], expected.data()[i]) << "byte " << i;
    }

    /** @brief Checks that an image holds exactly the listed bytes, in order. */
    void expect_bytes(const Image& image, std::initializer_list<std::uint8_t> expected)
    {
        ASSERT_EQ(image.byte_count(), expected.size());
        std::size_t offset = 0;
        for (const std::uint8_t value : expected)
        {
            EXPECT_EQ(image.data()[offset], value) << "byte " << offset;
            ++offset;
        }
    }

    /** @brief Checks that two flips of a source with a flipped plan give the source back. */
    void expect_two_flips_restore(const Image& source)
    {
        const FlipPlan plan{source.size(), true};
        Image once;
        Image twice;

        Vision::Imaging::apply_flip(source.view(), plan, once);
        Vision::Imaging::apply_flip(once.view(), plan, twice);

        expect_same_image(twice, source);
    }

    /** @brief Checks the four coordinates of a box against expected values, exactly. */
    void expect_box(const PixelBox& actual, const PixelBox& expected)
    {
        EXPECT_EQ(actual.x1, expected.x1);
        EXPECT_EQ(actual.y1, expected.y1);
        EXPECT_EQ(actual.x2, expected.x2);
        EXPECT_EQ(actual.y2, expected.y2);
    }

    /** @brief Describes one probability, the seed of its draw and the decision it must give. */
    struct DecisionRow
    {
        float probability;
        std::uint64_t seed;
        bool expected_flip;
    };
} // namespace

/** @brief Tests that the plan records the source size it was made for. */
TEST(Small_ImagingFlip, PlanRecordsTheSourceSize)
{
    Random random(42);

    const FlipPlan plan =
        require_value(Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{480, 640}, random));

    EXPECT_EQ(plan.source, (ImageSize{480, 640}));
}

/** @brief Tests that the plan makes exactly one Bernoulli draw and records its outcome. */
TEST(Small_ImagingFlip, PlanMakesExactlyOneDraw)
{
    Random random(42);
    Random replay(42);

    const FlipPlan plan =
        require_value(Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{480, 640}, random));

    EXPECT_EQ(plan.is_flipped, replay.bernoulli(FlipSettings{}.probability));
    EXPECT_EQ(random, replay);
}

/** @brief Rows of a flip probability, a seed and the decision it must give. */
class Small_ImagingFlipDecision : public ::testing::TestWithParam<DecisionRow>
{
};

/** @brief Tests that probability 0 never flips and 1 always flips, each with one draw. */
TEST_P(Small_ImagingFlipDecision, ProbabilityZeroNeverFlipsAndOneAlwaysFlips)
{
    const DecisionRow& row = GetParam();
    FlipSettings settings;
    settings.probability = row.probability;
    Random random(row.seed);
    Random replay(row.seed);

    const FlipPlan plan =
        require_value(Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random));

    EXPECT_EQ(plan.is_flipped, row.expected_flip);
    static_cast<void>(replay.bernoulli(row.probability));
    EXPECT_EQ(random, replay);
}

INSTANTIATE_TEST_SUITE_P(Decisions, Small_ImagingFlipDecision,
                         ::testing::Values(DecisionRow{0.0f, 0, false}, DecisionRow{0.0f, 7, false},
                                           DecisionRow{0.0f, 20261007, false},
                                           DecisionRow{1.0f, 0, true}, DecisionRow{1.0f, 7, true},
                                           DecisionRow{1.0f, 20261007, true}));

/** @brief Tests that the default probability flips about half of a fixed-seed run of plans. */
TEST(Small_ImagingFlip, DefaultProbabilityFlipsAboutHalfOfTheDraws)
{
    Random random(K_STATISTICAL_SEED);
    std::uint64_t flipped = 0;

    for (std::uint64_t sample = 0; sample < K_SAMPLE_COUNT; ++sample)
    {
        if (require_value(Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{8, 8},
                                                     random)).is_flipped)
            ++flipped;
    }

    const double frequency = static_cast<double>(flipped) / static_cast<double>(K_SAMPLE_COUNT);
    EXPECT_NEAR(frequency, 0.5, frequency_tolerance());
}

/** @brief Tests that a probability outside zero to one is refused by the plan and by validate. */
TEST(Small_ImagingFlip, PlanRejectsAProbabilityOutsideZeroToOne)
{
    Random random(1);
    FlipSettings settings;

    settings.probability = 1.5f;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random),
                      StatusCode::InvalidDescription, "validate_flip", "probability", "1.5");
    SA_EXPECT_FAILURE(Vision::Imaging::validate_flip(settings), StatusCode::InvalidDescription,
                      "validate_flip", "probability");

    settings.probability = -0.1f;
    SA_EXPECT_FAILURE(Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random),
                      StatusCode::InvalidDescription, "validate_flip", "probability");
}

/** @brief Tests that a plan refused for its probability makes no draw. */
TEST(Small_ImagingFlip, PlanRejectedForItsProbabilityMakesNoDraw)
{
    Random random(1);
    FlipSettings settings;

    settings.probability = 1.5f;
    static_cast<void>(Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random));
    settings.probability = -0.1f;
    static_cast<void>(Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random));

    EXPECT_EQ(random, Random(1));
}

/** @brief Tests that a NaN probability is refused before any draw. */
TEST(Small_ImagingFlip, PlanRejectsAProbabilityThatIsNotANumberBeforeAnyDraw)
{
    Random random(3);
    FlipSettings settings;
    settings.probability = std::numeric_limits<float>::quiet_NaN();

    SA_EXPECT_FAILURE(Vision::Imaging::plan_flip(settings, ImageSize{8, 8}, random),
                      StatusCode::InvalidDescription, "validate_flip", "probability");
    EXPECT_EQ(random, Random(3));
}

/** @brief Tests that a source side that is not positive is refused before any draw. */
TEST(Small_ImagingFlip, PlanRejectsASideThatIsNotPositiveBeforeAnyDraw)
{
    Random random(5);

    SA_EXPECT_FAILURE(Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{0, 8}, random),
                      StatusCode::InvalidDescription, "plan_flip", "source size", "0x8");
    SA_EXPECT_FAILURE(Vision::Imaging::plan_flip(FlipSettings{}, ImageSize{8, -2}, random),
                      StatusCode::InvalidDescription, "plan_flip", "source size", "8x-2");
    EXPECT_EQ(random, Random(5));
}

/** @brief Tests that applying a flip to a source without pixels stops the process. */
TEST(Medium_ImagingFlip, SourceWithoutPixelsIsRejected)
{
    const FlipPlan plan{ImageSize{2, 3}, true};

    Image destination;
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_flip(Vision::Imaging::ImageView{},
                                                                  plan, destination)),
                    "apply_flip", "no pixels");
}

/** @brief Tests that a source held by the destination, whole or as a view of rows, is refused. */
TEST(Medium_ImagingFlip, SourceHeldByTheDestinationIsRejected)
{
    Image whole = make_counted_image(ImageSize{2, 3}, 3);
    const FlipPlan whole_plan{ImageSize{2, 3}, true};
    Image rows = make_counted_image(ImageSize{4, 3}, 3);
    const Vision::Imaging::ImageView lower_rows(rows.row(1), ImageSize{3, 3}, 3);
    const FlipPlan rows_plan{ImageSize{3, 3}, true};

    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_flip(whole.view(), whole_plan, whole)),
                    "apply_flip", "destination holds the source's pixels");
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_flip(lower_rows, rows_plan, rows)),
                    "apply_flip", "destination holds the source's pixels");
}

/** @brief Tests that a flipped image holds each row's pixels in reverse column order. */
TEST(Small_ImagingFlip, FlippedImageHoldsEachRowInReverseColumnOrder)
{
    const Image source = make_counted_image(ImageSize{2, 3}, 3);
    const FlipPlan plan{ImageSize{2, 3}, true};

    Image destination;
    Vision::Imaging::apply_flip(source.view(), plan, destination);

    ASSERT_EQ(destination.size(), source.size());
    ASSERT_EQ(destination.channels(), 3);
    expect_bytes(destination, {6, 7, 8, 3, 4, 5, 0, 1, 2, 15, 16, 17, 12, 13, 14, 9, 10, 11});
}

/** @brief Tests that an unflipped plan copies the image unchanged. */
TEST(Small_ImagingFlip, UnflippedImageIsCopiedUnchanged)
{
    const Image source = make_counted_image(ImageSize{3, 5}, 3);
    const FlipPlan plan{ImageSize{3, 5}, false};

    Image destination;
    Vision::Imaging::apply_flip(source.view(), plan, destination);

    expect_same_image(destination, source);
}

/** @brief Tests that two flips restore a colour image and a grey image. */
TEST(Small_ImagingFlip, FlippingAnImageTwiceRestoresIt)
{
    expect_two_flips_restore(make_counted_image(ImageSize{4, 7}, 3));
    expect_two_flips_restore(make_counted_image(ImageSize{3, 4}, 1));
}

/** @brief Tests that a source of another size than the plan's is refused. */
TEST(Medium_ImagingFlip, ImageOfAnotherSizeThanThePlanIsRejected)
{
    const Image source = make_counted_image(ImageSize{2, 3}, 3);
    const FlipPlan wider{ImageSize{2, 4}, true};
    const FlipPlan taller{ImageSize{5, 3}, true};

    Image destination;
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_flip(source.view(), wider,
                                                                  destination)), "apply_flip",
                    "plan was made for a 2x4 source", "2x3");
    SA_EXPECT_FATAL(static_cast<void>(Vision::Imaging::apply_flip(source.view(), taller,
                                                                  destination)), "apply_flip",
                    "plan was made for a 5x3 source", "2x3");
}

/** @brief Tests that flipped boxes mirror about the width and keep their order and classes. */
TEST(Small_ImagingFlip, FlippedBoxesMirrorAboutTheWidthAndKeepTheirOrder)
{
    const FlipPlan plan{ImageSize{640, 640}, true};

    const Labels labels({4, 9}, {PixelBox{100.0f, 50.0f, 300.0f, 200.0f},
                                 PixelBox{0.0f, 10.0f, 640.0f, 20.0f}});

    const Labels mirrored = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(mirrored.size(), 2u);
    EXPECT_EQ(mirrored.classes(), labels.classes());
    expect_box(mirrored.boxes()[0], PixelBox{340.0f, 50.0f, 540.0f, 200.0f});
    expect_box(mirrored.boxes()[1], PixelBox{0.0f, 10.0f, 640.0f, 20.0f});
    EXPECT_LE(mirrored.boxes()[0].x1, mirrored.boxes()[0].x2);
}

/** @brief Tests that an unflipped plan returns the boxes unchanged. */
TEST(Small_ImagingFlip, UnflippedBoxesAreReturnedUnchanged)
{
    const FlipPlan plan{ImageSize{640, 640}, false};

    const Labels labels({4}, {PixelBox{100.0f, 50.0f, 300.0f, 200.0f}});

    const Labels same = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(same.size(), 1u);
    EXPECT_EQ(same.classes(), labels.classes());
    expect_box(same.boxes()[0], labels.boxes()[0]);
}

/** @brief Tests that two flips of the boxes restore them. */
TEST(Small_ImagingFlip, FlippingBoxesTwiceRestoresThem)
{
    const FlipPlan plan{ImageSize{640, 640}, true};

    const Labels labels({1, 2}, {PixelBox{12.5f, 3.0f, 99.25f, 40.0f},
                                 PixelBox{600.0f, 0.0f, 640.0f, 640.0f}});

    const Labels twice =
        Vision::Imaging::apply_flip(Vision::Imaging::apply_flip(labels, plan), plan);

    ASSERT_EQ(twice.size(), 2u);
    EXPECT_EQ(twice.classes(), labels.classes());
    expect_box(twice.boxes()[0], labels.boxes()[0]);
    expect_box(twice.boxes()[1], labels.boxes()[1]);
}

/** @brief Tests that a flipped polygon has every x mirrored about the width and every y kept. */
TEST(Small_ImagingFlip, FlippedPolygonMirrorsEveryVertexAboutTheWidth)
{
    const FlipPlan plan{ImageSize{480, 640}, true};
    Labels labels;
    labels.add(2, PixelBox{100.0f, 50.0f, 300.0f, 200.0f},
               Vision::Imaging::Polygon{Vision::Imaging::Point{100.0f, 50.0f},
               Vision::Imaging::Point{300.0f, 60.0f}, Vision::Imaging::Point{150.5f, 200.0f}});
    labels.add(3, PixelBox{0.0f, 0.0f, 10.0f, 10.0f});

    const Labels mirrored = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(mirrored.size(), 2u);
    ASSERT_EQ(mirrored.polygons()[0].size(), 3u);
    EXPECT_EQ(mirrored.polygons()[0][0].x, 540.0f);
    EXPECT_EQ(mirrored.polygons()[0][0].y, 50.0f);
    EXPECT_EQ(mirrored.polygons()[0][1].x, 340.0f);
    EXPECT_EQ(mirrored.polygons()[0][1].y, 60.0f);
    EXPECT_EQ(mirrored.polygons()[0][2].x, 489.5f);
    EXPECT_EQ(mirrored.polygons()[0][2].y, 200.0f);
    EXPECT_TRUE(mirrored.polygons()[1].empty());
}

/** @brief Tests that an unflipped plan returns the polygons as they were. */
TEST(Small_ImagingFlip, UnflippedPolygonIsReturnedUnchanged)
{
    const FlipPlan plan{ImageSize{480, 640}, false};
    Labels labels;
    labels.add(2, PixelBox{100.0f, 50.0f, 300.0f, 200.0f},
               Vision::Imaging::Polygon{Vision::Imaging::Point{100.0f, 50.0f},
               Vision::Imaging::Point{300.0f, 60.0f}});

    const Labels same = Vision::Imaging::apply_flip(labels, plan);

    ASSERT_EQ(same.polygons()[0].size(), 2u);
    EXPECT_EQ(same.polygons()[0][1].x, 300.0f);
    EXPECT_EQ(same.polygons()[0][1].y, 60.0f);
}

/** @brief Tests that validate_flip accepts both ends of the probability range. */
TEST(Small_ImagingFlip, ValidateAcceptsZeroAndOne)
{
    FlipSettings never;
    never.probability = 0.0f;
    FlipSettings always;
    always.probability = 1.0f;

    SA_EXPECT_OK(Vision::Imaging::validate_flip(never));
    SA_EXPECT_OK(Vision::Imaging::validate_flip(always));
    SA_EXPECT_OK(Vision::Imaging::validate_flip(FlipSettings{}));
}
