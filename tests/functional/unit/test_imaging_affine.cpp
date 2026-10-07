/**************************************************************************/
/* test_imaging_affine.cpp                                                */
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
 * @file test_imaging_affine.cpp
 * @brief Tests the random affine plan, its image warp and its warp of boxes and polygons.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/imaging/affine.hpp>

#include "../../common/test_throws.hpp"

using namespace SushiAI::Imaging;

namespace
{
    /** @brief Holds the absolute tolerance, in pixels, of a box edge or a matrix term. */
    constexpr float K_PIXEL_TOLERANCE = 1e-3f;

    /** @brief Holds a 3 by 3 float matrix, row-major, as the reference builds its factors. */
    using Matrix3 = std::array<float, 9>;

    /** @brief Returns the product a b of two 3 by 3 matrices, each term summed left to right. */
    Matrix3 multiply(const Matrix3& a, const Matrix3& b)
    {
        Matrix3 product{};

        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 3; ++column)
                product[row * 3 + column] = a[row * 3] * b[column] +
                                            a[row * 3 + 1] * b[3 + column] +
                                            a[row * 3 + 2] * b[6 + column];

        return product;
    }

    /** @brief Returns an image whose byte at row y, column x, channel c is a known pattern. */
    Image make_pattern_image(ImageSize size, std::int32_t channels)
    {
        Image image(size, channels);

        for (std::int32_t y = 0; y < size.height; ++y)
            for (std::int32_t x = 0; x < size.width; ++x)
                for (std::int32_t c = 0; c < channels; ++c)
                    image.row(y)[x * channels + c] =
                        static_cast<std::uint8_t>((y * 31 + x * 7 + c * 53 + 1) % 256);

        return image;
    }

    /** @brief Returns a one-channel image holding the given bytes in row-major order. */
    Image make_gray_image(ImageSize size, const std::vector<std::uint8_t>& bytes)
    {
        Image image(size, 1);
        std::copy(bytes.begin(), bytes.end(), image.data());
        return image;
    }

    /** @brief Returns a plan with the given scale and matrix translation and no draw. */
    AffinePlan make_plan(ImageSize source, ImageSize output, float scale, float offset_x,
                         float offset_y)
    {
        AffinePlan plan;
        plan.source = source;
        plan.output = output;
        plan.scale = scale;
        plan.translate_x = offset_x + scale * static_cast<float>(source.width) / 2.0f;
        plan.translate_y = offset_y + scale * static_cast<float>(source.height) / 2.0f;
        plan.matrix = {scale, 0.0f, offset_x, 0.0f, scale, offset_y};
        return plan;
    }

    /** @brief Returns a label set whose classes count up from zero, one per box. */
    Labels make_labels(const std::vector<PixelBox>& boxes)
    {
        Labels labels;

        for (std::size_t i = 0; i < boxes.size(); ++i)
            labels.add(static_cast<std::int32_t>(i), boxes[i]);

        return labels;
    }

    /** @brief Checks that the four edges of two boxes agree within K_PIXEL_TOLERANCE. */
    void expect_box_near(const PixelBox& actual, const PixelBox& expected)
    {
        EXPECT_NEAR(actual.x1, expected.x1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.y1, expected.y1, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.x2, expected.x2, K_PIXEL_TOLERANCE);
        EXPECT_NEAR(actual.y2, expected.y2, K_PIXEL_TOLERANCE);
    }

    /** @brief Returns the bytes of an image in row-major order. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }
} // namespace

TEST(Unit_ImagingAffine, ZeroRangesPutTheSourceCentreOnTheOutputCentre)
{
    AffineSettings settings;
    settings.scale = 0.0f;
    settings.translate = 0.0f;
    Random random(11);

    const AffinePlan plan = plan_affine(settings, ImageSize{1200, 1000}, ImageSize{480, 640},
                                        random);

    // By hand: scale 1, T = (320, 240), C = (-500, -600), so M = [1 0 -180; 0 1 -360].
    EXPECT_NEAR(plan.scale, 1.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(plan.translate_x, 320.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(plan.translate_y, 240.0f, K_PIXEL_TOLERANCE);

    const std::array<float, 6> expected{1.0f, 0.0f, -180.0f, 0.0f, 1.0f, -360.0f};

    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(plan.matrix[i], expected[i], K_PIXEL_TOLERANCE) << "matrix term " << i;
}

TEST(Unit_ImagingAffine, MatrixEqualsTranslationTimesScaleTimesCentre)
{
    const AffineSettings settings;
    const ImageSize source{1280, 960};
    const ImageSize output{640, 480};
    Random random(2026);
    Random replay = random;

    const AffinePlan plan = plan_affine(settings, source, output, random);

    const float scale = replay.uniform(0.5f, 1.5f);
    const float translate_x = replay.uniform(0.5f - 0.1f, 0.5f + 0.1f) * 480.0f;
    const float translate_y = replay.uniform(0.5f - 0.1f, 0.5f + 0.1f) * 640.0f;

    // At scale 1 the product is the same in any order, so the seed must give another scale.
    ASSERT_GT(std::abs(scale - 1.0f), 0.05f);

    const Matrix3 centre{1.0f, 0.0f, -480.0f, 0.0f, 1.0f, -640.0f, 0.0f, 0.0f, 1.0f};
    const Matrix3 scaling{scale, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 1.0f};
    const Matrix3 translation{1.0f, 0.0f, translate_x, 0.0f, 1.0f, translate_y,
                              0.0f, 0.0f, 1.0f};
    const Matrix3 expected = multiply(translation, multiply(scaling, centre));

    EXPECT_NEAR(plan.scale, scale, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(plan.translate_x, translate_x, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(plan.translate_y, translate_y, K_PIXEL_TOLERANCE);

    for (std::size_t i = 0; i < plan.matrix.size(); ++i)
        EXPECT_NEAR(plan.matrix[i], expected[i], K_PIXEL_TOLERANCE) << "matrix term " << i;
}

TEST(Unit_ImagingAffine, PlanConsumesExactlyThreeDraws)
{
    Random random(5);
    Random replay = random;

    const AffinePlan plan = plan_affine(AffineSettings{}, ImageSize{64, 64}, ImageSize{32, 32},
                                        random);
    static_cast<void>(plan);

    for (int draw = 0; draw < 3; ++draw)
        static_cast<void>(replay.next_u64());

    EXPECT_TRUE(random == replay);
}

TEST(Unit_ImagingAffine, PlanCarriesSizesBorderValueAndFilter)
{
    AffineSettings settings;
    settings.border_value = 37;
    settings.filter.min_side = 5.0f;
    settings.filter.min_area_ratio = 0.25f;
    Random random(3);

    const AffinePlan plan = plan_affine(settings, ImageSize{20, 30}, ImageSize{8, 12}, random);

    EXPECT_TRUE(plan.source == (ImageSize{20, 30}));
    EXPECT_TRUE(plan.output == (ImageSize{8, 12}));
    EXPECT_EQ(plan.border_value, 37);
    EXPECT_NEAR(plan.filter.min_side, 5.0f, 1e-6f);
    EXPECT_NEAR(plan.filter.min_area_ratio, 0.25f, 1e-6f);
}

TEST(Unit_ImagingAffine, DrawnParametersStayInRangeAndCentreOnTheirMeans)
{
    const AffineSettings settings;
    const ImageSize source{1280, 1280};
    const ImageSize output{640, 480};
    const float low_fraction = 0.5f - 0.1f;
    const float high_fraction = 0.5f + 0.1f;

    double scale_sum = 0.0;
    double x_sum = 0.0;
    double y_sum = 0.0;
    float scale_min = 2.0f;
    float scale_max = 0.0f;
    float x_min = 1.0f;
    float x_max = 0.0f;
    float y_min = 1.0f;
    float y_max = 0.0f;
    int equal_fractions = 0;
    int count = 0;

    for (std::uint64_t seed = 0; seed < 200; ++seed)
    {
        Random random(seed);

        for (int repeat = 0; repeat < 100; ++repeat, ++count)
        {
            const AffinePlan plan = plan_affine(settings, source, output, random);

            ASSERT_GE(plan.scale, 0.5f);
            ASSERT_LT(plan.scale, 1.5f);
            ASSERT_GE(plan.translate_x, low_fraction * 480.0f);
            ASSERT_LE(plan.translate_x, high_fraction * 480.0f);
            ASSERT_GE(plan.translate_y, low_fraction * 640.0f);
            ASSERT_LE(plan.translate_y, high_fraction * 640.0f);

            const float x_fraction = plan.translate_x / 480.0f;
            const float y_fraction = plan.translate_y / 640.0f;

            scale_sum += plan.scale;
            x_sum += x_fraction;
            y_sum += y_fraction;
            scale_min = std::min(scale_min, plan.scale);
            scale_max = std::max(scale_max, plan.scale);
            x_min = std::min(x_min, x_fraction);
            x_max = std::max(x_max, x_fraction);
            y_min = std::min(y_min, y_fraction);
            y_max = std::max(y_max, y_fraction);

            // Fractions closer than 1e-6 would mean one draw served both axes.
            if (std::abs(x_fraction - y_fraction) < 1e-6f)
                ++equal_fractions;
        }
    }

    // 20000 draws: the standard error is 0.0020 for the scale and 0.0004 for a fraction.
    EXPECT_NEAR(scale_sum / count, 1.0, 0.012);
    EXPECT_NEAR(x_sum / count, 0.5, 0.0025);
    EXPECT_NEAR(y_sum / count, 0.5, 0.0025);

    EXPECT_LT(scale_min, 0.51f);
    EXPECT_GT(scale_max, 1.49f);
    EXPECT_LT(x_min, 0.402f);
    EXPECT_GT(x_max, 0.598f);
    EXPECT_LT(y_min, 0.402f);
    EXPECT_GT(y_max, 0.598f);
    EXPECT_LT(equal_fractions, count / 100);
}

TEST(Unit_ImagingAffine, PlanRejectsSettingsAndSizesOutsideTheirRanges)
{
    const ImageSize size{16, 16};
    Random random(1);

    AffineSettings wide_scale;
    wide_scale.scale = 1.0f;
    SA_EXPECT_ERROR(static_cast<void>(plan_affine(wide_scale, size, size, random)),
                    "validate_affine", "scale");
    SA_EXPECT_ERROR(validate_affine(wide_scale), "validate_affine", "scale half-range of 1");

    AffineSettings wide_translate;
    wide_translate.translate = 0.6f;
    SA_EXPECT_ERROR(static_cast<void>(plan_affine(wide_translate, size, size, random)),
                    "validate_affine", "translate");
    SA_EXPECT_ERROR(validate_affine(wide_translate), "validate_affine", "translate half-range");

    AffineSettings negative_scale;
    negative_scale.scale = -0.1f;
    SA_EXPECT_ERROR(static_cast<void>(plan_affine(negative_scale, size, size, random)), "scale");

    AffineSettings unordered_scale;
    unordered_scale.scale = std::numeric_limits<float>::quiet_NaN();
    SA_EXPECT_ERROR(static_cast<void>(plan_affine(unordered_scale, size, size, random)),
                    "scale");

    AffineSettings negative_translate;
    negative_translate.translate = -0.1f;
    SA_EXPECT_ERROR(static_cast<void>(plan_affine(negative_translate, size, size, random)),
                    "translate");

    const AffineSettings settings;
    SA_EXPECT_ERROR(static_cast<void>(plan_affine(settings, ImageSize{0, 16}, size, random)),
                    "plan_affine", "source size of 0x16");
    SA_EXPECT_ERROR(static_cast<void>(plan_affine(settings, size, ImageSize{16, 0}, random)),
                    "plan_affine", "output size of 16x0");
    EXPECT_TRUE(random == Random(1)) << "a rejected plan made a draw";
    EXPECT_NO_THROW(validate_affine(settings));
}

TEST(Unit_ImagingAffine, IdentityPlanLeavesTheImageUnchanged)
{
    AffineSettings settings;
    settings.scale = 0.0f;
    settings.translate = 0.0f;
    const ImageSize size{5, 7};
    Random random(9);
    const AffinePlan plan = plan_affine(settings, size, size, random);
    const Image source = make_pattern_image(size, 3);

    Image destination;
    apply_affine(source.view(), plan, destination);

    ASSERT_TRUE(destination.size() == size);
    ASSERT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), bytes_of(source));
}

TEST(Unit_ImagingAffine, IdentityPlanLeavesBoxesUnchanged)
{
    AffineSettings settings;
    settings.scale = 0.0f;
    settings.translate = 0.0f;
    const ImageSize size{480, 640};
    Random random(9);
    const AffinePlan plan = plan_affine(settings, size, size, random);
    const Labels labels = make_labels({{10.0f, 20.0f, 110.0f, 220.0f},
                                       {0.0f, 0.0f, 640.0f, 480.0f},
                                       {300.5f, 100.25f, 340.75f, 150.5f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), labels.size());
    EXPECT_EQ(warped.classes(), labels.classes());

    for (std::size_t i = 0; i < labels.size(); ++i)
        expect_box_near(warped.boxes()[i], labels.boxes()[i]);
}

TEST(Unit_ImagingAffine, TranslationMovesPixelsAndFillsTheUncoveredBorder)
{
    const ImageSize source_size{3, 4};
    const ImageSize output_size{5, 6};
    const Image source = make_pattern_image(source_size, 3);
    const AffinePlan plan = make_plan(source_size, output_size, 1.0f, 2.0f, 1.0f);

    Image destination;
    apply_affine(source.view(), plan, destination);

    ASSERT_TRUE(destination.size() == output_size);
    ASSERT_EQ(destination.channels(), 3);

    for (std::int32_t y = 0; y < output_size.height; ++y)
    {
        for (std::int32_t x = 0; x < output_size.width; ++x)
        {
            const std::int32_t source_y = y - 1;
            const std::int32_t source_x = x - 2;
            const bool is_covered = source_y >= 0 && source_y < 3 && source_x >= 0 &&
                                    source_x < 4;

            for (std::int32_t c = 0; c < 3; ++c)
            {
                const int expected = is_covered ? source.row(source_y)[source_x * 3 + c] : 114;

                EXPECT_EQ(destination.row(y)[x * 3 + c], expected)
                    << "row " << y << ", column " << x << ", channel " << c;
            }
        }
    }
}

TEST(Unit_ImagingAffine, ScaleByTwoInterpolatesAndFadesIntoTheBorder)
{
    const Image source = make_gray_image(ImageSize{2, 2}, {20, 60, 40, 80});
    AffinePlan plan = make_plan(ImageSize{2, 2}, ImageSize{4, 4}, 2.0f, 0.0f, 0.0f);
    plan.border_value = 100;

    Image destination;
    apply_affine(source.view(), plan, destination);

    // Odd coordinates fall halfway between two neighbours; index 2 of the source is border.
    const std::vector<std::uint8_t> expected{20, 40, 60, 80,
                                             30, 50, 70, 85,
                                             40, 60, 80, 90,
                                             70, 80, 90, 95};

    ASSERT_TRUE(destination.size() == (ImageSize{4, 4}));
    EXPECT_EQ(bytes_of(destination), expected);
}

TEST(Unit_ImagingAffine, HalfPixelBlendRoundsToNearest)
{
    const Image source = make_gray_image(ImageSize{1, 2}, {10, 13});
    const AffinePlan plan = make_plan(ImageSize{1, 2}, ImageSize{1, 3}, 1.0f, 0.5f, 0.0f);

    Image destination;
    apply_affine(source.view(), plan, destination);

    // Means with border 114: (114 + 10) / 2 = 62, (10 + 13) / 2 = 11.5, (13 + 114) / 2 = 63.5.
    EXPECT_EQ(bytes_of(destination), (std::vector<std::uint8_t>{62, 12, 64}));
}

TEST(Unit_ImagingAffine, FractionRoundsToTheNearestOf2048Weights)
{
    const Image source = make_gray_image(ImageSize{1, 2}, {20, 18});
    const float offset = 6141.0f / 8192.0f;
    const AffinePlan plan = make_plan(ImageSize{1, 2}, ImageSize{1, 3}, 1.0f, offset, 0.0f);

    Image destination;
    apply_affine(source.view(), plan, destination);

    // The fraction 2051 / 8192 times 2048 is 512.75: weights 1535 and 513, not 1536 and 512.
    const int left = (1535 * 114 + 513 * 20 + 1024) / 2048;
    const int middle = (1535 * 20 + 513 * 18 + 1024) / 2048;
    const int right = (1535 * 18 + 513 * 114 + 1024) / 2048;

    ASSERT_EQ(left, 90);
    ASSERT_EQ(middle, 19);
    ASSERT_EQ(right, 42);
    EXPECT_EQ(bytes_of(destination), (std::vector<std::uint8_t>{90, 19, 42}));
}

TEST(Unit_ImagingAffine, OutputFarFromTheSourceIsAllBorder)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 3);
    AffinePlan plan = make_plan(ImageSize{4, 4}, ImageSize{3, 5}, 1.0f, 1000.0f, -1000.0f);
    plan.border_value = 9;

    Image destination;
    apply_affine(source.view(), plan, destination);

    EXPECT_EQ(bytes_of(destination), std::vector<std::uint8_t>(3 * 5 * 3, 9));
}

TEST(Unit_ImagingAffine, ImageWarpRejectsAPlanItCannotApply)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 1);
    Image destination;

    const AffinePlan other_size = make_plan(ImageSize{4, 5}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
    SA_EXPECT_ERROR(apply_affine(source.view(), other_size, destination), "apply_affine",
                    "plan was made for a 4x5 source", "4x4");

    AffinePlan rotated = make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
    rotated.matrix[1] = 0.5f;
    SA_EXPECT_ERROR(apply_affine(source.view(), rotated, destination), "off-diagonal");

    AffinePlan flat = make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
    flat.matrix[0] = 0.0f;
    SA_EXPECT_ERROR(apply_affine(source.view(), flat, destination), "above zero");

    AffinePlan other_scale = make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 2.0f, 0.0f, 0.0f);
    other_scale.scale = 1.0f;
    SA_EXPECT_ERROR(apply_affine(source.view(), other_scale, destination), "plan's scale");

    AffinePlan unbounded = make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
    unbounded.matrix[2] = std::numeric_limits<float>::infinity();
    SA_EXPECT_ERROR(apply_affine(source.view(), unbounded, destination), "finite translation");

    const AffinePlan plain = make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
    SA_EXPECT_ERROR(apply_affine(ImageView(), plain, destination), "pixels");

    const std::vector<std::uint8_t> five_channels(4 * 4 * 5, 0);
    const ImageView wide(five_channels.data(), ImageSize{4, 4}, 5);
    SA_EXPECT_ERROR(apply_affine(wide, plain, destination), "channel count");

    Image aliased = make_pattern_image(ImageSize{4, 4}, 1);
    const AffinePlan same_size = make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
    SA_EXPECT_ERROR(apply_affine(aliased.view(), same_size, aliased), "apply_affine",
                    "destination holds the source's pixels");

    const ImageView lower_rows(aliased.row(2), ImageSize{2, 4}, 1);
    const AffinePlan two_rows = make_plan(ImageSize{2, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
    SA_EXPECT_ERROR(apply_affine(lower_rows, two_rows, aliased), "apply_affine",
                    "destination holds the source's pixels");
    EXPECT_EQ(bytes_of(aliased), bytes_of(make_pattern_image(ImageSize{4, 4}, 1)));
}

TEST(Unit_ImagingAffine, TranslationMovesBoxesByTheOffset)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 30.0f,
                                      -10.0f);

    const Labels warped = apply_affine(make_labels({{10.0f, 20.0f, 50.0f, 60.0f}}), plan);

    ASSERT_EQ(warped.size(), 1u);
    expect_box_near(warped.boxes()[0], PixelBox{40.0f, 10.0f, 80.0f, 50.0f});
}

TEST(Unit_ImagingAffine, ScaleMultipliesBoxCornersBeforeTheOffset)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 200}, 2.0f, 10.0f,
                                      -20.0f);

    const Labels warped = apply_affine(make_labels({{5.0f, 20.0f, 25.0f, 40.0f}}), plan);

    ASSERT_EQ(warped.size(), 1u);
    expect_box_near(warped.boxes()[0], PixelBox{20.0f, 20.0f, 60.0f, 60.0f});
}

TEST(Unit_ImagingAffine, WarpedBoxIsTheHullOfItsFourWarpedCorners)
{
    const AffinePlan plan = make_plan(ImageSize{400, 400}, ImageSize{100, 100}, 0.5f, -20.0f,
                                      30.0f);
    const PixelBox box{100.0f, 20.0f, 200.0f, 120.0f};
    const std::array<std::array<float, 2>, 4> corners{{{box.x1, box.y1}, {box.x2, box.y2},
                                                       {box.x1, box.y2}, {box.x2, box.y1}}};
    PixelBox hull{1e9f, 1e9f, -1e9f, -1e9f};

    for (const std::array<float, 2>& corner : corners)
    {
        const float x = plan.matrix[0] * corner[0] + plan.matrix[1] * corner[1] + plan.matrix[2];
        const float y = plan.matrix[3] * corner[0] + plan.matrix[4] * corner[1] + plan.matrix[5];

        hull.x1 = std::min(hull.x1, x);
        hull.y1 = std::min(hull.y1, y);
        hull.x2 = std::max(hull.x2, x);
        hull.y2 = std::max(hull.y2, y);
    }

    const Labels warped = apply_affine(make_labels({box}), plan);

    ASSERT_EQ(warped.size(), 1u);
    expect_box_near(warped.boxes()[0], hull);
    expect_box_near(warped.boxes()[0], PixelBox{30.0f, 40.0f, 80.0f, 90.0f});
}

TEST(Unit_ImagingAffine, BoxesAreClippedThenFilteredAndSurvivorsKeepTheirOrder)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f,
                                      0.0f);

    // Each box is 40 by 40; after the shift its width inside the output is 10, 5, 3, 2, 0.
    const Labels labels = make_labels({{10.0f, 10.0f, 50.0f, 50.0f},
                                       {15.0f, 10.0f, 55.0f, 50.0f},
                                       {17.0f, 10.0f, 57.0f, 50.0f},
                                       {18.0f, 10.0f, 58.0f, 50.0f},
                                       {30.0f, 10.0f, 70.0f, 50.0f}});

    const Labels warped = apply_affine(labels, plan);

    // Width 3 fails the area ratio (0.075), width 2 fails the side, width 0 fails both.
    ASSERT_EQ(warped.size(), 2u);
    EXPECT_EQ(warped.classes(), (std::vector<std::int32_t>{0, 1}));
    expect_box_near(warped.boxes()[0], PixelBox{90.0f, 10.0f, 100.0f, 50.0f});
    expect_box_near(warped.boxes()[1], PixelBox{95.0f, 10.0f, 100.0f, 50.0f});
}

TEST(Unit_ImagingAffine, FilterComparesAgainstTheSourceBoxTimesTheScale)
{
    // A 40 by 40 box becomes 10 by 10: ratio 1 against the scaled box, 0.0625 against the raw.
    const AffinePlan shrink = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 0.25f, 0.0f,
                                        0.0f);
    const Labels shrunk = apply_affine(make_labels({{40.0f, 40.0f, 80.0f, 80.0f}}), shrink);

    ASSERT_EQ(shrunk.size(), 1u);
    expect_box_near(shrunk.boxes()[0], PixelBox{10.0f, 10.0f, 20.0f, 20.0f});

    // A 20 by 20 box becomes 40 by 40 and keeps 3 by 40: ratio 0.075 scaled, 0.3 against raw.
    const AffinePlan grow = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 2.0f, 77.0f,
                                      0.0f);
    const Labels grown = apply_affine(make_labels({{10.0f, 10.0f, 30.0f, 30.0f}}), grow);

    EXPECT_TRUE(grown.is_empty());
}

TEST(Unit_ImagingAffine, BoxWarpUsesThePlansFilter)
{
    AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 0.0f, 0.0f);
    const Labels labels = make_labels({{10.0f, 10.0f, 14.0f, 50.0f}});

    ASSERT_EQ(apply_affine(labels, plan).size(), 1u);

    plan.filter.min_side = 4.0f;

    EXPECT_TRUE(apply_affine(labels, plan).is_empty());
}

TEST(Unit_ImagingAffine, BoxWarpRejectsARotatedMatrixAndAScaleThatIsNotThePlans)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 0.0f,
                                      0.0f);

    AffinePlan rotated = plan;
    rotated.matrix[3] = 0.25f;
    SA_EXPECT_ERROR(static_cast<void>(apply_affine(make_labels({}), rotated)), "off-diagonal");

    AffinePlan other_scale = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 2.0f, 0.0f,
                                       0.0f);
    other_scale.scale = 1.0f;
    SA_EXPECT_ERROR(static_cast<void>(apply_affine(make_labels({}), other_scale)),
                    "plan's scale");
}

/** @brief Tests that the settings and the plan default to the 0.10 and the 0.01 area ratios. */
TEST(Unit_ImagingAffine, FiltersDefaultToTheBoxAndThePolygonAreaRatios)
{
    Random random(4u);
    const AffineSettings settings;
    const AffinePlan plan = plan_affine(settings, ImageSize{8, 8}, ImageSize{8, 8}, random);

    EXPECT_NEAR(settings.filter.min_area_ratio, 0.10f, 1e-7f);
    EXPECT_NEAR(settings.polygon_filter.min_area_ratio, 0.01f, 1e-7f);
    EXPECT_NEAR(plan.filter.min_area_ratio, 0.10f, 1e-7f);
    EXPECT_NEAR(plan.polygon_filter.min_area_ratio, 0.01f, 1e-7f);
    EXPECT_NEAR(AffinePlan{}.polygon_filter.min_area_ratio, 0.01f, 1e-7f);
}

/** @brief Tests that every vertex of a polygon is scaled, then moved by the matrix translation. */
TEST(Unit_ImagingAffine, PolygonVerticesAreScaledThenTranslated)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 2.0f, 10.0f,
                                      -5.0f);
    Labels labels;
    labels.add(6, PixelBox{5.0f, 10.0f, 25.0f, 30.0f},
               Polygon{Point{5.0f, 10.0f}, Point{25.0f, 12.0f}, Point{15.0f, 30.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 1u);
    EXPECT_EQ(warped.classes()[0], 6);
    expect_box_near(warped.boxes()[0], PixelBox{20.0f, 15.0f, 60.0f, 55.0f});
    ASSERT_EQ(warped.polygons()[0].size(), 3u);
    EXPECT_NEAR(warped.polygons()[0][0].x, 20.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(warped.polygons()[0][0].y, 15.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(warped.polygons()[0][1].x, 60.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(warped.polygons()[0][1].y, 19.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(warped.polygons()[0][2].x, 40.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(warped.polygons()[0][2].y, 55.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that an object with a polygon takes the box of the polygon's visible part. */
TEST(Unit_ImagingAffine, PolygonObjectTakesTheBoxOfItsVisiblePart)
{
    // The reference's segment2box gives (0, 0, 36.666664, 33) for this triangle in 100 by 80.
    const AffinePlan plan = make_plan(ImageSize{80, 100}, ImageSize{80, 100}, 1.0f, 0.0f, 0.0f);
    const PixelBox bounds{-40.0f, -40.0f, 70.0f, 60.0f};
    Labels with_polygon;
    with_polygon.add(1, bounds,
                     Polygon{Point{-40.0f, -40.0f}, Point{70.0f, -30.0f}, Point{-30.0f, 60.0f}});
    const Labels without_polygon({1}, {bounds});

    const Labels from_polygon = apply_affine(with_polygon, plan);
    const Labels from_box = apply_affine(without_polygon, plan);

    ASSERT_EQ(from_polygon.size(), 1u);
    expect_box_near(from_polygon.boxes()[0], PixelBox{0.0f, 0.0f, 36.666664f, 33.0f});
    ASSERT_EQ(from_box.size(), 1u);
    expect_box_near(from_box.boxes()[0], PixelBox{0.0f, 0.0f, 70.0f, 60.0f});
}

/** @brief Tests that a warped polygon is clamped to the box rebuilt from it. */
TEST(Unit_ImagingAffine, WarpedPolygonIsClampedToItsNewBox)
{
    const AffinePlan plan = make_plan(ImageSize{80, 100}, ImageSize{80, 100}, 1.0f, 0.0f, 0.0f);
    Labels labels;
    labels.add(1, PixelBox{-40.0f, -40.0f, 70.0f, 60.0f},
               Polygon{Point{-40.0f, -40.0f}, Point{70.0f, -30.0f}, Point{-30.0f, 60.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 1u);
    const Polygon& polygon = warped.polygons()[0];
    ASSERT_EQ(polygon.size(), 3u);
    EXPECT_NEAR(polygon[0].x, 0.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(polygon[0].y, 0.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(polygon[1].x, 36.666664f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(polygon[1].y, 0.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(polygon[2].x, 0.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(polygon[2].y, 33.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that no vertex stays outside the output when a crossing rounds below zero. */
TEST(Unit_ImagingAffine, WarpedPolygonStaysInsideTheOutputWhenACrossingRoundsOutside)
{
    // In float the edge from x = -0.03 to x = 0.25 crosses x = 0 at x = -1.86e-9.
    const ImageSize output{80, 100};
    const AffinePlan plan = make_plan(output, output, 1.0f, 0.0f, 0.0f);
    Labels labels;
    labels.add(1, PixelBox{-0.03f, 10.0f, 50.0f, 30.0f},
               Polygon{Point{-0.03f, 10.0f}, Point{0.25f, 20.0f}, Point{50.0f, 30.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 1u);
    const PixelBox& box = warped.boxes()[0];
    EXPECT_EQ(box.x1, 0.0f);
    ASSERT_EQ(warped.polygons()[0].size(), 3u);
    for (const Point& point : warped.polygons()[0])
    {
        EXPECT_GE(point.x, 0.0f);
        EXPECT_LE(point.x, static_cast<float>(output.width));
        EXPECT_GE(point.y, 0.0f);
        EXPECT_LE(point.y, static_cast<float>(output.height));
        EXPECT_GE(point.x, box.x1);
        EXPECT_LE(point.x, box.x2);
    }
}

/** @brief Tests that an object with a polygon is filtered at 0.01 where a bare box is at 0.10. */
TEST(Unit_ImagingAffine, PolygonObjectIsFilteredAtTheLowerAreaRatio)
{
    // The 40 by 40 box keeps 3 by 40 inside the output: an area ratio of 0.075.
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f,
                                      0.0f);
    const PixelBox box{17.0f, 10.0f, 57.0f, 50.0f};
    Labels with_polygon;
    with_polygon.add(3, box,
                     Polygon{Point{17.0f, 10.0f}, Point{57.0f, 10.0f}, Point{57.0f, 50.0f},
                             Point{17.0f, 50.0f}});

    const Labels kept = apply_affine(with_polygon, plan);

    ASSERT_EQ(kept.size(), 1u);
    expect_box_near(kept.boxes()[0], PixelBox{97.0f, 10.0f, 100.0f, 50.0f});
    EXPECT_TRUE(apply_affine(Labels({3}, {box}), plan).is_empty());
}

/** @brief Tests that the polygon filter of the plan decides for an object with a polygon. */
TEST(Unit_ImagingAffine, PolygonWarpUsesThePlansPolygonFilter)
{
    AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f, 0.0f);
    Labels labels;
    labels.add(3, PixelBox{17.0f, 10.0f, 57.0f, 50.0f},
               Polygon{Point{17.0f, 10.0f}, Point{57.0f, 10.0f}, Point{57.0f, 50.0f},
                       Point{17.0f, 50.0f}});

    ASSERT_EQ(apply_affine(labels, plan).size(), 1u);

    plan.filter.min_area_ratio = 0.0f;
    plan.polygon_filter.min_area_ratio = 0.08f;

    EXPECT_TRUE(apply_affine(labels, plan).is_empty());
}

/** @brief Tests that a polygon the warp moves out of the output drops its object. */
TEST(Unit_ImagingAffine, PolygonOutsideTheOutputDropsItsObject)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 500.0f,
                                      0.0f);
    Labels labels;
    labels.add(0, PixelBox{10.0f, 10.0f, 50.0f, 50.0f},
               Polygon{Point{10.0f, 10.0f}, Point{50.0f, 10.0f}, Point{30.0f, 50.0f}});
    labels.add(1, PixelBox{-460.0f, 10.0f, -420.0f, 50.0f},
               Polygon{Point{-460.0f, 10.0f}, Point{-420.0f, 10.0f}, Point{-440.0f, 50.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 1u);
    EXPECT_EQ(warped.classes()[0], 1);
    expect_box_near(warped.boxes()[0], PixelBox{40.0f, 10.0f, 80.0f, 50.0f});
}
