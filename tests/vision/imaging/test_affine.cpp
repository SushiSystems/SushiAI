/**************************************************************************/
/* test_affine.cpp                                                        */
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
 * @file test_affine.cpp
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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/affine.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    /**
     * @brief Holds the absolute gap under which two fractions count as equal: 32 times the float
     *        unit roundoff, about 1.9e-6.
     */
    constexpr float K_DISTINCT_FRACTION_GAP =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Counts the plans a statistical test draws, all from one generator. */
    constexpr int K_SAMPLE_COUNT = 2048;

    /** @brief Holds the seed of the generator a statistical test draws from. */
    constexpr std::uint64_t K_STATISTICAL_SEED = 20261007;

    /** @brief Holds the chance that a correct run of plans still misses its bound. */
    constexpr double K_FALSE_FAILURE_RATE = 1e-9;

    /**
     * @brief Returns the distance a sample mean over K_SAMPLE_COUNT draws stays within of its mean.
     * @param range_width The width of the interval the draws lie in.
     *
     * Hoeffding's inequality gives P(|mean - m| >= t) <= 2 exp(-2 n t^2 / w^2), so t is
     * w sqrt(ln(2 / rate) / (2 n)) for n draws in an interval of width w and a false-failure rate.
     */
    double mean_tolerance(double range_width)
    {
        return range_width * std::sqrt(std::log(2.0 / K_FALSE_FAILURE_RATE) /
                                       (2.0 * static_cast<double>(K_SAMPLE_COUNT)));
    }

    using SushiAI::StatusCode;
    using SushiAI::Test::require_value;

    /**
     * @brief Holds the absolute tolerance, in pixels, of a box edge or a matrix term that float
     *        arithmetic rounds: 32768 times the float unit roundoff, about 2e-3.
     *
     * Such a value is the sum of three products or one interpolation of terms below 1500; each
     * rounding is off by at most 1500 u = 9e-5, so the bound covers a few of them with a margin of
     * ten.
     */
    constexpr float K_PIXEL_TOLERANCE =
        static_cast<float>(32768.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds the x of the crossing the polygon cases expect: 110 / 3, 36.666664 in float. */
    constexpr float K_CROSSING_X = 110.0f / 3.0f;

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

    /**
     * @brief Returns an image of @p size where the pixel at (y, x) is the source pixel at
     *        (y - shift_y, x - shift_x) when that lies inside the source, and @p border otherwise.
     */
    Image make_shifted_image(const Image& source, ImageSize size, std::int32_t shift_x,
                             std::int32_t shift_y, std::uint8_t border)
    {
        Image image(size, source.channels());
        image.fill(border);

        for (std::int32_t y = 0; y < size.height; ++y)
        {
            for (std::int32_t x = 0; x < size.width; ++x)
            {
                const std::int32_t source_y = y - shift_y;
                const std::int32_t source_x = x - shift_x;
                if (source_y < 0 || source_y >= source.height() || source_x < 0 ||
                    source_x >= source.width())
                    continue;

                std::copy_n(source.row(source_y) + source_x * source.channels(), source.channels(),
                            image.row(y) + x * source.channels());
            }
        }

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

    /** @brief Returns the identity plan of a 4 by 4 image. */
    AffinePlan make_identity_plan()
    {
        return make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);
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

    /** @brief Checks that the four edges of two boxes are equal. */
    void expect_box(const PixelBox& actual, const PixelBox& expected)
    {
        EXPECT_EQ(actual.x1, expected.x1);
        EXPECT_EQ(actual.y1, expected.y1);
        EXPECT_EQ(actual.x2, expected.x2);
        EXPECT_EQ(actual.y2, expected.y2);
    }

    /** @brief Checks that a vertex lies inside the output and between the box's left and right. */
    void expect_vertex_inside(const Point& point, ImageSize output, const PixelBox& box)
    {
        EXPECT_GE(point.x, 0.0f);
        EXPECT_LE(point.x, static_cast<float>(output.width));
        EXPECT_GE(point.y, 0.0f);
        EXPECT_LE(point.y, static_cast<float>(output.height));
        EXPECT_GE(point.x, box.x1);
        EXPECT_LE(point.x, box.x2);
    }

    /** @brief Returns the bytes of an image in row-major order. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Describes the area ratios of a plan and whether its polygon object must survive. */
    struct RatioRow
    {
        float plain_ratio;
        float polygon_ratio;
        bool expected_kept;
    };
} // namespace

/** @brief Tests that zero ranges put the source centre on the output centre. */
TEST(Small_ImagingAffine, ZeroRangesPutTheSourceCentreOnTheOutputCentre)
{
    AffineSettings settings;
    settings.scale = 0.0f;
    settings.translate = 0.0f;
    Random random(11);

    const AffinePlan plan =
        require_value(plan_affine(settings, ImageSize{1200, 1000}, ImageSize{480, 640}, random));

    // By hand: scale 1, T = (320, 240), C = (-500, -600), so M = [1 0 -180; 0 1 -360].
    EXPECT_EQ(plan.scale, 1.0f);
    EXPECT_EQ(plan.translate_x, 320.0f);
    EXPECT_EQ(plan.translate_y, 240.0f);
    EXPECT_EQ(plan.matrix, (std::array<float, 6>{1.0f, 0.0f, -180.0f, 0.0f, 1.0f, -360.0f}));
}

/** @brief Tests that the plan's matrix is the translation times the scale times the centring. */
TEST(Small_ImagingAffine, MatrixEqualsTranslationTimesScaleTimesCentre)
{
    const AffineSettings settings;
    const ImageSize source{1280, 960};
    const ImageSize output{640, 480};
    Random random(2026);

    const AffinePlan plan = require_value(plan_affine(settings, source, output, random));

    // At scale 1 the product is the same in any order, so the seed must give another scale.
    ASSERT_GT(std::abs(plan.scale - 1.0f), 0.05f);

    const Matrix3 centre{1.0f, 0.0f, -480.0f, 0.0f, 1.0f, -640.0f, 0.0f, 0.0f, 1.0f};
    const Matrix3 scaling{plan.scale, 0.0f, 0.0f, 0.0f, plan.scale, 0.0f, 0.0f, 0.0f, 1.0f};
    const Matrix3 translation{1.0f, 0.0f, plan.translate_x, 0.0f, 1.0f, plan.translate_y,
                              0.0f, 0.0f, 1.0f};
    const Matrix3 expected = multiply(translation, multiply(scaling, centre));

    for (std::size_t i = 0; i < plan.matrix.size(); ++i)
        EXPECT_NEAR(plan.matrix[i], expected[i], K_PIXEL_TOLERANCE) << "matrix term " << i;
}

/** @brief Tests that the plan makes exactly three draws. */
TEST(Small_ImagingAffine, PlanConsumesExactlyThreeDraws)
{
    Random random(5);
    Random replay = random;

    const AffinePlan plan =
        require_value(plan_affine(AffineSettings{}, ImageSize{64, 64}, ImageSize{32, 32}, random));
    static_cast<void>(plan);

    static_cast<void>(replay.next_u64());
    static_cast<void>(replay.next_u64());
    static_cast<void>(replay.next_u64());

    EXPECT_TRUE(random == replay);
}

/** @brief Tests that the plan holds the sizes, the border value and the filter of the settings. */
TEST(Small_ImagingAffine, PlanCarriesSizesBorderValueAndFilter)
{
    AffineSettings settings;
    settings.border_value = 37;
    settings.filter.min_side = 5.0f;
    settings.filter.min_area_ratio = 0.25f;
    Random random(3);

    const AffinePlan plan =
        require_value(plan_affine(settings, ImageSize{20, 30}, ImageSize{8, 12}, random));

    EXPECT_TRUE(plan.source == (ImageSize{20, 30}));
    EXPECT_TRUE(plan.output == (ImageSize{8, 12}));
    EXPECT_EQ(plan.border_value, 37);
    EXPECT_EQ(plan.filter.min_side, 5.0f);
    EXPECT_EQ(plan.filter.min_area_ratio, 0.25f);
}

/** @brief Tests that drawn parameters stay in range and have the means of their ranges. */
TEST(Small_ImagingAffine, DrawnParametersStayInRangeAndCentreOnTheirMeans)
{
    const AffineSettings settings;
    const ImageSize source{1280, 1280};
    const ImageSize output{640, 480};
    const float low_fraction = 0.5f - 0.1f;
    const float high_fraction = 0.5f + 0.1f;
    Random random(K_STATISTICAL_SEED);

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

    for (int count = 0; count < K_SAMPLE_COUNT; ++count)
    {
        const AffinePlan plan = require_value(plan_affine(settings, source, output, random));

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

        // Fractions closer than K_DISTINCT_FRACTION_GAP would mean one draw served both axes.
        if (std::abs(x_fraction - y_fraction) < K_DISTINCT_FRACTION_GAP)
            ++equal_fractions;
    }

    // The scale lies in an interval of width 1 and a fraction in one of width 0.2.
    EXPECT_NEAR(scale_sum / K_SAMPLE_COUNT, 1.0, mean_tolerance(1.0));
    EXPECT_NEAR(x_sum / K_SAMPLE_COUNT, 0.5, mean_tolerance(0.2));
    EXPECT_NEAR(y_sum / K_SAMPLE_COUNT, 0.5, mean_tolerance(0.2));

    // 2048 uniform draws miss the outer twentieth of a range with probability 0.95^2048, 2e-46.
    EXPECT_LT(scale_min, 0.55f);
    EXPECT_GT(scale_max, 1.45f);
    EXPECT_LT(x_min, 0.41f);
    EXPECT_GT(x_max, 0.59f);
    EXPECT_LT(y_min, 0.41f);
    EXPECT_GT(y_max, 0.59f);
    EXPECT_LT(equal_fractions, K_SAMPLE_COUNT / 100);
}

/** @brief Tests that a scale range outside its bounds is refused by the plan and by validate. */
TEST(Small_ImagingAffine, PlanRejectsAScaleRangeOutsideItsBounds)
{
    const ImageSize size{16, 16};
    Random random(1);

    AffineSettings wide_scale;
    wide_scale.scale = 1.0f;
    SA_EXPECT_FAILURE(plan_affine(wide_scale, size, size, random),
                      StatusCode::InvalidDescription, "validate_affine", "scale");
    SA_EXPECT_FAILURE(validate_affine(wide_scale), StatusCode::InvalidDescription,
                      "validate_affine", "scale half-range of 1");

    AffineSettings negative_scale;
    negative_scale.scale = -0.1f;
    SA_EXPECT_FAILURE(plan_affine(negative_scale, size, size, random),
                      StatusCode::InvalidDescription, "scale");

    AffineSettings unordered_scale;
    unordered_scale.scale = std::numeric_limits<float>::quiet_NaN();
    SA_EXPECT_FAILURE(plan_affine(unordered_scale, size, size, random),
                      StatusCode::InvalidDescription, "scale");
}

/** @brief Tests that a translate range outside its bounds is refused by the plan and validate. */
TEST(Small_ImagingAffine, PlanRejectsATranslateRangeOutsideItsBounds)
{
    const ImageSize size{16, 16};
    Random random(1);

    AffineSettings wide_translate;
    wide_translate.translate = 0.6f;
    SA_EXPECT_FAILURE(plan_affine(wide_translate, size, size, random),
                      StatusCode::InvalidDescription, "validate_affine", "translate");
    SA_EXPECT_FAILURE(validate_affine(wide_translate), StatusCode::InvalidDescription,
                      "validate_affine", "translate half-range");

    AffineSettings negative_translate;
    negative_translate.translate = -0.1f;
    SA_EXPECT_FAILURE(plan_affine(negative_translate, size, size, random),
                      StatusCode::InvalidDescription, "translate");
}

/** @brief Tests that a source or an output size that is not positive is refused. */
TEST(Small_ImagingAffine, PlanRejectsASizeThatIsNotPositive)
{
    const ImageSize size{16, 16};
    const AffineSettings settings;
    Random random(1);

    SA_EXPECT_FAILURE(plan_affine(settings, ImageSize{0, 16}, size, random),
                      StatusCode::InvalidDescription, "plan_affine", "source size of 0x16");
    SA_EXPECT_FAILURE(plan_affine(settings, size, ImageSize{16, 0}, random),
                      StatusCode::InvalidDescription, "plan_affine", "output size of 16x0");
}

/** @brief Tests that a plan refused for its settings or its sizes makes no draw. */
TEST(Small_ImagingAffine, PlanRejectedForItsSettingsOrSizesMakesNoDraw)
{
    const ImageSize size{16, 16};
    const AffineSettings settings;
    AffineSettings wide_scale;
    wide_scale.scale = 1.0f;
    AffineSettings wide_translate;
    wide_translate.translate = 0.6f;
    Random random(1);

    static_cast<void>(plan_affine(wide_scale, size, size, random));
    static_cast<void>(plan_affine(wide_translate, size, size, random));
    static_cast<void>(plan_affine(settings, ImageSize{0, 16}, size, random));
    static_cast<void>(plan_affine(settings, size, ImageSize{16, 0}, random));

    EXPECT_TRUE(random == Random(1)) << "a rejected plan made a draw";
}

/** @brief Tests that the default settings pass validation. */
TEST(Small_ImagingAffine, ValidateAcceptsTheDefaultSettings)
{
    SA_EXPECT_OK(validate_affine(AffineSettings{}));
}

/** @brief Tests that the identity plan warps an image to the same bytes. */
TEST(Small_ImagingAffine, IdentityPlanLeavesTheImageUnchanged)
{
    AffineSettings settings;
    settings.scale = 0.0f;
    settings.translate = 0.0f;
    const ImageSize size{5, 7};
    Random random(9);
    const AffinePlan plan = require_value(plan_affine(settings, size, size, random));
    const Image source = make_pattern_image(size, 3);

    Image destination;
    apply_affine(source.view(), plan, destination);

    ASSERT_TRUE(destination.size() == size);
    ASSERT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), bytes_of(source));
}

/** @brief Tests that the identity plan keeps the classes and the edges of three boxes. */
TEST(Small_ImagingAffine, IdentityPlanLeavesBoxesUnchanged)
{
    AffineSettings settings;
    settings.scale = 0.0f;
    settings.translate = 0.0f;
    const ImageSize size{480, 640};
    Random random(9);
    const AffinePlan plan = require_value(plan_affine(settings, size, size, random));
    const Labels labels =
        make_labels({{10.0f, 20.0f, 110.0f, 220.0f}, {0.0f, 0.0f, 640.0f, 480.0f},
                     {300.5f, 100.25f, 340.75f, 150.5f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), labels.size());
    EXPECT_EQ(warped.classes(), labels.classes());

    for (std::size_t i = 0; i < labels.size(); ++i)
        expect_box(warped.boxes()[i], labels.boxes()[i]);
}

/** @brief Tests that a translation moves the pixels and fills the uncovered border. */
TEST(Small_ImagingAffine, TranslationMovesPixelsAndFillsTheUncoveredBorder)
{
    const ImageSize source_size{3, 4};
    const ImageSize output_size{5, 6};
    const Image source = make_pattern_image(source_size, 3);
    const AffinePlan plan = make_plan(source_size, output_size, 1.0f, 2.0f, 1.0f);

    Image destination;
    apply_affine(source.view(), plan, destination);

    ASSERT_TRUE(destination.size() == output_size);
    ASSERT_EQ(destination.channels(), 3);
    EXPECT_EQ(bytes_of(destination), bytes_of(make_shifted_image(source, output_size, 2, 1, 114)));
}

/** @brief Tests that a scale of two interpolates and fades into the border. */
TEST(Small_ImagingAffine, ScaleByTwoInterpolatesAndFadesIntoTheBorder)
{
    const Image source = make_gray_image(ImageSize{2, 2}, {20, 60, 40, 80});
    AffinePlan plan = make_plan(ImageSize{2, 2}, ImageSize{4, 4}, 2.0f, 0.0f, 0.0f);
    plan.border_value = 100;

    Image destination;
    apply_affine(source.view(), plan, destination);

    // Odd coordinates fall halfway between two neighbours; index 2 of the source is border.
    const std::vector<std::uint8_t> expected{20, 40, 60, 80, 30, 50, 70, 85, 40, 60, 80, 90, 70, 80,
                                             90, 95};

    ASSERT_TRUE(destination.size() == (ImageSize{4, 4}));
    EXPECT_EQ(bytes_of(destination), expected);
}

/** @brief Tests that a half-pixel blend rounds to the nearest level, half up. */
TEST(Small_ImagingAffine, HalfPixelBlendRoundsToNearest)
{
    const Image source = make_gray_image(ImageSize{1, 2}, {10, 13});
    const AffinePlan plan = make_plan(ImageSize{1, 2}, ImageSize{1, 3}, 1.0f, 0.5f, 0.0f);

    Image destination;
    apply_affine(source.view(), plan, destination);

    // Means with border 114: (114 + 10) / 2 = 62, (10 + 13) / 2 = 11.5, (13 + 114) / 2 = 63.5.
    EXPECT_EQ(bytes_of(destination), (std::vector<std::uint8_t>{62, 12, 64}));
}

/** @brief Tests that a fraction rounds to the nearest of 2048 weights. */
TEST(Small_ImagingAffine, FractionRoundsToTheNearestOf2048Weights)
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

/** @brief Tests that an output far from the source is all border. */
TEST(Small_ImagingAffine, OutputFarFromTheSourceIsAllBorder)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 3);
    AffinePlan plan = make_plan(ImageSize{4, 4}, ImageSize{3, 5}, 1.0f, 1000.0f, -1000.0f);
    plan.border_value = 9;

    Image destination;
    apply_affine(source.view(), plan, destination);

    EXPECT_EQ(bytes_of(destination), std::vector<std::uint8_t>(3 * 5 * 3, 9));
}

/** @brief Tests that the image warp stops on a plan made for another source size. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnAPlanOfAnotherSourceSize)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 1);
    Image destination;
    const AffinePlan other_size = make_plan(ImageSize{4, 5}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(source.view(), other_size, destination)),
                    "apply_affine", "plan was made for a 4x5 source", "4x4");
}

/** @brief Tests that the image warp stops on a matrix with an off-diagonal term. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnAShear)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 1);
    Image destination;
    AffinePlan rotated = make_identity_plan();
    rotated.matrix[1] = 0.5f;

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(source.view(), rotated, destination)),
                    "off-diagonal");
}

/** @brief Tests that the image warp stops on a diagonal term of zero. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnAZeroDiagonal)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 1);
    Image destination;
    AffinePlan flat = make_identity_plan();
    flat.matrix[0] = 0.0f;

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(source.view(), flat, destination)),
                    "above zero");
}

/** @brief Tests that the image warp stops on a matrix scale that is not the plan's scale. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnAScaleThatIsNotThePlans)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 1);
    Image destination;
    AffinePlan other_scale = make_plan(ImageSize{4, 4}, ImageSize{4, 4}, 2.0f, 0.0f, 0.0f);
    other_scale.scale = 1.0f;

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(source.view(), other_scale, destination)),
                    "plan's scale");
}

/** @brief Tests that the image warp stops on an infinite translation. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnAnInfiniteTranslation)
{
    const Image source = make_pattern_image(ImageSize{4, 4}, 1);
    Image destination;
    AffinePlan unbounded = make_identity_plan();
    unbounded.matrix[2] = std::numeric_limits<float>::infinity();

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(source.view(), unbounded, destination)),
                    "finite translation");
}

/** @brief Tests that the image warp stops on a source without pixels. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnASourceWithoutPixels)
{
    Image destination;

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(ImageView(), make_identity_plan(), destination)),
                    "pixels");
}

/** @brief Tests that the image warp stops on a source of five channels. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnASourceOfFiveChannels)
{
    Image destination;
    const std::vector<std::uint8_t> five_channels(4 * 4 * 5, 0);
    const ImageView wide(five_channels.data(), ImageSize{4, 4}, 5);

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(wide, make_identity_plan(), destination)),
                    "channel count");
}

/** @brief Tests that the image warp stops on a source the destination holds, whole or in rows. */
TEST(Medium_ImagingAffine, ImageWarpStopsOnASourceTheDestinationHolds)
{
    Image aliased = make_pattern_image(ImageSize{4, 4}, 1);
    const ImageView lower_rows(aliased.row(2), ImageSize{2, 4}, 1);
    const AffinePlan two_rows = make_plan(ImageSize{2, 4}, ImageSize{4, 4}, 1.0f, 0.0f, 0.0f);

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(aliased.view(), make_identity_plan(), aliased)),
                    "apply_affine", "destination holds the source's pixels");
    SA_EXPECT_FATAL(static_cast<void>(apply_affine(lower_rows, two_rows, aliased)), "apply_affine",
                    "destination holds the source's pixels");
}

/** @brief Tests that a translation moves a box by the offset. */
TEST(Small_ImagingAffine, TranslationMovesBoxesByTheOffset)
{
    const AffinePlan plan =
        make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 30.0f, -10.0f);

    const Labels warped = apply_affine(make_labels({{10.0f, 20.0f, 50.0f, 60.0f}}), plan);

    ASSERT_EQ(warped.size(), 1u);
    expect_box(warped.boxes()[0], PixelBox{40.0f, 10.0f, 80.0f, 50.0f});
}

/** @brief Tests that the scale multiplies the box corners before the offset is added. */
TEST(Small_ImagingAffine, ScaleMultipliesBoxCornersBeforeTheOffset)
{
    const AffinePlan plan =
        make_plan(ImageSize{100, 100}, ImageSize{100, 200}, 2.0f, 10.0f, -20.0f);

    const Labels warped = apply_affine(make_labels({{5.0f, 20.0f, 25.0f, 40.0f}}), plan);

    ASSERT_EQ(warped.size(), 1u);
    expect_box(warped.boxes()[0], PixelBox{20.0f, 20.0f, 60.0f, 60.0f});
}

/** @brief Tests that a warped box is the hull of its four warped corners. */
TEST(Small_ImagingAffine, WarpedBoxIsTheHullOfItsFourWarpedCorners)
{
    const AffinePlan plan =
        make_plan(ImageSize{400, 400}, ImageSize{100, 100}, 0.5f, -20.0f, 30.0f);

    // The four corners map to (30, 40), (80, 40), (30, 90) and (80, 90); their hull is the box.
    const Labels warped = apply_affine(make_labels({{100.0f, 20.0f, 200.0f, 120.0f}}), plan);

    ASSERT_EQ(warped.size(), 1u);
    expect_box(warped.boxes()[0], PixelBox{30.0f, 40.0f, 80.0f, 90.0f});
}

/** @brief Tests that boxes are clipped to the output. */
TEST(Small_ImagingAffine, BoxesAreClippedToTheOutput)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f, 0.0f);

    // Each box is 40 by 40; after the shift its width inside the output is 10, 5, 3, 2, 0.
    const Labels labels =
        make_labels({{10.0f, 10.0f, 50.0f, 50.0f}, {15.0f, 10.0f, 55.0f, 50.0f},
                     {17.0f, 10.0f, 57.0f, 50.0f}, {18.0f, 10.0f, 58.0f, 50.0f},
                     {30.0f, 10.0f, 70.0f, 50.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 2u);
    expect_box(warped.boxes()[0], PixelBox{90.0f, 10.0f, 100.0f, 50.0f});
    expect_box(warped.boxes()[1], PixelBox{95.0f, 10.0f, 100.0f, 50.0f});
}

/** @brief Tests that clipped boxes that fail the side or the area ratio are dropped. */
TEST(Small_ImagingAffine, ClippedBoxesBelowTheFilterAreDropped)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f, 0.0f);
    const Labels labels =
        make_labels({{10.0f, 10.0f, 50.0f, 50.0f}, {15.0f, 10.0f, 55.0f, 50.0f},
                     {17.0f, 10.0f, 57.0f, 50.0f}, {18.0f, 10.0f, 58.0f, 50.0f},
                     {30.0f, 10.0f, 70.0f, 50.0f}});

    const Labels warped = apply_affine(labels, plan);

    // Width 3 fails the area ratio (0.075), width 2 fails the side, width 0 fails both.
    EXPECT_EQ(warped.size(), 2u);
}

/** @brief Tests that the boxes that survive the filter keep their order. */
TEST(Small_ImagingAffine, SurvivingBoxesKeepTheirOrder)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f, 0.0f);
    const Labels labels =
        make_labels({{10.0f, 10.0f, 50.0f, 50.0f}, {15.0f, 10.0f, 55.0f, 50.0f},
                     {17.0f, 10.0f, 57.0f, 50.0f}, {18.0f, 10.0f, 58.0f, 50.0f},
                     {30.0f, 10.0f, 70.0f, 50.0f}});

    const Labels warped = apply_affine(labels, plan);

    EXPECT_EQ(warped.classes(), (std::vector<std::int32_t>{0, 1}));
}

/** @brief Tests that the filter compares against the source box times the scale. */
TEST(Small_ImagingAffine, FilterComparesAgainstTheSourceBoxTimesTheScale)
{
    // A 40 by 40 box becomes 10 by 10: ratio 1 against the scaled box, 0.0625 against the raw.
    const AffinePlan shrink =
        make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 0.25f, 0.0f, 0.0f);
    const Labels shrunk = apply_affine(make_labels({{40.0f, 40.0f, 80.0f, 80.0f}}), shrink);

    ASSERT_EQ(shrunk.size(), 1u);
    expect_box(shrunk.boxes()[0], PixelBox{10.0f, 10.0f, 20.0f, 20.0f});

    // A 20 by 20 box becomes 40 by 40 and keeps 3 by 40: ratio 0.075 scaled, 0.3 against raw.
    const AffinePlan grow = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 2.0f, 77.0f, 0.0f);
    const Labels grown = apply_affine(make_labels({{10.0f, 10.0f, 30.0f, 30.0f}}), grow);

    EXPECT_TRUE(grown.is_empty());
}

/** @brief Tests that the box warp uses the plan's filter, here its minimum side. */
TEST(Small_ImagingAffine, BoxWarpUsesThePlansFilter)
{
    AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 0.0f, 0.0f);
    const Labels labels = make_labels({{10.0f, 10.0f, 14.0f, 50.0f}});

    ASSERT_EQ(apply_affine(labels, plan).size(), 1u);

    plan.filter.min_side = 4.0f;

    EXPECT_TRUE(apply_affine(labels, plan).is_empty());
}

/** @brief Tests that the box warp stops on a matrix with an off-diagonal term. */
TEST(Medium_ImagingAffine, BoxWarpStopsOnARotatedMatrix)
{
    AffinePlan rotated = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 0.0f, 0.0f);
    rotated.matrix[3] = 0.25f;

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(make_labels({}), rotated)), "off-diagonal");
}

/** @brief Tests that the box warp stops on a matrix scale that is not the plan's scale. */
TEST(Medium_ImagingAffine, BoxWarpStopsOnAScaleThatIsNotThePlans)
{
    AffinePlan other_scale = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 2.0f, 0.0f, 0.0f);
    other_scale.scale = 1.0f;

    SA_EXPECT_FATAL(static_cast<void>(apply_affine(make_labels({}), other_scale)), "plan's scale");
}

/** @brief Tests that every vertex of a polygon is scaled, then moved by the matrix translation. */
TEST(Small_ImagingAffine, PolygonVerticesAreScaledThenTranslated)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 2.0f, 10.0f, -5.0f);
    Labels labels;
    labels.add(6, PixelBox{5.0f, 10.0f, 25.0f, 30.0f},
               Polygon{Point{5.0f, 10.0f}, Point{25.0f, 12.0f}, Point{15.0f, 30.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 1u);
    EXPECT_EQ(warped.classes()[0], 6);
    expect_box(warped.boxes()[0], PixelBox{20.0f, 15.0f, 60.0f, 55.0f});
    ASSERT_EQ(warped.polygons()[0].size(), 3u);
    EXPECT_EQ(warped.polygons()[0][0].x, 20.0f);
    EXPECT_EQ(warped.polygons()[0][0].y, 15.0f);
    EXPECT_EQ(warped.polygons()[0][1].x, 60.0f);
    EXPECT_EQ(warped.polygons()[0][1].y, 19.0f);
    EXPECT_EQ(warped.polygons()[0][2].x, 40.0f);
    EXPECT_EQ(warped.polygons()[0][2].y, 55.0f);
}

/** @brief Tests that an object with a polygon takes the box of the polygon's visible part. */
TEST(Small_ImagingAffine, PolygonObjectTakesTheBoxOfItsVisiblePart)
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
    expect_box_near(from_polygon.boxes()[0], PixelBox{0.0f, 0.0f, K_CROSSING_X, 33.0f});
    ASSERT_EQ(from_box.size(), 1u);
    expect_box(from_box.boxes()[0], PixelBox{0.0f, 0.0f, 70.0f, 60.0f});
}

/** @brief Tests that a warped polygon is clamped to the box rebuilt from it. */
TEST(Small_ImagingAffine, WarpedPolygonIsClampedToItsNewBox)
{
    const AffinePlan plan = make_plan(ImageSize{80, 100}, ImageSize{80, 100}, 1.0f, 0.0f, 0.0f);
    Labels labels;
    labels.add(1, PixelBox{-40.0f, -40.0f, 70.0f, 60.0f},
               Polygon{Point{-40.0f, -40.0f}, Point{70.0f, -30.0f}, Point{-30.0f, 60.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 1u);
    const Polygon& polygon = warped.polygons()[0];
    ASSERT_EQ(polygon.size(), 3u);
    EXPECT_EQ(polygon[0].x, 0.0f);
    EXPECT_EQ(polygon[0].y, 0.0f);
    EXPECT_NEAR(polygon[1].x, K_CROSSING_X, K_PIXEL_TOLERANCE);
    EXPECT_EQ(polygon[1].y, 0.0f);
    EXPECT_EQ(polygon[2].x, 0.0f);
    EXPECT_NEAR(polygon[2].y, 33.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that no vertex stays outside the output when a crossing rounds below zero. */
TEST(Small_ImagingAffine, WarpedPolygonStaysInsideTheOutputWhenACrossingRoundsOutside)
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
    expect_vertex_inside(warped.polygons()[0][0], output, box);
    expect_vertex_inside(warped.polygons()[0][1], output, box);
    expect_vertex_inside(warped.polygons()[0][2], output, box);
}

namespace
{
/** @brief Rows of a box ratio and a polygon ratio and whether the polygon object survives. */
class Small_ImagingAffinePolygonRatio : public ::testing::TestWithParam<RatioRow>
{
};
} // namespace

/** @brief Tests that an object with a polygon is filtered at the polygon ratio of the plan. */
TEST_P(Small_ImagingAffinePolygonRatio, PolygonObjectIsFilteredAtThePolygonAreaRatio)
{
    // The 40 by 40 box keeps 3 by 40 inside the output: an area ratio of 0.075.
    const RatioRow& row = GetParam();
    Labels with_polygon;
    with_polygon.add(3, PixelBox{17.0f, 10.0f, 57.0f, 50.0f},
                     Polygon{Point{17.0f, 10.0f}, Point{57.0f, 10.0f}, Point{57.0f, 50.0f},
                             Point{17.0f, 50.0f}});
    AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f, 0.0f);
    plan.filter.min_area_ratio = row.plain_ratio;
    plan.polygon_filter.min_area_ratio = row.polygon_ratio;

    const Labels warped = apply_affine(with_polygon, plan);

    EXPECT_EQ(warped.size(), row.expected_kept ? 1u : 0u);
}

INSTANTIATE_TEST_SUITE_P(Ratios, Small_ImagingAffinePolygonRatio,
                         ::testing::Values(RatioRow{0.10f, 0.01f, true},
                                           RatioRow{0.0f, 0.08f, false}));

/** @brief Tests that the same box without a polygon is filtered at the higher box ratio. */
TEST(Small_ImagingAffine, BoxWithoutPolygonIsFilteredAtTheBoxAreaRatio)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 80.0f, 0.0f);
    const PixelBox box{17.0f, 10.0f, 57.0f, 50.0f};

    EXPECT_TRUE(apply_affine(Labels({3}, {box}), plan).is_empty());
}

/** @brief Tests that a polygon the warp moves out of the output drops its object. */
TEST(Small_ImagingAffine, PolygonOutsideTheOutputDropsItsObject)
{
    const AffinePlan plan = make_plan(ImageSize{100, 100}, ImageSize{100, 100}, 1.0f, 500.0f, 0.0f);
    Labels labels;
    labels.add(0, PixelBox{10.0f, 10.0f, 50.0f, 50.0f},
               Polygon{Point{10.0f, 10.0f}, Point{50.0f, 10.0f}, Point{30.0f, 50.0f}});
    labels.add(1, PixelBox{-460.0f, 10.0f, -420.0f, 50.0f},
               Polygon{Point{-460.0f, 10.0f}, Point{-420.0f, 10.0f}, Point{-440.0f, 50.0f}});

    const Labels warped = apply_affine(labels, plan);

    ASSERT_EQ(warped.size(), 1u);
    EXPECT_EQ(warped.classes()[0], 1);
    expect_box(warped.boxes()[0], PixelBox{40.0f, 10.0f, 80.0f, 50.0f});
}
