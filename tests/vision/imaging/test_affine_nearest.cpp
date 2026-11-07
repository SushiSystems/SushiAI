/**************************************************************************/
/* test_affine_nearest.cpp                                                */
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
 * @file test_affine_nearest.cpp
 * @brief Tests that the nearest warp resets its destination and fills with the plan's border.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/affine.hpp>
#include <SushiAI/vision/imaging/affine_nearest.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Returns a plan from @p size to itself that shifts by @p shift_x columns. */
    AffinePlan make_shift_plan(ImageSize size, float shift_x, std::uint8_t border)
    {
        AffinePlan plan;
        plan.source = size;
        plan.output = size;
        plan.matrix = {1.0f, 0.0f, shift_x, 0.0f, 1.0f, 0.0f};
        plan.border_value = border;
        return plan;
    }

    /** @brief Returns a one-channel image whose byte i holds i + 1. */
    Image counting_image(ImageSize size)
    {
        Image image(size, 1);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(i + 1);
        return image;
    }
    /** @brief Holds the byte the tests give a pixel no source pixel covers. */
    constexpr std::uint8_t K_VOID = 255;

    /** @brief Returns a plan that scales by one factor and translates, with a border value. */
    AffinePlan make_plan(ImageSize source, ImageSize output, float scale, float shift_x,
                         float shift_y)
    {
        AffinePlan plan;
        plan.source = source;
        plan.output = output;
        plan.scale = scale;
        plan.matrix = {scale, 0.0f, shift_x, 0.0f, scale, shift_y};
        plan.border_value = K_VOID;
        return plan;
    }

    /** @brief Returns an image holding exactly the listed bytes, in order. */
    Image make_image_of(ImageSize size, std::int32_t channels,
                        std::initializer_list<std::uint8_t> bytes)
    {
        Image image(size, channels);
        std::size_t offset = 0;
        for (const std::uint8_t value : bytes)
        {
            image.data()[offset] = value;
            ++offset;
        }

        return image;
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
} // namespace

/** @brief Tests that a destination of another size and channel count takes the plan's shape. */
TEST(Small_ImagingAffineNearest, ADestinationOfAnotherShapeTakesThePlanOutput)
{
    const ImageSize size{2, 3};
    const Image source = counting_image(size);
    Image warped(ImageSize{5, 5}, 3);
    warped.fill(9);

    SA_EXPECT_OK(apply_affine_nearest(source.view(), make_shift_plan(size, 0.0f, 0), warped));

    EXPECT_EQ(warped.size(), size);
    EXPECT_EQ(warped.channels(), 1);
}

/** @brief Tests that the identity warp into a used destination copies the source bytes. */
TEST(Small_ImagingAffineNearest, TheIdentityWarpIntoAUsedDestinationCopiesTheSourceBytes)
{
    const ImageSize size{2, 3};
    const Image source = counting_image(size);
    Image warped(ImageSize{5, 5}, 3);
    warped.fill(9);

    SA_EXPECT_OK(apply_affine_nearest(source.view(), make_shift_plan(size, 0.0f, 0), warped));

    ASSERT_EQ(warped.byte_count(), source.byte_count());
    for (std::size_t i = 0; i < source.byte_count(); ++i)
        EXPECT_EQ(warped.data()[i], source.data()[i]) << "byte " << i;
}

/** @brief Tests that the plan's own border value fills the column a shift uncovers. */
TEST(Small_ImagingAffineNearest, ThePlansBorderValueFillsWhatTheSourceDoesNotCover)
{
    const ImageSize size{1, 3};
    const Image source = counting_image(size);
    Image warped;

    SA_EXPECT_OK(apply_affine_nearest(source.view(), make_shift_plan(size, 1.0f, 17), warped));

    EXPECT_EQ(warped.data()[0], 17);
    EXPECT_EQ(warped.data()[1], 1);
    EXPECT_EQ(warped.data()[2], 2);
}

/** @brief Tests that a scale of 2 repeats pixels, rounding a half up, and blends none. */
TEST(Small_ImagingAffineNearest, AScaleRepeatsPixelsAndBlendsNone)
{
    const Image source = make_image_of(ImageSize{1, 4}, 1, {3, 50, 7, 120});

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(
        source.view(), make_plan(ImageSize{1, 4}, ImageSize{1, 8}, 2.0f, 0.0f, 0.0f), warped));

    expect_bytes(warped, {3, 50, 50, 7, 7, 120, 120, K_VOID});
}

/** @brief Tests that every channel of a pixel moves together. */
TEST(Small_ImagingAffineNearest, EveryChannelOfAPixelMovesTogether)
{
    const Image source = make_image_of(ImageSize{1, 2}, K_RGB_CHANNELS, {1, 2, 3, 4, 5, 6});

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(
        source.view(), make_plan(ImageSize{1, 2}, ImageSize{1, 2}, 1.0f, 1.0f, 0.0f), warped));

    expect_bytes(warped, {K_VOID, K_VOID, K_VOID, 1, 2, 3});
}

/** @brief Tests that a view without pixels and a plan of another source size are refused. */
TEST(Small_ImagingAffineNearest, RefusesAViewWithoutPixelsAndAPlanOfAnotherSize)
{
    const ImageSize size{2, 2};
    const Image source = counting_image(size);
    Image warped;

    SA_EXPECT_FAILURE(apply_affine_nearest(ImageView(), make_plan(size, size, 1.0f, 0.0f, 0.0f),
                                           warped),
                      StatusCode::InvalidDescription, "holds no image");
    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(),
                                           make_plan(ImageSize{3, 3}, size, 1.0f, 0.0f, 0.0f),
                                           warped),
                      StatusCode::Incompatible, "3x3", "2x2");
}

/** @brief Tests that a destination that shares the source's pixels is refused. */
TEST(Small_ImagingAffineNearest, RefusesADestinationThatSharesThePixelsOfTheSource)
{
    const ImageSize size{2, 2};
    Image source = counting_image(size);

    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(), make_plan(size, size, 1.0f, 0.0f, 0.0f),
                                           source),
                      StatusCode::InvalidDescription, "share pixels");
}

/** @brief Tests that a matrix with an off-diagonal term is refused. */
TEST(Small_ImagingAffineNearest, RefusesAShearedMatrix)
{
    const ImageSize size{2, 2};
    const Image source = counting_image(size);
    Image warped;
    AffinePlan sheared = make_plan(size, size, 1.0f, 0.0f, 0.0f);
    sheared.matrix[1] = 0.5f;

    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(), sheared, warped),
                      StatusCode::InvalidDescription, "off-diagonal");
}
