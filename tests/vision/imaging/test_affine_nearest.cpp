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
 * @file test_affine_nearest.cpp
 * @brief Tests that the nearest warp resets its destination and fills with the plan's border.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/affine.hpp>
#include <SushiAI/vision/imaging/affine_nearest.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
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
} // namespace

/** @brief Tests that a destination of another size and channel count takes the plan's shape. */
TEST(Small_ImagingAffineNearest, ADestinationOfAnotherShapeTakesThePlanOutput)
{
    const ImageSize size{2, 3};
    const Image source = counting_image(size);
    Image warped(ImageSize{5, 5}, 3);
    warped.fill(9);

    SA_EXPECT_OK(apply_affine_nearest(source.view(), make_shift_plan(size, 0.0f, 0), warped));

    ASSERT_EQ(warped.size(), size);
    ASSERT_EQ(warped.channels(), 1);
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
