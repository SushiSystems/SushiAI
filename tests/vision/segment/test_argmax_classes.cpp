/**************************************************************************/
/* test_argmax_classes.cpp                                                */
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
 * @file test_argmax_classes.cpp
 * @brief Tests the class pick of each pixel.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/segment/argmax_classes.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>

using namespace SushiAI;
using Vision::Segment::ConfusionMatrix;

namespace
{
    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

} // namespace

/** @brief Tests that the pick is the largest score of each pixel, the lowest class on a tie. */
TEST(Small_SegmentMetrics, ArgmaxPicksTheLargestScoreAndTheLowestClassOnATie)
{
    const std::vector<float> scores{0.1f, 0.9f, 0.0f, 2.0f, -1.0f, 2.0f, -3.0f, -2.0f, -1.0f};
    Classes picked(3, -1);

    Vision::Segment::argmax_classes(
        SushiRuntime::span<const float>(scores.data(), scores.size()), 3,
        SushiRuntime::span<std::int32_t>(picked.data(), picked.size()));

    EXPECT_EQ(picked, (Classes{1, 0, 2}));
}
