/**************************************************************************/
/* test_top_k.cpp                                                         */
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
 * @file test_top_k.cpp
 * @brief Tests the top-k selection.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/top_k.hpp>

namespace
{
    using SushiAI::Test::equal_reference;

} // namespace

/** @brief Tests that equal values are selected by ascending index and larger values first. */
TEST(Small_DetectTopK, OrdersByValueThenIndex)
{
    const std::vector<float> values{0.5f, 0.9f, 0.5f, 0.1f, 0.9f};
    std::vector<std::int64_t> indices;
    SushiAI::Vision::Detect::select_top_k(values, 4, indices);

    EXPECT_TRUE(equal_reference(indices, std::vector<std::int64_t>{1, 4, 0, 2}));
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::select_top_k(values, 6, indices), "6", "5 values");
}
