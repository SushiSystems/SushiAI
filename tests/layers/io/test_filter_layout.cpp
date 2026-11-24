/**************************************************************************/
/* test_filter_layout.cpp                                                 */
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
 * @file test_filter_layout.cpp
 * @brief Tests the moves of convolution weights between the kernel-first and kernel-last layouts.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/io/filter_layout.hpp>

using namespace SushiAI;

/** @brief Tests that a [out, in, kh, kw] weight is copied as [out, kh, kw, in]. */
TEST(Small_FilterLayout, ADenseWeightRunsOutputTapInput)
{
    // Element (o, i, t) of a [2, 3, 1, 2] weight holds 6 o + 2 i + t; the copy runs o, t, i.
    const std::vector<float> source = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<float> destination(source.size());

    IO::copy_kernel_last(source.data(), IO::FilterExtents{2, 3, 1, 2}, destination.data());

    const std::vector<float> expected = {0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11};
    EXPECT_EQ(destination, expected);
}

/** @brief Tests that a [C, 1, kh, kw] depthwise weight is copied as [kh, kw, C]. */
TEST(Small_FilterLayout, ADepthwiseWeightRunsTapChannel)
{
    // Element (c, t) of a [3, 1, 2, 1] weight holds 2 c + t; the copy runs t, c.
    const std::vector<float> source = {0, 1, 2, 3, 4, 5};
    std::vector<float> destination(source.size());

    IO::copy_depthwise_kernel_last(source.data(), IO::FilterExtents{3, 1, 2, 1},
                                   destination.data());

    const std::vector<float> expected = {0, 2, 4, 1, 3, 5};
    EXPECT_EQ(destination, expected);
}

/** @brief Tests that a seeded [2, 3, 3, 4] weight moved last then first comes back whole. */
TEST(Small_FilterLayout, ADenseWeightMovedBothWaysComesBack)
{
    std::vector<float> source(2 * 3 * 3 * 4);
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>((i * 37u) % 101u) - 50.0f;
    std::vector<float> last(source.size());
    std::vector<float> first(source.size());
    const IO::FilterExtents extents{2, 4, 3, 3};

    IO::copy_kernel_last(source.data(), extents, last.data());
    IO::copy_kernel_first(last.data(), extents, first.data());

    EXPECT_EQ(first, source);
}

/** @brief Tests that a seeded [3, 3, 5] depthwise weight moved first then last comes back. */
TEST(Small_FilterLayout, ADepthwiseWeightMovedBothWaysComesBack)
{
    std::vector<float> source(3 * 3 * 5);
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>((i * 53u) % 97u) - 40.0f;
    std::vector<float> first(source.size());
    std::vector<float> last(source.size());
    const IO::FilterExtents extents{5, 1, 3, 3};

    IO::copy_depthwise_kernel_first(source.data(), extents, first.data());
    IO::copy_depthwise_kernel_last(first.data(), extents, last.data());

    EXPECT_EQ(last, source);
}
