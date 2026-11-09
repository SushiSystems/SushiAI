/**************************************************************************/
/* test_feed.cpp                                                          */
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
 * @file test_feed.cpp
 * @brief Tests check_feed, which compares a batch tensor with the leaf a step copies it into.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/feed.hpp>

using namespace SushiAI;

namespace
{
    /**
     * @brief Returns a tensor header of @p extents and @p dtype that owns no storage.
     *
     * check_feed reads the rank, the extents and the element type only, so no device is needed.
     */
    SushiBLAS::Tensor header_of(std::initializer_list<std::int64_t> extents, DType dtype)
    {
        SushiBLAS::Tensor tensor{};
        tensor.rank = static_cast<std::int32_t>(extents.size());
        tensor.dtype = dtype;
        std::size_t axis = 0;
        for (const std::int64_t extent : extents)
            tensor.shape[axis++] = extent;
        return tensor;
    }

    /** @brief Holds a source shape, a leaf shape and the words of the refusal naming both. */
    struct ShapeRow
    {
        std::string name;                       /**< The row's name. */
        std::vector<std::int64_t> source;       /**< The extents of the batch tensor. */
        std::string first;                      /**< The source shape the message holds. */
    };

    /** @brief Runs the refusal of a source of another shape than the [4, 6] leaf. */
    class Small_FeedShape : public ::testing::TestWithParam<ShapeRow>
    {

    };
} // namespace

/** @brief Tests that tensors of one rank, one set of extents and one element type pass. */
TEST(Small_Feed, MatchingTensorsAreAccepted)
{
    SA_EXPECT_OK(Train::check_feed(header_of({4, 6}, DType::Float32),
                                   header_of({4, 6}, DType::Float32), "inputs"));
}

/** @brief Tests that other extents or another rank are refused with both shapes named. */
TEST_P(Small_FeedShape, AnotherShapeIsIncompatibleAndBothShapesAreNamed)
{
    const ShapeRow& row = GetParam();
    SushiBLAS::Tensor source = header_of({}, DType::Float32);
    source.rank = static_cast<std::int32_t>(row.source.size());
    for (std::size_t axis = 0; axis < row.source.size(); ++axis)
        source.shape[axis] = row.source[axis];

    SA_EXPECT_FAILURE(Train::check_feed(source, header_of({4, 6}, DType::Float32), "inputs"),
                      StatusCode::Incompatible, "inputs", row.first, "[4, 6]");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_FeedShape,
    ::testing::Values(ShapeRow{"AnotherExtent", {4, 5}, "[4, 5]"},
                      ShapeRow{"AnotherRankOfTheSameCount", {24}, "[24]"}),
    [](const ::testing::TestParamInfo<ShapeRow>& info) { return info.param.name; });

/** @brief Tests that equal shapes of two element types are refused with both types named. */
TEST(Small_Feed, AnotherElementTypeIsIncompatibleAndBothTypesAreNamed)
{
    SA_EXPECT_FAILURE(Train::check_feed(header_of({4, 3}, DType::Int32),
                                        header_of({4, 3}, DType::Float32), "targets"),
                      StatusCode::Incompatible, "targets", "int32", "float32");
}

/** @brief Tests that an extent no Shape can hold is reported, not thrown. */
TEST(Small_Feed, AnExtentNoShapeHoldsIsAnInvalidDescription)
{
    SA_EXPECT_FAILURE(Train::check_feed(header_of({4, 0}, DType::Float32),
                                        header_of({4, 6}, DType::Float32), "inputs"),
                      StatusCode::InvalidDescription, "positive");
}
