/**************************************************************************/
/* test_shape.cpp                                                         */
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
 * @file test_shape.cpp
 * @brief Tests Shape: its extents, its limits and the messages it raises.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using SushiAI::Shape;
using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueID;

/** @brief Tests that dim returns the extents outermost first. */
TEST(Small_Shape, ExtentsAreStoredOutermostFirst)
{
    const Shape shape{64, 784};
    EXPECT_EQ(shape.rank(), 2);
    EXPECT_EQ(shape.dim(0), 64);
    EXPECT_EQ(shape.dim(1), 784);
}

/** @brief Tests that the element count is the product of the extents. */
TEST(Small_Shape, TheElementCountIsTheProductOfTheExtents)
{
    EXPECT_EQ((Shape{64, 784}).num_elements(), 64 * 784);
}

/** @brief Tests that to_string prints the extents as a bracketed list, the scalar as "[]". */
TEST(Small_Shape, ToStringPrintsTheBracketedExtents)
{
    EXPECT_EQ((Shape{64, 784}).to_string(), "[64, 784]");
    EXPECT_EQ(Shape().to_string(), "[]");
}

TEST(Small_Shape, NegativeIndicesCountFromTheInnermost)
{
    const Shape shape{2, 3, 4};
    EXPECT_EQ(shape.dim(-1), 4);
    EXPECT_EQ(shape.dim(-2), 3);
    EXPECT_EQ(shape.dim(-3), 2);
}

TEST(Small_Shape, BuildsFromExtentsKnownOnlyAtRunTime)
{
    // The case an initializer list cannot serve: a rank that came from a file.
    const std::vector<int64_t> dims{64, 784};
    const Shape shape(SushiRuntime::span<const int64_t>(dims.data(), dims.size()));

    EXPECT_EQ(shape, (Shape{64, 784}));
}

TEST(Small_Shape, BuildsTheScalarFromAnEmptyExtentSpan)
{
    const std::vector<int64_t> none;
    const Shape shape{SushiRuntime::span<const int64_t>(none.data(), none.size())};
    EXPECT_EQ(shape.rank(), 0);
    EXPECT_EQ(shape.num_elements(), 1);
}

TEST(Small_Shape, MakeBuildsTheShapeOfValidRuntimeExtents)
{
    const std::vector<int64_t> dims{64, 784};

    SA_ASSERT_OK_AND_ASSIGN(const Shape shape, Shape::make(SushiRuntime::span<const int64_t>(
                                                   dims.data(), dims.size())));

    EXPECT_EQ(shape, (Shape{64, 784}));
}

TEST(Small_Shape, MakeBuildsTheScalarFromAnEmptySpan)
{
    const std::vector<int64_t> none;

    SA_ASSERT_OK_AND_ASSIGN(const Shape shape, Shape::make(SushiRuntime::span<const int64_t>(
                                                   none.data(), none.size())));

    EXPECT_EQ(shape.rank(), 0);
}

TEST(Small_Shape, MakeRefusesANonPositiveExtentWithAStatus)
{
    const std::vector<int64_t> dims{4, 0, 2};

    SA_EXPECT_FAILURE(Shape::make(SushiRuntime::span<const int64_t>(dims.data(), dims.size())),
                      StatusCode::InvalidDescription, "Shape extents must be positive", "got 0");
}

TEST(Small_Shape, MakeRefusesARankBeyondTheMaximumWithAStatus)
{
    const std::vector<int64_t> dims(static_cast<std::size_t>(SushiAI::K_MAX_RANK) + 1, 1);

    SA_EXPECT_FAILURE(Shape::make(SushiRuntime::span<const int64_t>(dims.data(), dims.size())),
                      StatusCode::InvalidDescription, "Shape rank 7", "exceeds the maximum of 6");
}

TEST(Small_Shape, EqualityComparesRankAndEveryExtent)
{
    EXPECT_EQ(Shape({4, 5}), Shape({4, 5}));
    EXPECT_NE(Shape({4, 5}), Shape({5, 4}));
    EXPECT_NE(Shape({4, 5}), Shape({4, 5, 1}));
    EXPECT_NE(Shape({4}), Shape());
}

/** @brief Tests that equal shapes hash equal. */
TEST(Small_Shape, EqualShapesHashEqual)
{
    EXPECT_EQ(Shape({4, 1}).hash(), Shape({4, 1}).hash());
}

/** @brief Tests that shapes that differ only by rank hash apart. */
TEST(Small_Shape, HashSeparatesShapesThatDifferOnlyByRank)
{
    // The tensor pool keys constant operands on this hash; see tests/core/shape/README.md.
    EXPECT_NE(Shape({4, 1}).hash(), Shape({4}).hash());
}

/**
 * @brief Tests that the hash keeps the values it has always had, computed apart from SushiAI
 *        as 64-bit FNV-1a over the rank and then each extent, each as eight bytes low first.
 */
TEST(Small_Shape, HashKeepsItsValues)
{
    EXPECT_EQ(Shape().hash(), 0xa8c7f832281a39c5ull);
    EXPECT_EQ(Shape({4, 3}).hash(), 0x3638e6130dedb860ull);
    EXPECT_EQ(Shape({2, 3, 4}).hash(), 0x359a961b832f81a3ull);
}

TEST(Small_Shape, WorksAsAnUnorderedMapKey)
{
    std::unordered_map<Shape, int> counts;
    counts[Shape({2, 2})] = 1;
    counts[Shape({2, 2})] += 1;
    counts[Shape({4})] = 7;

    EXPECT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts[Shape({2, 2})], 2);
    EXPECT_EQ(counts[Shape({4})], 7);
}

/** @brief Tests that a rank above the maximum stops the process, from a list or a span. */
TEST(Medium_Shape, ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit)
{
    // The limit in the message is SushiBLAS's MAX_RANK; see tests/core/shape/README.md.
    const std::vector<int64_t> too_many(static_cast<std::size_t>(SushiAI::K_MAX_RANK) + 1, 1);

    SA_EXPECT_FATAL((void)Shape({1, 1, 1, 1, 1, 1, 1}),
                    "Shape rank 7", "exceeds the maximum of 6");
    SA_EXPECT_FATAL((void)Shape(SushiRuntime::span<const int64_t>(too_many.data(), too_many.size())),
                    "Shape rank 7", "exceeds the maximum of 6");
}

/** @brief Tests that a non-positive extent stops the process, from a list or a span. */
TEST(Medium_Shape, ShapeNonPositiveExtentNamesTheOffendingExtent)
{
    // Shape itself refuses a zero-element tensor; see tests/core/shape/README.md.
    const std::vector<int64_t> non_positive{4, 0};

    SA_EXPECT_FATAL((void)Shape({4, 0}), "Shape extents must be positive", "got 0");
    SA_EXPECT_FATAL((void)Shape({-3}), "Shape extents must be positive", "got -3");
    SA_EXPECT_FATAL((void)Shape({-1}), "Shape extents must be positive", "got -1");
    SA_EXPECT_FATAL((void)Shape(SushiRuntime::span<const int64_t>(non_positive.data(), 2)),
                    "Shape extents must be positive", "got 0");
}

/** @brief Tests that dim stops the process for an index outside the rank. */
TEST(Medium_Shape, ShapeDimensionOutOfRangeNamesTheIndexAndTheRank)
{
    const Shape shape{4, 3};

    SA_EXPECT_FATAL((void)shape.dim(2), "Dimension 2", "rank-2 shape");
    SA_EXPECT_FATAL((void)shape.dim(-3), "Dimension -3", "rank-2 shape");
}
