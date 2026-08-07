/**************************************************************************/
/* test_shape.cpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: Shape is host-lane vocabulary and must stay testable without a
// RuntimeContext, which is what keeps the unit tier fast.

#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/shape.hpp>

using SushiAI::Shape;

TEST(Unit_Shape, DefaultIsAScalarWithOneElement)
{
    const Shape scalar;
    EXPECT_EQ(scalar.rank(), 0);
    EXPECT_EQ(scalar.num_elements(), 1);
    EXPECT_EQ(scalar.to_string(), "[]");
}

TEST(Unit_Shape, ExtentsAreStoredOutermostFirst)
{
    const Shape shape{64, 784};
    EXPECT_EQ(shape.rank(), 2);
    EXPECT_EQ(shape.dim(0), 64);
    EXPECT_EQ(shape.dim(1), 784);
    EXPECT_EQ(shape.num_elements(), 64 * 784);
    EXPECT_EQ(shape.to_string(), "[64, 784]");
}

TEST(Unit_Shape, NegativeIndicesCountFromTheInnermost)
{
    const Shape shape{2, 3, 4};
    EXPECT_EQ(shape.dim(-1), 4);
    EXPECT_EQ(shape.dim(-2), 3);
    EXPECT_EQ(shape.dim(-3), 2);
}

TEST(Unit_Shape, OutOfRangeDimensionThrows)
{
    const Shape shape{2, 3};
    EXPECT_THROW((void)shape.dim(2), SushiAI::Error);
    EXPECT_THROW((void)shape.dim(-3), SushiAI::Error);
}

TEST(Unit_Shape, NonPositiveExtentIsRejectedAtConstruction)
{
    EXPECT_THROW(Shape({2, 0}), SushiAI::Error);
    EXPECT_THROW(Shape({-1}), SushiAI::Error);
}

TEST(Unit_Shape, RankBeyondTheMaximumIsRejected)
{
    const std::initializer_list<int64_t> too_many = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    ASSERT_GT(static_cast<int32_t>(too_many.size()), SushiAI::MAX_RANK);
    EXPECT_THROW((void)Shape(too_many), SushiAI::Error);
}

TEST(Unit_Shape, BuildsFromExtentsKnownOnlyAtRunTime)
{
    // The case an initializer list cannot serve: a rank that came from a file.
    const std::vector<int64_t> dims{64, 784};
    const Shape shape(SushiRuntime::span<const int64_t>(dims.data(), dims.size()));

    EXPECT_EQ(shape, (Shape{64, 784}));
    EXPECT_EQ(shape.to_string(), "[64, 784]");
}

TEST(Unit_Shape, BuildsTheScalarFromAnEmptyExtentSpan)
{
    const std::vector<int64_t> none;
    const Shape shape{SushiRuntime::span<const int64_t>(none.data(), none.size())};
    EXPECT_EQ(shape.rank(), 0);
    EXPECT_EQ(shape.num_elements(), 1);
}

TEST(Unit_Shape, RuntimeExtentsGetTheSameValidationAsAListDoes)
{
    const std::vector<int64_t> non_positive{4, 0};
    EXPECT_THROW((void)Shape(SushiRuntime::span<const int64_t>(non_positive.data(), 2)),
                 SushiAI::Error);

    const std::vector<int64_t> too_many(static_cast<std::size_t>(SushiAI::MAX_RANK) + 1, 1);
    EXPECT_THROW((void)Shape(SushiRuntime::span<const int64_t>(too_many.data(), too_many.size())),
                 SushiAI::Error);
}

TEST(Unit_Shape, EqualityComparesRankAndEveryExtent)
{
    EXPECT_EQ(Shape({4, 5}), Shape({4, 5}));
    EXPECT_NE(Shape({4, 5}), Shape({5, 4}));
    EXPECT_NE(Shape({4, 5}), Shape({4, 5, 1}));
    EXPECT_NE(Shape({4}), Shape());
}

TEST(Unit_Shape, HashSeparatesShapesThatDifferOnlyByRank)
{
    // [4, 1] and [4] hold the same element count and the same leading extent;
    // a hash that folded only the extents would collide them, and the tensor
    // pool keys its constant operands on exactly this hash.
    EXPECT_NE(Shape({4, 1}).hash(), Shape({4}).hash());
    EXPECT_EQ(Shape({4, 1}).hash(), Shape({4, 1}).hash());
}

TEST(Unit_Shape, WorksAsAnUnorderedMapKey)
{
    std::unordered_map<Shape, int> counts;
    counts[Shape({2, 2})] = 1;
    counts[Shape({2, 2})] += 1;
    counts[Shape({4})] = 7;

    EXPECT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts[Shape({2, 2})], 2);
    EXPECT_EQ(counts[Shape({4})], 7);
}
