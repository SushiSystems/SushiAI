/**************************************************************************/
/* test_axis_translation.cpp                                              */
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
 * @file test_axis_translation.cpp
 * @brief Tests the channel-first to channel-last rewrite of axis attributes and filters.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/interchange/axis_translation.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names one axis and the position the rewrite gives it. */
    struct AxisRow
    {
        std::string_view name; /**< The row's name. */
        std::int64_t axis;     /**< The channel-first axis. */
        std::int64_t rank;     /**< The tensor's rank. */
        std::int64_t expected; /**< The channel-last position. */
    };

    /**
     * @brief Lists axes by hand. The channels (axis 1) go to the end and every axis after them
     *        moves one place down; a negative axis counts from the end first.
     */
    const AxisRow K_AXIS_ROWS[] = {
        {"BatchOfRank4", 0, 4, 0},
        {"ChannelsOfRank4", 1, 4, 3},
        {"HeightOfRank4", 2, 4, 1},
        {"WidthOfRank4", 3, 4, 2},
        {"LastOfRank4", -1, 4, 2},
        {"ThirdFromEndOfRank4", -3, 4, 3},
        {"FirstOfRank4", -4, 4, 0},
        {"ChannelsOfRank2", 1, 2, 1},
        {"ChannelsOfRank3", 1, 3, 2},
        {"LengthOfRank3", 2, 3, 1},
        {"OnlyAxisOfRank1", 0, 1, 0},
    };

    /** @brief Runs one row of K_AXIS_ROWS. */
    class Small_ChannelsLastAxis : public ::testing::TestWithParam<AxisRow>
    {

    };

    /** @brief Returns the name of the axis row @p info runs. */
    std::string axis_name(const ::testing::TestParamInfo<AxisRow>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Returns a weight [out, in, 2, 2] whose element is o * 100 + i * 10 + y * 2 + x. */
    std::vector<float> numbered_filter(std::int64_t outputs, std::int64_t inputs)
    {
        std::vector<float> weights;
        for (std::int64_t o = 0; o < outputs; ++o)
            for (std::int64_t i = 0; i < inputs; ++i)
                for (std::int64_t y = 0; y < 2; ++y)
                    for (std::int64_t x = 0; x < 2; ++x)
                        weights.push_back(static_cast<float>(o * 100 + i * 10 + y * 2 + x));
        return weights;
    }

    /** @brief Returns a view of @p values. */
    SushiRuntime::span<const float> view_of(const std::vector<float>& values)
    {
        return SushiRuntime::span<const float>(values.data(), values.size());
    }

    using Integers = std::vector<std::int64_t>;
}

/** @brief Tests that each row moves its axis to the place the layout rule gives. */
TEST_P(Small_ChannelsLastAxis, MovesTheAxisAsTheLayoutRuleSays)
{
    const AxisRow& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t moved,
                            Interchange::channels_last_axis(row.axis, row.rank));

    EXPECT_EQ(moved, row.expected);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ChannelsLastAxis, ::testing::ValuesIn(K_AXIS_ROWS), axis_name);

/** @brief Tests that an axis outside [-rank, rank) is refused. */
TEST(Small_AxisTranslation, RefusesAnAxisOutOfRange)
{
    SA_EXPECT_FAILURE(Interchange::channels_last_axis(4, 4), StatusCode::InvalidDescription,
                      "axis 4", "rank 4");
    SA_EXPECT_FAILURE(Interchange::channels_last_axis(-5, 4), StatusCode::InvalidDescription,
                      "axis -5");
}

/** @brief Tests that Concat, Softmax, Slice and the reductions rewrite their axes by value. */
TEST(Small_AxisTranslation, RewritesTheAxesOfConcatSoftmaxSliceAndReduce)
{
    SA_ASSERT_OK_AND_ASSIGN(const Integers concat,
                            Interchange::translate_axis_attribute("Concat", "axis", {1}, 4));
    SA_ASSERT_OK_AND_ASSIGN(const Integers softmax,
                            Interchange::translate_axis_attribute("Softmax", "axis", {-1}, 4));
    SA_ASSERT_OK_AND_ASSIGN(const Integers slice,
                            Interchange::translate_axis_attribute("Slice", "axes", {2, 3}, 4));
    SA_ASSERT_OK_AND_ASSIGN(const Integers reduce,
                            Interchange::translate_axis_attribute("ReduceMean", "axes", {2, 3}, 4));

    EXPECT_EQ(concat, (Integers{3}));
    EXPECT_EQ(softmax, (Integers{2}));
    EXPECT_EQ(slice, (Integers{1, 2}));
    EXPECT_EQ(reduce, (Integers{1, 2}));
}

/** @brief Tests that an attribute that holds no axis passes through. */
TEST(Small_AxisTranslation, LeavesAttributesThatHoldNoAxisAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const Integers strides,
                            Interchange::translate_axis_attribute("Conv", "strides", {2, 2}, 4));

    EXPECT_EQ(strides, (Integers{2, 2}));
    EXPECT_EQ(Interchange::axis_attribute_kind("Conv", "strides"),
              Interchange::AxisAttributeKind::None);
    EXPECT_EQ(Interchange::axis_attribute_kind("Concat", "axis"),
              Interchange::AxisAttributeKind::Axis);
}

/**
 * @brief Tests the Transpose permutation by value. The identity stays the identity; the
 *        swap of the two spatial axes of the channel-first tensor is the swap of the two
 *        middle axes of the channel-last one; the move of the channels to the end is itself.
 */
TEST(Small_AxisTranslation, RewritesATransposePermutation)
{
    SA_ASSERT_OK_AND_ASSIGN(const Integers identity,
                            Interchange::channels_last_permutation({0, 1, 2, 3}));
    SA_ASSERT_OK_AND_ASSIGN(const Integers swap_spatial,
                            Interchange::channels_last_permutation({0, 1, 3, 2}));
    SA_ASSERT_OK_AND_ASSIGN(const Integers channels_last,
                            Interchange::channels_last_permutation({0, 2, 3, 1}));

    EXPECT_EQ(identity, (Integers{0, 1, 2, 3}));
    EXPECT_EQ(swap_spatial, (Integers{0, 2, 1, 3}));
    EXPECT_EQ(channels_last, (Integers{0, 2, 3, 1}));
}

/** @brief Tests that a list that is not a permutation is refused. */
TEST(Small_AxisTranslation, RefusesAPermutationThatRepeatsOrOverflows)
{
    SA_EXPECT_FAILURE(Interchange::channels_last_permutation({0, 1, 1, 3}),
                      StatusCode::InvalidDescription, "twice");
    SA_EXPECT_FAILURE(Interchange::channels_last_permutation({0, 1, 2, 4}),
                      StatusCode::InvalidDescription, "entry 4", "rank 4");
}

/** @brief Tests that a target shape moves with its axes and keeps its 0 and -1. */
TEST(Small_AxisTranslation, RewritesAReshapeTarget)
{
    SA_ASSERT_OK_AND_ASSIGN(const Integers shape,
                            Interchange::channels_last_shape({0, 3, 8, -1}));
    SA_ASSERT_OK_AND_ASSIGN(const Integers through_dispatch,
                            Interchange::translate_axis_attribute("Reshape", "shape",
                                                                  {1, 3, 8, 8}, 4));

    EXPECT_EQ(shape, (Integers{0, 8, -1, 3}));
    EXPECT_EQ(through_dispatch, (Integers{1, 8, 8, 3}));
}

/** @brief Tests that a Flatten axis rewrites where it keeps its meaning, else is refused. */
TEST(Small_AxisTranslation, RewritesAFlattenAxisOnlyWhereItKeepsItsMeaning)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t at_batch,
                            Interchange::channels_last_flatten_axis(0, 4));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t after_batch,
                            Interchange::channels_last_flatten_axis(1, 4));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t from_end,
                            Interchange::channels_last_flatten_axis(-3, 4));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t past_all,
                            Interchange::channels_last_flatten_axis(4, 4));

    EXPECT_EQ(at_batch, 0);
    EXPECT_EQ(after_batch, 1);
    EXPECT_EQ(from_end, 1);
    EXPECT_EQ(past_all, 4);
    SA_EXPECT_FAILURE(Interchange::channels_last_flatten_axis(2, 4), StatusCode::Unsupported,
                      "axis 2", "splits the channels");
}

/** @brief Tests that [out, in, kh, kw] becomes [out, kh, kw, in] element by element. */
TEST(Small_AxisTranslation, PermutesAnOIHWWeightToTheRecordLayout)
{
    const std::vector<float> weights = numbered_filter(2, 3);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> record,
                            Interchange::filter_to_record_layout(view_of(weights), {2, 3, 2, 2},
                                                                 false));
    SA_ASSERT_OK_AND_ASSIGN(const Integers dimensions,
                            Interchange::filter_record_dimensions({2, 3, 2, 2}, false));

    EXPECT_EQ(dimensions, (Integers{2, 2, 2, 3}));
    ASSERT_EQ(record.size(), weights.size());
    for (std::int64_t o = 0; o < 2; ++o)
        for (std::int64_t y = 0; y < 2; ++y)
            for (std::int64_t x = 0; x < 2; ++x)
                for (std::int64_t i = 0; i < 3; ++i)
                {
                    const std::size_t at = static_cast<std::size_t>(((o * 2 + y) * 2 + x) * 3 + i);
                    EXPECT_EQ(record[at], static_cast<float>(o * 100 + i * 10 + y * 2 + x))
                        << "o " << o << " y " << y << " x " << x << " i " << i;
                }
}

/** @brief Tests that [C, 1, kh, kw] becomes [kh, kw, C]. */
TEST(Small_AxisTranslation, PermutesADepthwiseWeightToKernelHeightWidthChannel)
{
    const std::vector<float> weights = numbered_filter(3, 1);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> record,
                            Interchange::filter_to_record_layout(view_of(weights), {3, 1, 2, 2},
                                                                 true));
    SA_ASSERT_OK_AND_ASSIGN(const Integers dimensions,
                            Interchange::filter_record_dimensions({3, 1, 2, 2}, true));

    EXPECT_EQ(dimensions, (Integers{2, 2, 3}));
    ASSERT_EQ(record.size(), weights.size());
    for (std::int64_t c = 0; c < 3; ++c)
        for (std::int64_t y = 0; y < 2; ++y)
            for (std::int64_t x = 0; x < 2; ++x)
            {
                const std::size_t at = static_cast<std::size_t>((y * 2 + x) * 3 + c);
                EXPECT_EQ(record[at], static_cast<float>(c * 100 + y * 2 + x));
            }
}

/** @brief Tests that a weight of the wrong rank, extent, depthwise shape or count is refused. */
TEST(Small_AxisTranslation, RefusesAFilterThatIsNotAConvolutionWeight)
{
    const std::vector<float> five(5, 0.0f);

    SA_EXPECT_FAILURE(Interchange::filter_record_dimensions({2, 3, 2}, false),
                      StatusCode::InvalidDescription, "4 extents", "this one has 3");
    SA_EXPECT_FAILURE(Interchange::filter_record_dimensions({2, 0, 2, 2}, false),
                      StatusCode::InvalidDescription, "extent 0");
    SA_EXPECT_FAILURE(Interchange::filter_record_dimensions({2, 2, 2, 2}, true),
                      StatusCode::InvalidDescription, "depthwise", "reads 2");
    SA_EXPECT_FAILURE(Interchange::filter_to_record_layout(view_of(five), {2, 1, 2, 2}, false),
                      StatusCode::InvalidDescription, "5 floats", "need 8");
}

/**
 * @brief Tests that every unit row of a [C, H, W] inner-product weight is moved to [H, W, C].
 *
 * Two units over 2 channels and a 2 by 3 image: the input at channel c, row y and column x is
 * numbered u * 100 + c * 10 + y * 3 + x for the unit u, and the moved weight holds it at
 * u * 12 + (y * 3 + x) * 2 + c.
 */
TEST(Small_AxisTranslation, MovesTheInputsOfAnInnerProductWeightToChannelsLast)
{
    std::vector<float> weights;
    for (std::int64_t u = 0; u < 2; ++u)
        for (std::int64_t c = 0; c < 2; ++c)
            for (std::int64_t y = 0; y < 2; ++y)
                for (std::int64_t x = 0; x < 3; ++x)
                    weights.push_back(static_cast<float>(u * 100 + c * 10 + y * 3 + x));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<float> moved,
        Interchange::flatten_weights_to_channels_last(view_of(weights), 2, 2, 2, 3));

    ASSERT_EQ(moved.size(), weights.size());
    for (std::int64_t u = 0; u < 2; ++u)
        for (std::int64_t c = 0; c < 2; ++c)
            for (std::int64_t y = 0; y < 2; ++y)
                for (std::int64_t x = 0; x < 3; ++x)
                {
                    const std::size_t at = static_cast<std::size_t>(u * 12 + (y * 3 + x) * 2 + c);
                    EXPECT_EQ(moved[at], static_cast<float>(u * 100 + c * 10 + y * 3 + x))
                        << "unit " << u << " channel " << c << " y " << y << " x " << x;
                }
}

/** @brief Tests that a weight of one channel is left in place, since the order is the same. */
TEST(Small_AxisTranslation, AnInnerProductWeightOfOneChannelIsUnchanged)
{
    const std::vector<float> weights = {1, 2, 3, 4, 5, 6};

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<float> moved,
        Interchange::flatten_weights_to_channels_last(view_of(weights), 1, 1, 2, 3));

    EXPECT_EQ(moved, weights);
}

/** @brief Tests that a non-positive extent and a wrong float count are refused. */
TEST(Small_AxisTranslation, RefusesAnInnerProductWeightThatDoesNotFitItsExtents)
{
    const std::vector<float> five(5, 0.0f);

    SA_EXPECT_FAILURE(Interchange::flatten_weights_to_channels_last(view_of(five), 1, 0, 1, 5),
                      StatusCode::InvalidDescription, "positive extents");
    SA_EXPECT_FAILURE(Interchange::flatten_weights_to_channels_last(view_of(five), 2, 1, 1, 3),
                      StatusCode::InvalidDescription, "5 floats", "need 6");
}
