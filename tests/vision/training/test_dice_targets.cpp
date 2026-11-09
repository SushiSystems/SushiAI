/**************************************************************************/
/* test_dice_targets.cpp                                                  */
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
 * @file test_dice_targets.cpp
 * @brief Tests the Dice leaves of a mask with no kept pixel and of float64 leaves.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/training/dice_targets.hpp>

using namespace SushiAI;
using Vision::Training::DiceLeaves;
using Vision::Training::fill_dice_targets;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the pixels of the masks, R. */
    constexpr std::int64_t K_PIXELS = 4;

    /** @brief Holds the class count, C. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the value a leaf starts at, which no fill of these cases leaves. */
    constexpr double K_UNTOUCHED = -7.0;

    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

    /** @brief Provides an engine and the five leaves of one Dice fill. */
    class Medium_VisionTrainingDiceTargets : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns an INT32 tensor of @p shape holding @p values. */
            SushiBLAS::Tensor make_ids(const Shape& shape, const Classes& values)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(*engine, shape, DType::Int32);
                EXPECT_TRUE(made.status().is_ok());

                std::int32_t* const ids = made.value().data_as<std::int32_t>();
                for (std::size_t i = 0; i < values.size(); ++i)
                    ids[i] = values[i];

                return std::move(made).value();
            }

            /** @brief Returns a real tensor of @p shape and @p dtype at K_UNTOUCHED. */
            SushiBLAS::Tensor make_reals(const Shape& shape, DType dtype)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(*engine, shape, dtype);
                EXPECT_TRUE(made.status().is_ok());

                const std::vector<double> values(
                    static_cast<std::size_t>(made.value().num_elements), K_UNTOUCHED);
                SaTest::write_tensor(made.value(), values);

                return std::move(made).value();
            }

            /** @brief Returns the elements of the INT32 tensor @p tensor. */
            static Classes read_ids(const SushiBLAS::Tensor& tensor)
            {
                const std::int32_t* const ids = tensor.data_as<std::int32_t>();

                return Classes(ids, ids + tensor.num_elements);
            }

            /** @brief Expects the elements of @p tensor to equal @p expected; all are exact. */
            static void expect_reals(const SushiBLAS::Tensor& tensor,
                                     const std::vector<double>& expected)
            {
                EXPECT_EQ(SaTest::read_tensor(tensor), expected);
            }
    };

    /** @brief Names a mask, its ignore index, the leaf type and the leaves the fill must write. */
    struct DiceRow
    {
        const char* name;                  /**< Names the row in the test name. */
        DType dtype;                       /**< The element type of the real leaves. */
        Classes mask;                      /**< The class of each pixel. */
        std::int32_t ignore;               /**< The mask value of a pixel of no class. */
        Classes score_indices;             /**< The score each pixel points at. */
        Classes class_indices;             /**< The class each pixel points at. */
        std::vector<double> kept;          /**< One for a kept pixel, zero for an ignored one. */
        std::vector<double> class_pixels;  /**< The kept pixels of each class. */
        std::vector<double> class_weights; /**< One over the present classes, zero for the rest. */
    };

    /** @brief Provides the masks that count the kept pixels of each class. */
    class Medium_VisionTrainingDiceTargetsCounts
        : public Medium_VisionTrainingDiceTargets,
          public ::testing::WithParamInterface<DiceRow>
    {

    };
} // namespace

/** @brief Tests that a mask of ignored pixels alone gives zero in every real leaf. */
TEST_F(Medium_VisionTrainingDiceTargets, AMaskWithNoKeptPixelGivesZeros)
{
    const SushiBLAS::Tensor mask = make_ids(Shape{K_PIXELS}, Classes(4, 255));
    SushiBLAS::Tensor score_indices = make_ids(Shape{K_PIXELS}, Classes(4, -7));
    SushiBLAS::Tensor class_indices = make_ids(Shape{K_PIXELS}, Classes(4, -7));
    SushiBLAS::Tensor kept = make_reals(Shape{K_PIXELS, 1}, DType::Float32);
    SushiBLAS::Tensor class_pixels = make_reals(Shape{K_CLASSES}, DType::Float32);
    SushiBLAS::Tensor class_weights = make_reals(Shape{K_CLASSES}, DType::Float32);

    SA_EXPECT_OK(fill_dice_targets(
        mask, K_CLASSES, 255,
        DiceLeaves{&score_indices, &class_indices, &kept, &class_pixels, &class_weights}));

    // Each ignored pixel r points at score r * C and at class 0.
    EXPECT_EQ(read_ids(score_indices), (Classes{0, 3, 6, 9}));
    EXPECT_EQ(read_ids(class_indices), Classes(4, 0));
    expect_reals(kept, {0.0, 0.0, 0.0, 0.0});
    expect_reals(class_pixels, {0.0, 0.0, 0.0});
    expect_reals(class_weights, {0.0, 0.0, 0.0});
}

/** @brief Tests that the five Dice leaves count the kept pixels of each class. */
TEST_P(Medium_VisionTrainingDiceTargetsCounts, TheDiceLeavesCountTheKeptPixelsOfEachClass)
{
    const DiceRow& row = GetParam();
    const Shape pixels{static_cast<std::int64_t>(row.mask.size())};
    const SushiBLAS::Tensor mask = make_ids(pixels, row.mask);
    SushiBLAS::Tensor score_indices = make_ids(pixels, Classes(row.mask.size(), -7));
    SushiBLAS::Tensor class_indices = make_ids(pixels, Classes(row.mask.size(), -7));
    SushiBLAS::Tensor kept = make_reals(Shape{pixels[0], 1}, row.dtype);
    SushiBLAS::Tensor class_pixels = make_reals(Shape{K_CLASSES}, row.dtype);
    SushiBLAS::Tensor class_weights = make_reals(Shape{K_CLASSES}, row.dtype);

    SA_EXPECT_OK(fill_dice_targets(
        mask, K_CLASSES, row.ignore,
        DiceLeaves{&score_indices, &class_indices, &kept, &class_pixels, &class_weights}));

    EXPECT_EQ(read_ids(score_indices), row.score_indices);
    EXPECT_EQ(read_ids(class_indices), row.class_indices);
    expect_reals(kept, row.kept);
    expect_reals(class_pixels, row.class_pixels);
    expect_reals(class_weights, row.class_weights);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_VisionTrainingDiceTargetsCounts,
    ::testing::Values(
        DiceRow{"FloatSixtyFourIgnoreMinusOne", DType::Float64, {1, 1, 0, -1}, -1,
                {1, 4, 6, 9}, {1, 1, 0, 0}, {1.0, 1.0, 1.0, 0.0}, {1.0, 2.0, 0.0},
                {0.5, 0.5, 0.0}},
        DiceRow{"FloatThirtyTwoIgnoreTwoFiftyFive", DType::Float32, {0, 2, 255, 0, 2, 0}, 255,
                {0, 5, 6, 9, 14, 15}, {0, 2, 0, 0, 2, 0}, {1.0, 1.0, 0.0, 1.0, 1.0, 1.0},
                {3.0, 0.0, 2.0}, {0.5, 0.0, 0.5}}),
    [](const ::testing::TestParamInfo<DiceRow>& info) { return std::string(info.param.name); });
