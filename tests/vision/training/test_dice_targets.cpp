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
 * @file test_dice_targets.cpp
 * @brief Tests the Dice leaves of a mask with no kept pixel and of float64 leaves.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
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

    /** @brief Bounds the absolute error of a count or of 1 / n for n below ten. */
    constexpr double K_TOLERANCE = 1e-7;

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

            /** @brief Expects element i of @p tensor to be @p expected[i] within K_TOLERANCE. */
            static void expect_reals(const SushiBLAS::Tensor& tensor,
                                     const std::vector<double>& expected)
            {
                const std::vector<double> found = SaTest::read_tensor(tensor);
                ASSERT_EQ(found.size(), expected.size());
                for (std::size_t i = 0; i < found.size(); ++i)
                    EXPECT_NEAR(found[i], expected[i], K_TOLERANCE) << "element " << i;
            }
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

/** @brief Tests that float64 leaves and an ignore index of -1 give the counts float32 gives. */
TEST_F(Medium_VisionTrainingDiceTargets, FloatSixtyFourLeavesHoldTheSameCounts)
{
    // Class 1 holds two kept pixels and class 0 one; class 2 is absent and pixel 3 ignored.
    const SushiBLAS::Tensor mask = make_ids(Shape{K_PIXELS}, {1, 1, 0, -1});
    SushiBLAS::Tensor score_indices = make_ids(Shape{K_PIXELS}, Classes(4, -7));
    SushiBLAS::Tensor class_indices = make_ids(Shape{K_PIXELS}, Classes(4, -7));
    SushiBLAS::Tensor kept = make_reals(Shape{K_PIXELS, 1}, DType::Float64);
    SushiBLAS::Tensor class_pixels = make_reals(Shape{K_CLASSES}, DType::Float64);
    SushiBLAS::Tensor class_weights = make_reals(Shape{K_CLASSES}, DType::Float64);

    SA_EXPECT_OK(fill_dice_targets(
        mask, K_CLASSES, -1,
        DiceLeaves{&score_indices, &class_indices, &kept, &class_pixels, &class_weights}));

    EXPECT_EQ(read_ids(score_indices), (Classes{1, 4, 6, 9}));
    EXPECT_EQ(read_ids(class_indices), (Classes{1, 1, 0, 0}));
    expect_reals(kept, {1.0, 1.0, 1.0, 0.0});
    expect_reals(class_pixels, {1.0, 2.0, 0.0});
    expect_reals(class_weights, {0.5, 0.5, 0.0});
}
