/**************************************************************************/
/* test_pixel_targets.cpp                                                 */
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
 * @file test_pixel_targets.cpp
 * @brief Tests the label check and the leaf fills of the two per-pixel objectives.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/training/dice_targets.hpp>
#include <SushiAI/vision/training/pixel_labels.hpp>
#include <SushiAI/vision/training/pixel_targets.hpp>

using namespace SushiAI;
using Vision::Training::DiceLeaves;
using Vision::Training::fill_dice_targets;
using Vision::Training::fill_pixel_targets;
using Vision::Training::validate_pixel_labels;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the pixels of the mask the cases fill from, R. */
    constexpr std::int64_t K_PIXELS = 6;

    /** @brief Holds the class count, C. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the mask value of a pixel of no class. */
    constexpr std::int32_t K_IGNORE = 255;

    /** @brief Holds the value a leaf starts at, which no fill of these cases writes. */
    constexpr double K_UNTOUCHED = -7.0;

    /** @brief Bounds the absolute error of a float32 holding 1 / n for n below ten. */
    constexpr double K_TOLERANCE = 1e-7;

    /** @brief Names a list of class indices, one per pixel. */
    using Classes = std::vector<std::int32_t>;

    /** @brief Provides an engine, masks and the leaves they fill. */
    class Medium_VisionTrainingPixelTargets : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns an INT32 tensor of @p shape holding @p values. */
            SushiBLAS::Tensor make_ids(const Shape& shape, const Classes& values)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(*engine, shape, DType::Int32);
                EXPECT_TRUE(made.status().is_ok());
                EXPECT_EQ(static_cast<std::size_t>(made.value().num_elements), values.size());

                std::int32_t* const ids = made.value().data_as<std::int32_t>();
                for (std::size_t i = 0; i < values.size(); ++i)
                    ids[i] = values[i];

                return std::move(made).value();
            }

            /** @brief Returns a float32 tensor of @p shape with every element at K_UNTOUCHED. */
            SushiBLAS::Tensor make_reals(const Shape& shape)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(*engine, shape, DType::Float32);
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

/** @brief Tests that the check passes classes and the ignore index and names a bad pixel. */
TEST_F(Medium_VisionTrainingPixelTargets, TheLabelCheckNamesThePixelOutOfBounds)
{
    const SushiBLAS::Tensor good = make_ids(Shape{2, 3}, {0, 2, K_IGNORE, 1, 1, 0});
    const SushiBLAS::Tensor high = make_ids(Shape{2, 3}, {0, 2, K_IGNORE, 3, 1, 0});
    const SushiBLAS::Tensor negative = make_ids(Shape{2, 3}, {0, 2, K_IGNORE, 1, -1, 0});
    const SushiBLAS::Tensor reals = make_reals(Shape{2, 3});

    SA_EXPECT_OK(validate_pixel_labels(good, K_PIXELS, K_CLASSES, K_IGNORE));
    SA_EXPECT_FAILURE(validate_pixel_labels(high, K_PIXELS, K_CLASSES, K_IGNORE),
                      StatusCode::Incompatible, "holds 3 at pixel 3", "below 3", "255");
    SA_EXPECT_FAILURE(validate_pixel_labels(negative, K_PIXELS, K_CLASSES, K_IGNORE),
                      StatusCode::Incompatible, "holds -1 at pixel 4");
    SA_EXPECT_FAILURE(validate_pixel_labels(good, K_PIXELS + 1, K_CLASSES, K_IGNORE),
                      StatusCode::Incompatible, "6 pixels", "expects 7");
    SA_EXPECT_FAILURE(validate_pixel_labels(reals, K_PIXELS, K_CLASSES, K_IGNORE),
                      StatusCode::Incompatible, "mask is");
}

/** @brief Tests that pixel r of class l indexes r * C + l and weighs 1 over the kept count. */
TEST_F(Medium_VisionTrainingPixelTargets, AKeptPixelIndexesItsClassAndSharesTheWeight)
{
    const SushiBLAS::Tensor mask = make_ids(Shape{2, 3}, {0, 2, K_IGNORE, 1, 1, K_IGNORE});
    SushiBLAS::Tensor indices = make_ids(Shape{K_PIXELS}, Classes(6, -7));
    SushiBLAS::Tensor weights = make_reals(Shape{K_PIXELS, 1});

    SA_EXPECT_OK(fill_pixel_targets(mask, K_CLASSES, K_IGNORE, indices, weights));

    // An ignored pixel points at the first score of its own row, with a weight of zero.
    EXPECT_EQ(read_ids(indices), (Classes{0, 5, 6, 10, 13, 15}));
    expect_reals(weights, {0.25, 0.25, 0.0, 0.25, 0.25, 0.0});
}

/** @brief Tests that a mask of ignored pixels alone gives weights of zero. */
TEST_F(Medium_VisionTrainingPixelTargets, AMaskWithNoKeptPixelGivesZeroWeights)
{
    const SushiBLAS::Tensor mask = make_ids(Shape{K_PIXELS}, Classes(6, K_IGNORE));
    SushiBLAS::Tensor indices = make_ids(Shape{K_PIXELS}, Classes(6, -7));
    SushiBLAS::Tensor weights = make_reals(Shape{K_PIXELS, 1});

    SA_EXPECT_OK(fill_pixel_targets(mask, K_CLASSES, K_IGNORE, indices, weights));

    EXPECT_EQ(read_ids(indices), (Classes{0, 3, 6, 9, 12, 15}));
    expect_reals(weights, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}

/** @brief Tests that a refused cross-entropy fill leaves both leaves as they were. */
TEST_F(Medium_VisionTrainingPixelTargets, ARefusedPixelFillChangesNothing)
{
    const SushiBLAS::Tensor bad = make_ids(Shape{K_PIXELS}, {0, 2, 1, 1, 9, 0});
    const SushiBLAS::Tensor good = make_ids(Shape{K_PIXELS}, {0, 2, 1, 1, 0, 0});
    SushiBLAS::Tensor indices = make_ids(Shape{K_PIXELS}, Classes(6, -7));
    SushiBLAS::Tensor weights = make_reals(Shape{K_PIXELS, 1});
    SushiBLAS::Tensor short_weights = make_reals(Shape{K_PIXELS - 1, 1});

    SA_EXPECT_FAILURE(fill_pixel_targets(bad, K_CLASSES, K_IGNORE, indices, weights),
                      StatusCode::Incompatible, "holds 9 at pixel 4");
    SA_EXPECT_FAILURE(fill_pixel_targets(good, K_CLASSES, K_IGNORE, indices, short_weights),
                      StatusCode::Incompatible, "6 pixel indices", "5 pixel weights");
    SA_EXPECT_FAILURE(fill_pixel_targets(good, K_CLASSES, K_IGNORE, weights, weights),
                      StatusCode::Incompatible, "pixel indices are");

    EXPECT_EQ(read_ids(indices), Classes(6, -7));
    expect_reals(weights, std::vector<double>(6, K_UNTOUCHED));
}

/** @brief Tests the five Dice leaves of a mask with one absent class and one ignored pixel. */
TEST_F(Medium_VisionTrainingPixelTargets, TheDiceLeavesCountTheKeptPixelsOfEachClass)
{
    // Classes 0 and 2 are present, with three and two kept pixels; class 1 is absent.
    const SushiBLAS::Tensor mask = make_ids(Shape{K_PIXELS}, {0, 2, K_IGNORE, 0, 2, 0});
    SushiBLAS::Tensor score_indices = make_ids(Shape{K_PIXELS}, Classes(6, -7));
    SushiBLAS::Tensor class_indices = make_ids(Shape{K_PIXELS}, Classes(6, -7));
    SushiBLAS::Tensor kept = make_reals(Shape{K_PIXELS, 1});
    SushiBLAS::Tensor class_pixels = make_reals(Shape{K_CLASSES});
    SushiBLAS::Tensor class_weights = make_reals(Shape{K_CLASSES});
    const DiceLeaves leaves{&score_indices, &class_indices, &kept, &class_pixels,
                            &class_weights};

    SA_EXPECT_OK(fill_dice_targets(mask, K_CLASSES, K_IGNORE, leaves));

    // The ignored pixel 2 points at the first score of its row and at class 0, kept 0.
    EXPECT_EQ(read_ids(score_indices), (Classes{0, 5, 6, 9, 14, 15}));
    EXPECT_EQ(read_ids(class_indices), (Classes{0, 2, 0, 0, 2, 0}));
    expect_reals(kept, {1.0, 1.0, 0.0, 1.0, 1.0, 1.0});
    expect_reals(class_pixels, {3.0, 0.0, 2.0});
    expect_reals(class_weights, {0.5, 0.0, 0.5});
}

/** @brief Tests that a refused Dice fill leaves the five leaves as they were. */
TEST_F(Medium_VisionTrainingPixelTargets, ARefusedDiceFillChangesNothing)
{
    const SushiBLAS::Tensor bad = make_ids(Shape{K_PIXELS}, {0, 2, 1, 1, 3, 0});
    const SushiBLAS::Tensor good = make_ids(Shape{K_PIXELS}, {0, 2, 1, 1, 0, 0});
    SushiBLAS::Tensor score_indices = make_ids(Shape{K_PIXELS}, Classes(6, -7));
    SushiBLAS::Tensor class_indices = make_ids(Shape{K_PIXELS}, Classes(6, -7));
    SushiBLAS::Tensor kept = make_reals(Shape{K_PIXELS, 1});
    SushiBLAS::Tensor class_pixels = make_reals(Shape{K_CLASSES});
    SushiBLAS::Tensor class_weights = make_reals(Shape{K_CLASSES});
    SushiBLAS::Tensor short_weights = make_reals(Shape{K_CLASSES - 1});
    SushiBLAS::Tensor real_indices = make_reals(Shape{K_PIXELS});

    SA_EXPECT_FAILURE(
        fill_dice_targets(bad, K_CLASSES, K_IGNORE,
                          DiceLeaves{&score_indices, &class_indices, &kept, &class_pixels,
                                     &class_weights}),
        StatusCode::Incompatible, "holds 3 at pixel 4");
    SA_EXPECT_FAILURE(
        fill_dice_targets(good, K_CLASSES, K_IGNORE,
                          DiceLeaves{&score_indices, &class_indices, &kept, &class_pixels,
                                     &short_weights}),
        StatusCode::Incompatible, "'class_weights'", "2 elements");
    SA_EXPECT_FAILURE(
        fill_dice_targets(good, K_CLASSES, K_IGNORE,
                          DiceLeaves{&score_indices, &real_indices, &kept, &class_pixels,
                                     &class_weights}),
        StatusCode::Incompatible, "'class_indices'", "int32");

    EXPECT_EQ(read_ids(score_indices), Classes(6, -7));
    EXPECT_EQ(read_ids(class_indices), Classes(6, -7));
    expect_reals(kept, std::vector<double>(6, K_UNTOUCHED));
    expect_reals(class_pixels, std::vector<double>(3, K_UNTOUCHED));
}
