/**************************************************************************/
/* test_segmentation_metrics.cpp                                          */
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
 * @file test_segmentation_metrics.cpp
 * @brief Tests the mean IoU and the pixel accuracy a per-pixel objective reports of a step.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_source.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <SushiAI/vision/training/segmentation_metrics.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Vision::Training::measure_segmentation;
using Vision::Training::PixelExtents;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the extents of the cases: one 2 by 2 map of three classes. */
    constexpr PixelExtents K_EXTENTS{1, 2, 2, 3};

    /** @brief Holds the mask value of a pixel of no class. */
    constexpr std::int32_t K_IGNORE = 255;

    /**
     * @brief Holds the Bounds the absolute error of a ratio of two counts below ten, in double: 16
     *        times the double unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-15.
     */
    constexpr double K_TOLERANCE = 16.0 * TestSupport::unit_roundoff<double>();

    /** @brief Stands for a metric the step did not report; no ratio equals it. */
    constexpr double K_ABSENT = -1.0;

    /** @brief Returns the one value of a step, named as the objectives name it. */
    Train::StepOutput class_value(const SushiBLAS::Tensor& classes)
    {
        return Train::StepOutput{std::string(Vision::Training::K_PIXEL_CLASSES_VALUE), &classes};
    }

    /** @brief Provides an engine and the batch and the class values of one step. */
    class Medium_VisionTrainingSegmentationMetrics : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a batch whose mask role holds @p classes as INT32 [1, 2, 2]. */
            Data::Batch make_batch(const std::vector<std::int32_t>& classes)
            {
                Result<SushiBLAS::Tensor> made =
                    allocate_tensor(*engine, Shape{1, 2, 2}, DType::Int32, TensorAccess::Host);
                EXPECT_TRUE(made.status().is_ok());
                for (std::size_t i = 0; i < classes.size(); ++i)
                    made.value().data_as<std::int32_t>()[i] = classes[i];

                Data::Batch batch;
                batch.targets.push_back(
                    Data::BatchTarget{std::string(Data::K_ROLE_MASK), std::move(made).value()});

                return batch;
            }

            /** @brief Returns the INT64 [4] class values a device writes, holding @p classes. */
            SushiBLAS::Tensor make_classes(const std::vector<std::int64_t>& classes)
            {
                Result<SushiBLAS::Tensor> made =
                    allocate_tensor(*engine, Shape{static_cast<std::int64_t>(classes.size())},
                                    DType::Int64);
                EXPECT_TRUE(made.status().is_ok());
                SA_EXPECT_OK(write_elements(made.value(), classes));

                return std::move(made).value();
            }

    };

    /**
     * @brief Describes the tensors of a step that is refused before any element is read.
     *
     * A tensor it returns holds its extents and element type and no storage, so a case that uses
     * one is decided on facts alone and needs no device.
     */
    class Small_VisionTrainingSegmentationMetrics : public ::testing::Test
    {
        protected:
            /** @brief Returns a tensor of @p shape and @p dtype that holds no element. */
            SushiBLAS::Tensor described(const Shape& shape, DType dtype)
            {
                return source.create(shape, dtype).value();
            }

            /** @brief Returns a batch whose mask role is described as INT32 [1, 2, 2]. */
            Data::Batch described_batch()
            {
                Data::Batch batch;
                batch.targets.push_back(Data::BatchTarget{std::string(Data::K_ROLE_MASK),
                                                          described(Shape{1, 2, 2}, DType::Int32)});

                return batch;
            }

            Graph::InMemoryTensorSource source; /**< @brief Makes the described tensors. */
    };
} // namespace

/** @brief Tests both metrics against counts worked by hand, with one pixel ignored. */
TEST_F(Medium_VisionTrainingSegmentationMetrics, TheMetricsEqualTheHandCount)
{
    // Picks 0, 1, 0 and 1 against 0, 2, ignored and 1: two of three right.
    const SushiBLAS::Tensor classes = make_classes({0, 1, 0, 1});
    const Data::Batch batch = make_batch({0, 2, K_IGNORE, 1});
    const Train::StepOutput value = class_value(classes);
    const Train::StepView step{batch, 0.0, {},
                               SushiRuntime::span<const Train::StepOutput>(&value, 1)};
    Train::Metrics metrics;

    SA_EXPECT_OK(measure_segmentation(step, K_EXTENTS, K_IGNORE, metrics));

    // Class 0 has IoU 1, class 1 has 1 / 2 and class 2 has 0.
    EXPECT_EQ(metrics.all().size(), 2u);
    EXPECT_NEAR(metrics.value_or("mean_iou", K_ABSENT), 0.5, K_TOLERANCE);
    EXPECT_NEAR(metrics.value_or("pixel_accuracy", K_ABSENT), 2.0 / 3.0, K_TOLERANCE);
    const Train::Metric* const mean_iou = metrics.find("mean_iou");
    ASSERT_NE(mean_iou, nullptr);
    EXPECT_TRUE(mean_iou->is_fraction);
}

/**
 * @brief Tests the metrics over 4100 classes.
 *
 * Four pixels, so the matrix holds almost every count at zero.
 */
TEST_F(Medium_VisionTrainingSegmentationMetrics, MoreThan4096ClassesAreCounted)
{
    constexpr PixelExtents extents{1, 2, 2, 4100};
    // Classes 0, 4099, 0 and 2 against 0, 4099, 4099 and 2: three of four right.
    const SushiBLAS::Tensor classes = make_classes({0, 4099, 0, 2});
    const Data::Batch batch = make_batch({0, 4099, 4099, 2});
    const Train::StepOutput value = class_value(classes);
    const Train::StepView step{batch, 0.0, {},
                               SushiRuntime::span<const Train::StepOutput>(&value, 1)};
    Train::Metrics metrics;

    SA_EXPECT_OK(measure_segmentation(step, extents, K_IGNORE, metrics));

    // Class 0 has IoU 1 / 2, class 4099 has 1 / 2 and class 2 has 1.
    EXPECT_NEAR(metrics.value_or("mean_iou", K_ABSENT), 2.0 / 3.0, K_TOLERANCE);
    EXPECT_NEAR(metrics.value_or("pixel_accuracy", K_ABSENT), 3.0 / 4.0, K_TOLERANCE);
}

/** @brief Tests that a step that holds no class value is refused, naming the value. */
TEST_F(Small_VisionTrainingSegmentationMetrics, AStepWithNoClassValueIsRefused)
{
    const Data::Batch batch = described_batch();
    const Train::StepView step{batch, 0.0, {}, {}};
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(measure_segmentation(step, K_EXTENTS, K_IGNORE, metrics),
                      StatusCode::InvalidDescription, "'pixel_classes'",
                      "holds no objective value");
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that a batch with no mask role is refused, naming the role. */
TEST_F(Small_VisionTrainingSegmentationMetrics, ABatchWithNoMaskIsRefused)
{
    const SushiBLAS::Tensor classes = described(Shape{4}, DType::Int64);
    const Data::Batch no_mask;
    const Train::StepOutput value = class_value(classes);
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(
        measure_segmentation(
            Train::StepView{no_mask, 0.0, {},
                            SushiRuntime::span<const Train::StepOutput>(&value, 1)},
            K_EXTENTS, K_IGNORE, metrics),
        StatusCode::InvalidDescription, "'mask'");
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that a class value of another count than the extents state is refused. */
TEST_F(Small_VisionTrainingSegmentationMetrics, AClassValueOfAnotherCountIsRefused)
{
    const SushiBLAS::Tensor classes = described(Shape{3}, DType::Int64);
    const Data::Batch batch = described_batch();
    const Train::StepOutput value = class_value(classes);
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(
        measure_segmentation(
            Train::StepView{batch, 0.0, {},
                            SushiRuntime::span<const Train::StepOutput>(&value, 1)},
            K_EXTENTS, K_IGNORE, metrics),
        StatusCode::Incompatible, "class value holds 3 elements");
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that a class value that is not INT64 is refused. */
TEST_F(Small_VisionTrainingSegmentationMetrics, AClassValueOfAnotherTypeIsRefused)
{
    const SushiBLAS::Tensor classes = described(Shape{4}, DType::Int32);
    const Data::Batch batch = described_batch();
    const Train::StepOutput value = class_value(classes);
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(
        measure_segmentation(
            Train::StepView{batch, 0.0, {},
                            SushiRuntime::span<const Train::StepOutput>(&value, 1)},
            K_EXTENTS, K_IGNORE, metrics),
        StatusCode::Incompatible, "class value holds 4 elements of int32");
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that a label at the class count is refused, naming its pixel. */
TEST_F(Medium_VisionTrainingSegmentationMetrics, ALabelOutOfBoundsIsRefused)
{
    const SushiBLAS::Tensor classes = make_classes({0, 0, 0, 0});
    const Data::Batch bad_label = make_batch({0, 3, K_IGNORE, 1});
    const Train::StepOutput value = class_value(classes);
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(
        measure_segmentation(
            Train::StepView{bad_label, 0.0, {},
                            SushiRuntime::span<const Train::StepOutput>(&value, 1)},
            K_EXTENTS, K_IGNORE, metrics),
        StatusCode::Incompatible, "true class of pixel 1", "is 3");
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that a class the device wrote beyond the class count is refused. */
TEST_F(Medium_VisionTrainingSegmentationMetrics, AClassValueOutOfBoundsIsRefused)
{
    const SushiBLAS::Tensor classes = make_classes({0, 3, 0, 0});
    const Data::Batch batch = make_batch({0, 2, K_IGNORE, 1});
    const Train::StepOutput value = class_value(classes);
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(
        measure_segmentation(
            Train::StepView{batch, 0.0, {},
                            SushiRuntime::span<const Train::StepOutput>(&value, 1)},
            K_EXTENTS, K_IGNORE, metrics),
        StatusCode::Incompatible, "predicted class of pixel 1", "is 3");
    EXPECT_TRUE(metrics.all().empty());
}
