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
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <SushiAI/vision/training/segmentation_metrics.hpp>

using namespace SushiAI;
using Vision::Training::measure_segmentation;
using Vision::Training::PixelExtents;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the extents of the logits of the cases: one 2 by 2 map of three classes. */
    constexpr PixelExtents K_EXTENTS{1, 2, 2, 3};

    /** @brief Holds the mask value of a pixel of no class. */
    constexpr std::int32_t K_IGNORE = 255;

    /** @brief Bounds the absolute error of a ratio of two counts below ten, in double. */
    constexpr double K_TOLERANCE = 1e-15;

    /** @brief Stands for a metric the step did not report; no ratio equals it. */
    constexpr double K_ABSENT = -1.0;

    /** @brief Provides an engine and the batch and the logits of one step. */
    class Unit_VisionTrainingSegmentationMetrics : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a batch whose mask role holds @p classes as INT32 [1, 2, 2]. */
            Data::Batch make_batch(const std::vector<std::int32_t>& classes)
            {
                Result<SushiBLAS::Tensor> made =
                    allocate_tensor(*engine, Shape{1, 2, 2}, DType::Int32);
                EXPECT_TRUE(made.status().is_ok());
                for (std::size_t i = 0; i < classes.size(); ++i)
                    made.value().data_as<std::int32_t>()[i] = classes[i];

                Data::Batch batch;
                batch.targets.push_back(
                    Data::BatchTarget{std::string(Data::K_ROLE_MASK), std::move(made).value()});

                return batch;
            }

            /** @brief Returns float32 [1, 2, 2, 3] logits holding @p values. */
            SushiBLAS::Tensor make_logits(const std::vector<double>& values)
            {
                Result<SushiBLAS::Tensor> made =
                    allocate_tensor(*engine, Shape{1, 2, 2, 3}, DType::Float32);
                EXPECT_TRUE(made.status().is_ok());
                SaTest::write_tensor(made.value(), values);

                return std::move(made).value();
            }
    };
} // namespace

/** @brief Tests both metrics against counts worked by hand, with one pixel ignored. */
TEST_F(Unit_VisionTrainingSegmentationMetrics, TheMetricsEqualTheHandCount)
{
    // Picks 0, 1, 0 and 1 against 0, 2, ignored and 1: two of three right.
    const SushiBLAS::Tensor logits = make_logits(
        {2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 5.0, 0.0, 0.0, -1.0, 0.5, 0.0});
    const Data::Batch batch = make_batch({0, 2, K_IGNORE, 1});
    const Train::StepOutput output{"output", &logits};
    const Train::StepView step{batch, 0.0,
                               SushiRuntime::span<const Train::StepOutput>(&output, 1)};
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

/** @brief Tests that a step with two outputs, no mask or a label out of bounds is refused. */
TEST_F(Unit_VisionTrainingSegmentationMetrics, AStepThatDoesNotFitIsRefused)
{
    const SushiBLAS::Tensor logits = make_logits(std::vector<double>(12, 0.0));
    const Data::Batch batch = make_batch({0, 2, K_IGNORE, 1});
    const Data::Batch bad_label = make_batch({0, 3, K_IGNORE, 1});
    const Data::Batch no_mask;
    const std::vector<Train::StepOutput> two{{"a", &logits}, {"b", &logits}};
    const Train::StepOutput output{"output", &logits};
    const SushiRuntime::span<const Train::StepOutput> one(&output, 1);
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(
        measure_segmentation(
            Train::StepView{batch, 0.0,
                            SushiRuntime::span<const Train::StepOutput>(two.data(), two.size())},
            K_EXTENTS, K_IGNORE, metrics),
        StatusCode::InvalidDescription, "published 2");
    SA_EXPECT_FAILURE(
        measure_segmentation(Train::StepView{no_mask, 0.0, one}, K_EXTENTS, K_IGNORE, metrics),
        StatusCode::InvalidDescription, "'mask'");
    SA_EXPECT_FAILURE(
        measure_segmentation(Train::StepView{bad_label, 0.0, one}, K_EXTENTS, K_IGNORE, metrics),
        StatusCode::Incompatible, "true class of pixel 1", "is 3");
    SA_EXPECT_FAILURE(
        measure_segmentation(Train::StepView{batch, 0.0, one}, PixelExtents{1, 2, 2, 4},
                             K_IGNORE, metrics),
        StatusCode::Incompatible, "the logits");

    EXPECT_TRUE(metrics.all().empty());
}
