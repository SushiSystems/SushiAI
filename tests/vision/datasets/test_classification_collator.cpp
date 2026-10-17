/**************************************************************************/
/* test_classification_collator.cpp                                       */
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
 * @file test_classification_collator.cpp
 * @brief Tests the classification collator.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::PixelBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the size of every image of these tests. */
    constexpr ImageSize K_SIZE{1, 2};

    /** @brief Provides an engine, a two-file list and the samples of one batch. */
    class Medium_ClassificationCollator : public SaTest::EngineTest
    {
        protected:
            /** @brief Names the two files and makes one gray sample of K_SIZE per file. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                list.files = {"first.png", "crowded.png"};

                samples.resize(2);
                samples[0].image = Support::uniform_image(K_SIZE, 1, 51);
                samples[0].file = 0;
                samples[1].image = Support::uniform_image(K_SIZE, 1, 255);
                samples[1].file = 1;
            }

            /** @brief Returns the samples as the span a collator takes. */
            SushiRuntime::span<const FittedSample> span() const
            {
                return SushiRuntime::span<const FittedSample>(samples.data(), samples.size());
            }

            /** @brief Allocates a batch of two gray images of K_SIZE over some roles. */
            Data::Batch allocate(const std::vector<Data::TargetDescription>& targets)
            {
                const Data::InputDescription input{Shape{K_SIZE.height, K_SIZE.width, 1},
                                                   DType::Float32};
                Result<Data::Batch> batch = Data::allocate_batch(*engine, 2, input, targets);
                EXPECT_TRUE(batch.status().is_ok());
                return std::move(batch).value();
            }

            /** @brief Holds the files the samples name. */
            SampleList list;

            /** @brief Holds the fitted samples of the batch. */
            std::vector<FittedSample> samples;
    };

    /** @brief Holds the absolute tolerance of a byte divided by 255 in float. */
    constexpr float K_SCALE_TOLERANCE = 1e-6f;

} // namespace

/** @brief Tests that classification publishes labels as one int32 index over the classes. */
TEST_F(Medium_ClassificationCollator, ClassificationPublishesLabels)
{
    SA_EXPECT_FAILURE(ClassificationCollator::make(0), StatusCode::InvalidDescription,
                      "ClassificationCollator", "0 classes");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));

    EXPECT_EQ(collator->task(), K_TASK_CLASSIFICATION);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Int32);
    EXPECT_EQ(collator->targets()[0].classes, 7);
}

/** @brief Tests that classification writes the inputs and each sample's class index. */
TEST_F(Medium_ClassificationCollator, ClassificationWritesTheClassOfEachSample)
{
    samples[0].class_index = 6;
    samples[1].class_index = 0;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate(collator->targets());
    SA_EXPECT_OK(collator->collate(list, span(), batch));

    const std::int32_t* labels = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(labels[0], 6);
    EXPECT_EQ(labels[1], 0);
    EXPECT_NEAR(batch.inputs.data_as<float>()[2], 1.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that classification names the file of a sample it cannot label. */
TEST_F(Medium_ClassificationCollator, ClassificationNamesTheFileOfASampleWithoutAClass)
{
    samples[0].class_index = 1;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "crowded.png", "ClassificationCollator", "no class index");

    samples[1].class_index = 7;
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "crowded.png", "class index 7", "[0, 7)");
}

/** @brief Tests that a batch allocated over other roles is refused with the missing role. */
TEST_F(Medium_ClassificationCollator, ABatchWithoutTheRoleIsRefused)
{
    samples[0].class_index = 1;
    samples[1].class_index = 1;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate({});
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::Incompatible,
                      "ClassificationCollator", "no role 'labels'");
}
