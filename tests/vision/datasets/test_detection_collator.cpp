/**************************************************************************/
/* test_detection_collator.cpp                                            */
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
 * @file test_detection_collator.cpp
 * @brief Tests the detection collator.
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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
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
    class Unit_DetectionCollator : public SaTest::EngineTest
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
} // namespace

/** @brief Tests that detection publishes boxes, classes and valid with M slots per image. */
TEST_F(Unit_DetectionCollator, DetectionPublishesThreeRoles)
{
    SA_EXPECT_FAILURE(DetectionCollator::make(0, 4), StatusCode::InvalidDescription,
                      "DetectionCollator", "0 classes");
    SA_EXPECT_FAILURE(DetectionCollator::make(3, 0), StatusCode::InvalidDescription,
                      "DetectionCollator", "0 objects");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 4));
    const std::vector<Data::TargetDescription>& targets = collator->targets();

    EXPECT_EQ(collator->task(), K_TASK_DETECTION);
    ASSERT_EQ(targets.size(), 3u);
    EXPECT_EQ(targets[0].role, Data::K_ROLE_BOXES);
    EXPECT_EQ(targets[0].sample_shape, (Shape{4, 4}));
    EXPECT_EQ(targets[0].dtype, DType::Float32);
    EXPECT_EQ(targets[1].role, Data::K_ROLE_CLASSES);
    EXPECT_EQ(targets[1].sample_shape, Shape{4});
    EXPECT_EQ(targets[1].dtype, DType::Int32);
    EXPECT_EQ(targets[1].classes, 3);
    EXPECT_EQ(targets[2].role, Data::K_ROLE_VALID);
    EXPECT_EQ(targets[2].sample_shape, Shape{4});
    EXPECT_EQ(targets[2].dtype, DType::Float32);
}

/** @brief Tests that detection writes pixel corners in object order and zeroes the padding. */
TEST_F(Unit_DetectionCollator, DetectionWritesCornersAndPadsWithZero)
{
    samples[1].objects.add(2, PixelBox{0.25f, 0.5f, 1.5f, 1.0f});

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 2));
    Data::Batch batch = allocate(collator->targets());

    // A stale value in every slot shows that the padding is written and not left as it was.
    for (Data::BatchTarget& target : batch.targets)
    {
        if (target.values.dtype == DType::Float32)
            std::fill_n(target.values.data_as<float>(), target.values.num_elements, 9.0f);
        else
            std::fill_n(target.values.data_as<std::int32_t>(), target.values.num_elements, 9);
    }

    SA_EXPECT_OK(collator->collate(list, span(), batch));

    const float* boxes = batch.targets[0].values.data_as<float>();
    const std::int32_t* classes = batch.targets[1].values.data_as<std::int32_t>();
    const float* valid = batch.targets[2].values.data_as<float>();
    const std::vector<float> expected_boxes = {0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.25f, 0.5f, 1.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<std::int32_t> expected_classes = {0, 0, 2, 0};
    const std::vector<float> expected_valid = {0.0f, 0.0f, 1.0f, 0.0f};

    // The values are copied and never computed, so they compare exactly.
    EXPECT_EQ(std::vector<float>(boxes, boxes + 16), expected_boxes);
    EXPECT_EQ(std::vector<std::int32_t>(classes, classes + 4), expected_classes);
    EXPECT_EQ(std::vector<float>(valid, valid + 4), expected_valid);
}

/** @brief Tests that an image with more objects than M is a failure naming its file. */
TEST_F(Unit_DetectionCollator, DetectionNamesTheFileOfAnImageWithTooManyObjects)
{
    for (std::int32_t object = 0; object < 3; ++object)
        samples[1].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 2));
    Data::Batch batch = allocate(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "crowded.png", "DetectionCollator", "3 objects", "at most 2");
}

/** @brief Tests that an object of a class the set lacks is a failure naming its file. */
TEST_F(Unit_DetectionCollator, DetectionRefusesAClassOutsideTheSet)
{
    samples[0].objects.add(3, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 2));
    Data::Batch batch = allocate(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "first.png", "object 0", "class 3", "[0, 3)");
}
