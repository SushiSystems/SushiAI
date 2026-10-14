/**************************************************************************/
/* test_datasets_collators.cpp                                            */
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
 * @file test_datasets_collators.cpp
 * @brief Tests the input writer and the classification and detection collators.
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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::PixelBox;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the absolute tolerance of a byte divided by 255 in float. */
    constexpr float K_SCALE_TOLERANCE = 1e-6f;

    /** @brief Holds the size of every image of these tests. */
    constexpr ImageSize K_SIZE{1, 2};

    /** @brief Provides an engine, a two-file list and the samples of one batch. */
    class Unit_DatasetsCollators : public SaTest::EngineTest
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

/** @brief Tests that the images are written row by row as bytes divided by 255. */
TEST_F(Unit_DatasetsCollators, InputsAreBytesDividedBy255)
{
    Data::Batch batch = allocate({});
    SA_EXPECT_OK(write_batch_inputs(span(), batch.inputs));

    const float* inputs = batch.inputs.data_as<float>();
    EXPECT_NEAR(inputs[0], 0.2f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[1], 0.2f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[2], 1.0f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[3], 1.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that an image of another size is refused with its position in the batch. */
TEST_F(Unit_DatasetsCollators, AnImageOfAnotherSizeIsRefused)
{
    samples[1].image = Support::uniform_image(ImageSize{2, 2}, 1, 0);

    Data::Batch batch = allocate({});
    SA_EXPECT_FAILURE(write_batch_inputs(span(), batch.inputs), StatusCode::Incompatible,
                      "write_batch_inputs", "image 1", "2 by 2");
}

/** @brief Tests that a tensor of another image count is refused. */
TEST_F(Unit_DatasetsCollators, ATensorOfAnotherImageCountIsRefused)
{
    samples.pop_back();

    Data::Batch batch = allocate({});
    SA_EXPECT_FAILURE(write_batch_inputs(span(), batch.inputs), StatusCode::Incompatible,
                      "write_batch_inputs", "holds 2 images");
}

/** @brief Tests that classification publishes labels as one int32 index over the classes. */
TEST_F(Unit_DatasetsCollators, ClassificationPublishesLabels)
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
TEST_F(Unit_DatasetsCollators, ClassificationWritesTheClassOfEachSample)
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
TEST_F(Unit_DatasetsCollators, ClassificationNamesTheFileOfASampleWithoutAClass)
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
TEST_F(Unit_DatasetsCollators, ABatchWithoutTheRoleIsRefused)
{
    samples[0].class_index = 1;
    samples[1].class_index = 1;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate({});
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::Incompatible,
                      "ClassificationCollator", "no role 'labels'");
}

/** @brief Tests that detection publishes boxes, classes and valid with M slots per image. */
TEST_F(Unit_DatasetsCollators, DetectionPublishesThreeRoles)
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
TEST_F(Unit_DatasetsCollators, DetectionWritesCornersAndPadsWithZero)
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
TEST_F(Unit_DatasetsCollators, DetectionNamesTheFileOfAnImageWithTooManyObjects)
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
TEST_F(Unit_DatasetsCollators, DetectionRefusesAClassOutsideTheSet)
{
    samples[0].objects.add(3, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 2));
    Data::Batch batch = allocate(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "first.png", "object 0", "class 3", "[0, 3)");
}
