/**************************************************************************/
/* test_binary_collator.cpp                                               */
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
 * @file test_binary_collator.cpp
 * @brief Tests the binary collator.
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
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/multi_label_collator.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::PixelBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the size of every image and mask of these tests. */
    constexpr ImageSize K_SIZE{1, 2};

    /** @brief Provides an engine, a two-file list and the samples of one batch. */
    class Medium_BinaryCollator : public SaTest::EngineTest
    {
        protected:
            /** @brief Names the two files and makes one gray sample of K_SIZE per file. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                list.files = {"first.png", "second.png"};

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

            /** @brief Allocates a batch of two samples of a shape over some roles. */
            Data::Batch allocate(const Shape& sample,
                                 const std::vector<Data::TargetDescription>& targets)
            {
                const Data::InputDescription input{sample, DType::Float32};
                Result<Data::Batch> batch = Data::allocate_batch(*engine, 2, input, targets);
                EXPECT_TRUE(batch.status().is_ok());
                return std::move(batch).value();
            }

            /** @brief Allocates a batch of two gray images of K_SIZE over some roles. */
            Data::Batch allocate_images(const std::vector<Data::TargetDescription>& targets)
            {
                return allocate(Shape{K_SIZE.height, K_SIZE.width, 1}, targets);
            }

            /** @brief Holds the files the samples name. */
            SampleList list;

            /** @brief Holds the fitted samples of the batch. */
            std::vector<FittedSample> samples;
    };
} // namespace

/** @brief Tests that binary publishes labels as one float32 per sample that is no index. */
TEST_F(Medium_BinaryCollator, BinaryPublishesLabels)
{
    SA_EXPECT_FAILURE(BinaryCollator::make(2), StatusCode::InvalidDescription, "BinaryCollator",
                      "the positive class is 2");
    SA_EXPECT_FAILURE(BinaryCollator::make(-1), StatusCode::InvalidDescription,
                      "the positive class is -1");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                            BinaryCollator::make(1));

    EXPECT_EQ(collator->task(), K_TASK_BINARY);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);
    EXPECT_EQ(collator->targets()[0].classes, 0);
}

/** @brief Tests that binary writes 1 for the positive class, whichever of the two it is. */
TEST_F(Medium_BinaryCollator, BinaryWritesOneForThePositiveClass)
{
    samples[0].class_index = 1;
    samples[1].class_index = 0;

    for (const std::int32_t positive : {1, 0})
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                                BinaryCollator::make(positive));
        Data::Batch batch = allocate_images(collator->targets());
        SA_EXPECT_OK(collator->collate(list, span(), batch));

        // The labels are the constants 0 and 1, so they compare exactly.
        const float* labels = batch.targets[0].values.data_as<float>();
        EXPECT_EQ(labels[0], positive == 1 ? 1.0f : 0.0f);
        EXPECT_EQ(labels[1], positive == 1 ? 0.0f : 1.0f);
    }
}

/** @brief Tests that binary names the file of a sample with no class or a third class. */
TEST_F(Medium_BinaryCollator, BinaryNamesTheFileOfASampleItCannotLabel)
{
    samples[0].class_index = 1;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                            BinaryCollator::make(1));
    Data::Batch batch = allocate_images(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "BinaryCollator", "no class index");

    samples[1].class_index = 2;
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "class index 2", "[0, 2)");
}
