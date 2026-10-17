/**************************************************************************/
/* test_multi_label_collator.cpp                                          */
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
 * @file test_multi_label_collator.cpp
 * @brief Tests the multi-label collator.
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
    class Medium_MultiLabelCollator : public SaTest::EngineTest
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

/** @brief Tests that multi-label writes one flag per class a sample's objects name. */
TEST_F(Medium_MultiLabelCollator, MultiLabelWritesTheSetOfClassesPresent)
{
    SA_EXPECT_FAILURE(MultiLabelCollator::make(0), StatusCode::InvalidDescription,
                      "MultiLabelCollator", "0 classes");

    samples[0].objects.add(2, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    samples[0].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    samples[0].objects.add(2, PixelBox{1.0f, 0.0f, 2.0f, 1.0f});
    samples[0].class_index = 3;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MultiLabelCollator> collator,
                            MultiLabelCollator::make(4));

    EXPECT_EQ(collator->task(), K_TASK_MULTI_LABEL);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{4});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);

    Data::Batch batch = allocate_images(collator->targets());
    std::fill_n(batch.targets[0].values.data_as<float>(), 8, 9.0f);
    SA_EXPECT_OK(collator->collate(list, span(), batch));

    // The second sample has no object and no class, so its row is zeros and not stale.
    const float* labels = batch.targets[0].values.data_as<float>();
    const std::vector<float> expected = {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(std::vector<float>(labels, labels + 8), expected);
}

/** @brief Tests that multi-label names the file of a sample with a class outside the set. */
TEST_F(Medium_MultiLabelCollator, MultiLabelRefusesAClassOutsideTheSet)
{
    samples[1].objects.add(4, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MultiLabelCollator> collator,
                            MultiLabelCollator::make(4));
    Data::Batch batch = allocate_images(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "MultiLabelCollator", "the class 4", "[0, 4)");
}
