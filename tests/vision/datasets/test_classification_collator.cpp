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
 * @file test_classification_collator.cpp
 * @brief Tests the classification collator: its refusals, its int32 labels and the file it names.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/data/roles.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides a host engine, a two-file list and two gray samples. */
    class Small_ClassificationCollator : public Support::CollatorBench
    {
    };

    /**
     * @brief Holds the absolute tolerance of a byte divided by 255 in float: 32 times the float
     *        unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_SCALE_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());
} // namespace

/** @brief Tests that make refuses a count of no class. */
TEST(Small_ClassificationCollatorTargets, MakeRefusesZeroClasses)
{
    SA_EXPECT_FAILURE(ClassificationCollator::make(0), StatusCode::InvalidDescription,
                      "ClassificationCollator", "0 classes");
}

/** @brief Tests that classification publishes labels as one int32 index over the classes. */
TEST(Small_ClassificationCollatorTargets, ClassificationPublishesOneInt32LabelPerSample)
{
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
TEST_F(Small_ClassificationCollator, ClassificationWritesTheClassOfEachSample)
{
    samples[0].class_index = 6;
    samples[1].class_index = 0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_OK(collator->collate(list, span(), batch));

    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values), (std::vector<double>{6.0, 0.0}));
    EXPECT_NEAR(SaTest::read_tensor(batch.inputs)[2], 1.0, K_SCALE_TOLERANCE);
}

/** @brief Tests that classification names the file of a sample with no class index. */
TEST_F(Small_ClassificationCollator, ClassificationNamesTheFileOfASampleWithoutAClass)
{
    samples[0].class_index = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "ClassificationCollator", "no class index");
}

/** @brief Tests that classification names the file of a sample whose class is out of range. */
TEST_F(Small_ClassificationCollator, ClassificationNamesTheFileOfASampleOfAClassOutsideTheSet)
{
    samples[0].class_index = 1;
    samples[1].class_index = 7;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "class index 7", "[0, 7)");
}

/** @brief Tests that a batch allocated over other roles is refused with the missing role. */
TEST_F(Small_ClassificationCollator, ABatchWithoutTheRoleIsRefused)
{
    samples[0].class_index = 1;
    samples[1].class_index = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ClassificationCollator> collator,
                            ClassificationCollator::make(7));
    Data::Batch batch = allocate_images({});

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::Incompatible,
                      "ClassificationCollator", "no role 'labels'");
}
