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
 * @file test_multi_label_collator.cpp
 * @brief Tests the multi-label collator: its refusals, its float row and the file it names.
 * @author Mustafa Garip
 */

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
#include <SushiAI/vision/datasets/multi_label_collator.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::PixelBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides a host engine, a two-file list and two gray samples. */
    class Small_MultiLabelCollator : public Support::CollatorBench
    {
    };
} // namespace

/** @brief Tests that make refuses a count of no class. */
TEST(Small_MultiLabelCollatorTargets, MakeRefusesZeroClasses)
{
    SA_EXPECT_FAILURE(MultiLabelCollator::make(0), StatusCode::InvalidDescription,
                      "MultiLabelCollator", "0 classes");
}

/** @brief Tests that multi-label publishes a float row with one flag per class. */
TEST(Small_MultiLabelCollatorTargets, MultiLabelPublishesOneFloatFlagPerClass)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MultiLabelCollator> collator,
                            MultiLabelCollator::make(4));

    EXPECT_EQ(collator->task(), K_TASK_MULTI_LABEL);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{4});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);
}

/** @brief Tests that multi-label writes one flag per class a sample's objects name. */
TEST_F(Small_MultiLabelCollator, MultiLabelWritesTheSetOfClassesPresent)
{
    samples[0].objects.add(2, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    samples[0].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    samples[0].objects.add(2, PixelBox{1.0f, 0.0f, 2.0f, 1.0f});
    samples[0].class_index = 3;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MultiLabelCollator> collator,
                            MultiLabelCollator::make(4));
    Data::Batch batch = allocate_images(collator->targets());

    // A stale value in every slot shows that the empty sample's row is written as zeros.
    Support::fill_targets(batch, 9);
    SA_EXPECT_OK(collator->collate(list, span(), batch));

    // The second sample has no object and no class, so its row is zeros and not stale.
    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values),
              (std::vector<double>{1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0}));
}

/** @brief Tests that multi-label names the file of a sample with a class outside the set. */
TEST_F(Small_MultiLabelCollator, MultiLabelRefusesAClassOutsideTheSet)
{
    samples[1].objects.add(4, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MultiLabelCollator> collator,
                            MultiLabelCollator::make(4));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "MultiLabelCollator", "the class 4", "[0, 4)");
}
