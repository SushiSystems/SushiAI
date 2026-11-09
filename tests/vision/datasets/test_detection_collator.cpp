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
 * @file test_detection_collator.cpp
 * @brief Tests the detection collator: its refusals, its three roles and the file it names.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/data/roles.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::PixelBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides a host engine, a two-file list and two gray samples. */
    class Small_DetectionCollator : public Support::CollatorBench
    {
    };

    /** @brief Holds the count of classes and of objects a make refuses, and its message. */
    struct MakeRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the count of classes. */
        std::int32_t classes;

        /** @brief Holds the count of object slots per image. */
        std::int32_t objects;

        /** @brief Holds the text the refusal carries. */
        const char* message;
    };

    /** @brief Holds one refused make per row. */
    class Small_DetectionCollatorMakeRefusal : public ::testing::TestWithParam<MakeRefusal>
    {
    };

    /** @brief Adds some objects of one class, each covering the whole image, to a sample. */
    void add_full_boxes(FittedSample& sample, std::int32_t count, std::int32_t class_index)
    {
        for (std::int32_t object = 0; object < count; ++object)
            sample.objects.add(class_index, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    }
} // namespace

/** @brief Tests that make refuses a count of no class and a count of no object. */
TEST_P(Small_DetectionCollatorMakeRefusal, MakeRefusesACountOfNothing)
{
    const MakeRefusal& row = GetParam();

    SA_EXPECT_FAILURE(DetectionCollator::make(row.classes, row.objects),
                      StatusCode::InvalidDescription, "DetectionCollator", row.message);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectionCollatorMakeRefusal,
    ::testing::Values(MakeRefusal{"ZeroClasses", 0, 4, "0 classes"},
                      MakeRefusal{"ZeroObjects", 3, 0, "0 objects"}),
    [](const ::testing::TestParamInfo<MakeRefusal>& info) { return ::SushiAI::Test::parameter_name(info.param.name); });

/** @brief Tests that detection publishes boxes, classes and valid with M slots per image. */
TEST(Small_DetectionCollatorTargets, DetectionPublishesThreeRoles)
{
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
TEST_F(Small_DetectionCollator, DetectionWritesCornersAndPadsWithZero)
{
    samples[1].objects.add(2, PixelBox{0.25f, 0.5f, 1.5f, 1.0f});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 2));
    Data::Batch batch = allocate_images(collator->targets());

    // A stale value in every slot shows that the padding is written and not left as it was.
    Support::fill_targets(batch, 9);
    SA_EXPECT_OK(collator->collate(list, span(), batch));

    // The values are copied and never computed, so they compare exactly.
    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values),
              (std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.25, 0.5, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0}));
    EXPECT_EQ(SaTest::read_tensor(batch.targets[1].values),
              (std::vector<double>{0.0, 0.0, 2.0, 0.0}));
    EXPECT_EQ(SaTest::read_tensor(batch.targets[2].values),
              (std::vector<double>{0.0, 0.0, 1.0, 0.0}));
}

/** @brief Tests that an image with more objects than M is a failure naming its file. */
TEST_F(Small_DetectionCollator, DetectionNamesTheFileOfAnImageWithTooManyObjects)
{
    add_full_boxes(samples[1], 3, 0);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 2));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "DetectionCollator", "3 objects", "at most 2");
}

/** @brief Tests that an object of a class the set lacks is a failure naming its file. */
TEST_F(Small_DetectionCollator, DetectionRefusesAClassOutsideTheSet)
{
    add_full_boxes(samples[0], 1, 3);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectionCollator> collator,
                            DetectionCollator::make(3, 2));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "first.png", "object 0", "class 3", "[0, 3)");
}
