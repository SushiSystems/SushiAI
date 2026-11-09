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
 * @file test_binary_collator.cpp
 * @brief Tests the binary collator: its refusals, its one float label and the file it names.
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
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides a host engine, a two-file list and two gray samples. */
    class Small_BinaryCollator : public Support::CollatorBench
    {
    };

    /** @brief Holds a positive class make refuses. */
    class Small_BinaryCollatorPositive : public ::testing::TestWithParam<std::int32_t>
    {
    };

    /** @brief Holds a positive class, and the labels it makes of the classes 1 and 0. */
    struct LabelRow
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the class that is positive. */
        std::int32_t positive;

        /** @brief Holds the label of the sample of class 1. */
        float first;

        /** @brief Holds the label of the sample of class 0. */
        float second;
    };

    /** @brief Holds one choice of the positive class per row. */
    class Small_BinaryCollatorLabels : public Support::CollatorBench,
                                       public ::testing::WithParamInterface<LabelRow>
    {
    };

    /** @brief Names a test after the positive class it refuses. */
    std::string positive_row_name(const ::testing::TestParamInfo<std::int32_t>& info)
    {
        return info.param < 0 ? "Minus" + std::to_string(-info.param)
                              : "Class" + std::to_string(info.param);
    }
} // namespace

/** @brief Tests that make refuses a positive class that is neither 0 nor 1. */
TEST_P(Small_BinaryCollatorPositive, MakeRefusesAPositiveClassOtherThanZeroAndOne)
{
    const std::string message = "the positive class is " + std::to_string(GetParam());

    SA_EXPECT_FAILURE(BinaryCollator::make(GetParam()), StatusCode::InvalidDescription,
                      "BinaryCollator", message);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BinaryCollatorPositive, ::testing::Values(2, -1),
                         positive_row_name);

/** @brief Tests that binary publishes labels as one float32 per sample that is no index. */
TEST(Small_BinaryCollatorTargets, BinaryPublishesOneFloatLabelPerSample)
{
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
TEST_P(Small_BinaryCollatorLabels, BinaryWritesOneForThePositiveClass)
{
    const LabelRow& row = GetParam();
    samples[0].class_index = 1;
    samples[1].class_index = 0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                            BinaryCollator::make(row.positive));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_OK(collator->collate(list, span(), batch));

    // The labels are the constants 0 and 1, so they compare exactly.
    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values),
              (std::vector<double>{row.first, row.second}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BinaryCollatorLabels,
    ::testing::Values(LabelRow{"ClassOneIsPositive", 1, 1.0f, 0.0f},
                      LabelRow{"ClassZeroIsPositive", 0, 0.0f, 1.0f}),
    [](const ::testing::TestParamInfo<LabelRow>& info) { return ::SushiAI::Test::parameter_name(info.param.name); });

/** @brief Tests that binary names the file of a sample with no class index. */
TEST_F(Small_BinaryCollator, BinaryNamesTheFileOfASampleWithoutAClass)
{
    samples[0].class_index = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                            BinaryCollator::make(1));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "BinaryCollator", "no class index");
}

/** @brief Tests that binary names the file of a sample of a third class. */
TEST_F(Small_BinaryCollator, BinaryNamesTheFileOfASampleOfAThirdClass)
{
    samples[0].class_index = 1;
    samples[1].class_index = 2;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                            BinaryCollator::make(1));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "class index 2", "[0, 2)");
}
