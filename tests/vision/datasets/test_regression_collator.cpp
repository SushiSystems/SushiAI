/**************************************************************************/
/* test_regression_collator.cpp                                           */
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
 * @file test_regression_collator.cpp
 * @brief Tests the regression collator: its refusals, its one values role and the rows it copies.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/data/roles.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides a host engine, a two-file list and two gray samples. */
    class Small_RegressionCollator : public Support::CollatorBench
    {
    };

    /** @brief Holds the counts a make refuses and the text of the refusal. */
    struct MakeRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the count of features. */
        std::int32_t features;

        /** @brief Holds the count of targets. */
        std::int32_t targets;

        /** @brief Holds the text the refusal carries. */
        const char* message;
    };

    /** @brief Holds one refused make per row. */
    class Small_RegressionCollatorMakeRefusal : public ::testing::TestWithParam<MakeRefusal>
    {
    };

    /**
     * @brief Gives each sample three features and two targets out of rows of five values, and
     *        drops its image.
     */
    void hold_rows(std::vector<FittedSample>& samples, const std::vector<float>& rows)
    {
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            samples[i].image = Vision::Imaging::Image();
            samples[i].features = SushiRuntime::span<const float>(rows.data() + i * 5, 3);
            samples[i].targets = SushiRuntime::span<const float>(rows.data() + i * 5 + 3, 2);
        }
    }
} // namespace

/** @brief Tests that make refuses a count of no feature and a count of no target. */
TEST_P(Small_RegressionCollatorMakeRefusal, MakeRefusesAFeatureOrTargetCountOfZero)
{
    const MakeRefusal& row = GetParam();

    SA_EXPECT_FAILURE(RegressionCollator::make(row.features, row.targets),
                      StatusCode::InvalidDescription, "RegressionCollator", row.message);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_RegressionCollatorMakeRefusal,
    ::testing::Values(MakeRefusal{"ZeroFeatures", 0, 1, "0 features"},
                      MakeRefusal{"ZeroTargets", 3, 0, "0 targets"}),
    [](const ::testing::TestParamInfo<MakeRefusal>& info) { return ::SushiAI::Test::parameter_name(info.param.name); });

/** @brief Tests that regression publishes one values role of the target width. */
TEST(Small_RegressionCollatorTargets, RegressionPublishesOneValuesRole)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RegressionCollator> collator,
                            RegressionCollator::make(3, 2));

    EXPECT_EQ(collator->task(), K_TASK_REGRESSION);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_VALUES);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{2});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);
}

/** @brief Tests that regression copies the features into the inputs and the targets beside. */
TEST_F(Small_RegressionCollator, RegressionCopiesFeaturesAndTargets)
{
    const std::vector<float> rows = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f,
                                     4.0f, 5.0f, 6.0f, 40.0f, 50.0f};
    hold_rows(samples, rows);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RegressionCollator> collator,
                            RegressionCollator::make(3, 2));
    Data::Batch batch = allocate(Shape{3}, collator->targets());

    SA_EXPECT_OK(collator->collate(list, span(), batch));

    // The values are copied and never computed, so they compare exactly.
    EXPECT_EQ(SaTest::read_tensor(batch.inputs),
              (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values),
              (std::vector<double>{10.0, 20.0, 40.0, 50.0}));
}

/** @brief Tests that regression refuses samples that hold images and no row. */
TEST_F(Small_RegressionCollator, RegressionRefusesASampleThatHoldsNoRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RegressionCollator> collator,
                            RegressionCollator::make(3, 2));
    Data::Batch batch = allocate(Shape{3}, collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "first.png", "RegressionCollator", "0 features and 0 targets");
}

/** @brief Tests that regression refuses inputs narrower than the rows it copies. */
TEST_F(Small_RegressionCollator, RegressionRefusesInputsOfAnotherWidth)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RegressionCollator> collator,
                            RegressionCollator::make(3, 2));
    Data::Batch narrow = allocate(Shape{2}, collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), narrow), StatusCode::Incompatible,
                      "RegressionCollator", "float32 of [2, 3]");
}
