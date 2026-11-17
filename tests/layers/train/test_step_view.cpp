/**************************************************************************/
/* test_step_view.cpp                                                     */
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
 * @file test_step_view.cpp
 * @brief Tests what a StepView carries.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/step_view.hpp>

using namespace SushiAI;

/**
 * @brief Tests that a view refers to its batch and keeps the loss and the outputs in order.
 *        0.75 is a sum of powers of two, so the loss is held and compared with no rounding.
 */
TEST(Small_StepView, AViewCarriesTheBatchTheLossAndTheOutputsInOrder)
{
    const SushiBLAS::Tensor small{};
    const SushiBLAS::Tensor large{};
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"p3", &small},
                                                 Train::StepOutput{"p4", &large}};

    const Train::StepView view{
        batch, 0.75,
        SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};

    EXPECT_EQ(&view.batch, &batch);
    EXPECT_EQ(view.loss, 0.75);
    ASSERT_EQ(view.outputs.size(), 2u);
    EXPECT_EQ(view.outputs[0].name, "p3");
    EXPECT_EQ(view.outputs[0].tensor, &small);
    EXPECT_EQ(view.outputs[1].name, "p4");
    EXPECT_EQ(view.outputs[1].tensor, &large);
}

/** @brief Tests that named_value returns the tensor of the value published under a name. */
TEST(Small_StepView, ANamedValueIsFoundByName)
{
    const SushiBLAS::Tensor classes{};
    const SushiBLAS::Tensor scores{};
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> values{Train::StepOutput{"classes", &classes},
                                                Train::StepOutput{"scores", &scores}};
    const Train::StepView view{batch, 0.0, {},
                               SushiRuntime::span<const Train::StepOutput>(values.data(),
                                                                           values.size())};

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const found,
                            Train::named_value(view, "scores"));

    EXPECT_EQ(found, &scores);
}

/** @brief Tests that an unknown name is refused, listing the names the step holds. */
TEST(Small_StepView, AnUnknownValueNameIsRefusedListingTheNamesHeld)
{
    const SushiBLAS::Tensor classes{};
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> values{Train::StepOutput{"classes", &classes}};
    const Train::StepView view{batch, 0.0, {},
                               SushiRuntime::span<const Train::StepOutput>(values.data(),
                                                                           values.size())};

    SA_EXPECT_FAILURE(Train::named_value(view, "missing"), StatusCode::InvalidDescription,
                      "no objective value 'missing'", "'classes'");
}

/** @brief Tests that a view built without values holds none, as every view of a step without. */
TEST(Small_StepView, AViewBuiltWithoutValuesHoldsNone)
{
    const Data::Batch batch{};
    const Train::StepView view{batch, 0.0, {}};

    EXPECT_TRUE(view.values.empty());
    SA_EXPECT_FAILURE(Train::named_value(view, "classes"), StatusCode::InvalidDescription,
                      "no objective value 'classes'", "(none)");
}

/** @brief Tests that a value whose tensor is null is refused, naming the value. */
TEST(Small_StepView, AValueWithNoTensorIsRefused)
{
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> values{Train::StepOutput{"classes", nullptr}};
    const Train::StepView view{batch, 0.0, {},
                               SushiRuntime::span<const Train::StepOutput>(values.data(),
                                                                           values.size())};

    SA_EXPECT_FAILURE(Train::named_value(view, "classes"), StatusCode::InvalidDescription,
                      "'classes'", "no tensor");
}
