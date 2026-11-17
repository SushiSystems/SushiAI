/**************************************************************************/
/* test_extra_input_model.cpp                                             */
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
 * @file test_extra_input_model.cpp
 * @brief Tests the extra input and the sample extents of build_extra_input_model().
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/input_leaves.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/sample_models/extra_input_model.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;

/** @brief Tests that the model declares one extra input of the sample shape, in float32. */
TEST(Small_SampleModelsExtraInput, TheModelDeclaresOneExtraInputOfTheSampleShape)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_extra_input_model());

    ASSERT_EQ(model.extra_inputs().size(), 1u);
    EXPECT_EQ(model.extra_inputs()[0].name, SaTest::K_EXTRA_INPUT_OFFSET);
    EXPECT_EQ(model.extra_inputs()[0].sample, SaTest::extra_model_sample());
    EXPECT_EQ(model.extra_inputs()[0].dtype, DType::Float32);
}

/** @brief Tests that the extra input can be looked up by the name the header gives. */
TEST(Small_SampleModelsExtraInput, TheExtraInputIsFoundByItsName)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_extra_input_model());

    SA_EXPECT_OK(model.extra_input(SaTest::K_EXTRA_INPUT_OFFSET));
    SA_EXPECT_FAILURE(model.extra_input("absent"), StatusCode::InvalidDescription, "absent");
}

/** @brief Tests that the element count of the header is the element count of one sample. */
TEST(Small_SampleModelsExtraInput, TheElementCountIsTheElementCountOfTheSample)
{
    EXPECT_EQ(SaTest::extra_model_sample().num_elements(), SaTest::K_EXTRA_MODEL_ELEMENTS);
}
