/**************************************************************************/
/* test_precision.cpp                                                     */
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
 * @file test_precision.cpp
 * @brief Tests the precision policy interface and its two shipped policies.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>

using namespace SushiAI;
using Graph::MixedPrecisionPolicy;
using Graph::PrecisionPolicy;
using Graph::UniformPrecisionPolicy;

/** @brief Tests that the uniform policy is named "fp32" and the mixed one "mixed". */
TEST(Small_PrecisionPolicy, ThePoliciesNameThemselves)
{
    const UniformPrecisionPolicy uniform;
    const MixedPrecisionPolicy mixed;

    EXPECT_EQ(uniform.name(), "fp32");
    EXPECT_EQ(mixed.name(), "mixed");
}

/** @brief Tests that make stores the real float narrow type it is given. */
TEST(Small_PrecisionPolicy, MakeStoresARealFloatLowType)
{
    SA_ASSERT_OK_AND_ASSIGN(const MixedPrecisionPolicy mixed_float,
                            MixedPrecisionPolicy::make(DType::Float32));

    EXPECT_EQ(mixed_float.low(), DType::Float32);
}

/** @brief Tests that a mixed policy built without an argument narrows to HALF. */
TEST(Small_PrecisionPolicy, TheMixedPolicyDefaultsToHalf)
{
    const MixedPrecisionPolicy mixed;

    EXPECT_EQ(mixed.low(), DType::Half);
}

/** @brief Tests that make refuses a narrow type that is not a real float. */
TEST(Small_PrecisionPolicy, MakeRefusesALowTypeThatIsNotARealFloat)
{
    SA_EXPECT_FAILURE(MixedPrecisionPolicy::make(DType::Int32), StatusCode::InvalidDescription,
                      "narrows to a real floating type");
    SA_EXPECT_FAILURE(MixedPrecisionPolicy::make(DType::Complex32),
                      StatusCode::InvalidDescription, "narrows to a real floating type");
}

/** @brief Tests that make accepts HALF as the narrow type. */
TEST(Small_PrecisionPolicy, MakeAcceptsHalfAsTheLowType)
{
    SA_EXPECT_OK(MixedPrecisionPolicy::make(DType::Half));
}

/** @brief Tests that the dtype finder asks the policy it is given, so two policies disagree. */
TEST(Small_PrecisionPolicy, APolicyIsConsultedThroughTheInterface)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 3}, "a");
    const Graph::ValueID b = builder.input(Shape{3, 2}, "b");
    SA_EXPECT_OK(builder.matmul(a, b));

    const UniformPrecisionPolicy uniform;
    const MixedPrecisionPolicy mixed;
    const PrecisionPolicy& uniform_policy = uniform;
    const PrecisionPolicy& mixed_policy = mixed;

    EXPECT_TRUE(Graph::find_dtype_mismatches(graph, uniform_policy).empty());
    EXPECT_FALSE(Graph::find_dtype_mismatches(graph, mixed_policy).empty());
}
