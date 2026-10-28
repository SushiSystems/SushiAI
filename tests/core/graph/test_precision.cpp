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
 * @file test_precision.cpp
 * @brief Tests the precision policy interface and the names and defaults of its two policies.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>

using namespace SushiAI;
using Graph::MixedPrecisionPolicy;
using Graph::PrecisionChoice;
using Graph::PrecisionPolicy;
using Graph::PrecisionQuery;
using Graph::UniformPrecisionPolicy;

namespace
{
    /** @brief Computes every operation in FLOAT64, whatever the graph declares. */
    class EverythingWide final : public PrecisionPolicy
    {
        public:
            /** @brief Returns FLOAT64 with a fixed reason. */
            PrecisionChoice choose(const PrecisionQuery&) const override
            {
                return PrecisionChoice{DType::Float64, "the test widens everything"};
            }

            /** @brief Returns the name the test gave the policy. */
            std::string_view name() const noexcept override
            {
                return "wide";
            }
    };
} // namespace

/** @brief Tests that the two policies name themselves and that the mixed one narrows to half. */
TEST(Small_PrecisionPolicy, ThePoliciesNameThemselvesAndTheMixedOneDefaultsToHalf)
{
    const UniformPrecisionPolicy uniform;
    const MixedPrecisionPolicy mixed;
    SA_ASSERT_OK_AND_ASSIGN(const MixedPrecisionPolicy mixed_float,
                            MixedPrecisionPolicy::make(DType::Float32));

    EXPECT_EQ(uniform.name(), "fp32");
    EXPECT_EQ(mixed.name(), "mixed");
    EXPECT_EQ(mixed.low(), DType::Half);
    EXPECT_EQ(mixed_float.low(), DType::Float32);
}

/** @brief Tests that make refuses a narrow type that is not a real float and accepts HALF. */
TEST(Small_PrecisionPolicy, MakeRefusesALowTypeThatIsNotARealFloat)
{
    SA_EXPECT_FAILURE(MixedPrecisionPolicy::make(DType::Int32), StatusCode::InvalidDescription,
                      "narrows to a real floating type");
    SA_EXPECT_FAILURE(MixedPrecisionPolicy::make(DType::Complex32),
                      StatusCode::InvalidDescription, "narrows to a real floating type");
    SA_EXPECT_OK(MixedPrecisionPolicy::make(DType::Half));
}

/** @brief Tests that a default query names a FLOAT32 result and a default choice FLOAT32. */
TEST(Small_PrecisionPolicy, TheDefaultQueryAndChoiceAreFloat32)
{
    const PrecisionQuery query;
    const PrecisionChoice choice;

    EXPECT_EQ(query.declared, DType::Float32);
    EXPECT_EQ(choice.dtype, DType::Float32);
    EXPECT_TRUE(choice.reason.empty());
}

/** @brief Tests that a policy reached through the interface decides what the cast pass checks. */
TEST(Small_PrecisionPolicy, APolicyOfItsOwnIsConsultedThroughTheInterface)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    SA_EXPECT_OK(builder.relu(x));

    const EverythingWide wide;
    const PrecisionPolicy& policy = wide;

    EXPECT_EQ(policy.name(), "wide");
    EXPECT_EQ(policy.choose(PrecisionQuery{Ops::K_RELU.id, DType::Float32}).dtype,
              DType::Float64);
    EXPECT_FALSE(Graph::find_dtype_mismatches(graph, policy).empty());
}
