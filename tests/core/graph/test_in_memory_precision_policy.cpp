/**************************************************************************/
/* test_in_memory_precision_policy.cpp                                    */
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
 * @file test_in_memory_precision_policy.cpp
 * @brief Tests that InMemoryPrecisionPolicy keeps the promise of the PrecisionPolicy interface.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/in_memory_precision_policy.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/ops/ops.hpp>

using namespace SushiAI;
using Graph::InMemoryPrecisionPolicy;
using Graph::PrecisionChoice;
using Graph::PrecisionPolicy;
using Graph::PrecisionQuery;

/** @brief Tests that an operation with no script keeps the type it declared. */
TEST(Small_InMemoryPrecisionPolicy, AnUnscriptedOperationKeepsItsDeclaredType)
{
    const InMemoryPrecisionPolicy policy;
    const PrecisionPolicy& port = policy;

    const PrecisionChoice choice = port.choose(PrecisionQuery{Ops::K_ADD.id, DType::Float64});

    EXPECT_EQ(choice.dtype, DType::Float64);
    EXPECT_EQ(choice.reason, "declared");
}

/** @brief Tests that a scripted operation gets its scripted choice and no other does. */
TEST(Small_InMemoryPrecisionPolicy, AScriptedOperationGetsItsScriptedChoice)
{
    InMemoryPrecisionPolicy policy;
    policy.script(Ops::K_MATMUL.id, PrecisionChoice{DType::Half, "scripted"});

    const PrecisionChoice scripted =
        policy.choose(PrecisionQuery{Ops::K_MATMUL.id, DType::Float32});
    const PrecisionChoice other = policy.choose(PrecisionQuery{Ops::K_ADD.id, DType::Float32});

    EXPECT_EQ(scripted.dtype, DType::Half);
    EXPECT_EQ(scripted.reason, "scripted");
    EXPECT_EQ(other.dtype, DType::Float32);
}

/** @brief Tests that a second script for one operation replaces the first. */
TEST(Small_InMemoryPrecisionPolicy, ASecondScriptReplacesTheFirst)
{
    InMemoryPrecisionPolicy policy;
    policy.script(Ops::K_MATMUL.id, PrecisionChoice{DType::Half, "first"});
    policy.script(Ops::K_MATMUL.id, PrecisionChoice{DType::Float64, "second"});

    const PrecisionChoice choice = policy.choose(PrecisionQuery{Ops::K_MATMUL.id, DType::Float32});

    EXPECT_EQ(choice.dtype, DType::Float64);
    EXPECT_EQ(choice.reason, "second");
}

/** @brief Tests that the same query gets the same choice every time, as the interface requires. */
TEST(Small_InMemoryPrecisionPolicy, TheSameQueryGetsTheSameChoice)
{
    InMemoryPrecisionPolicy policy;
    policy.script(Ops::K_MATMUL.id, PrecisionChoice{DType::Half, "scripted"});
    const PrecisionQuery query{Ops::K_MATMUL.id, DType::Float32};

    const PrecisionChoice first = policy.choose(query);
    const PrecisionChoice second = policy.choose(query);

    EXPECT_EQ(first.dtype, second.dtype);
    EXPECT_EQ(first.reason, second.reason);
}

/** @brief Tests that the journal keeps each query in order and the policy names itself. */
TEST(Small_InMemoryPrecisionPolicy, TheJournalKeepsEachQueryInOrder)
{
    const InMemoryPrecisionPolicy policy;

    static_cast<void>(policy.choose(PrecisionQuery{Ops::K_ADD.id, DType::Float32}));
    static_cast<void>(policy.choose(PrecisionQuery{Ops::K_MATMUL.id, DType::Float64}));

    ASSERT_EQ(policy.queries().size(), 2u);
    EXPECT_EQ(policy.queries()[0].op, Ops::K_ADD.id);
    EXPECT_EQ(policy.queries()[1].declared, DType::Float64);
    EXPECT_EQ(policy.name(), std::string_view("in-memory"));
}
