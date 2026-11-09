/**************************************************************************/
/* test_in_memory_fed_leaves.cpp                                         */
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
 * @file test_in_memory_fed_leaves.cpp
 * @brief Tests that InMemoryFedLeaves lists the leaves it was given and refuses the others.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/in_memory_fed_leaves.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Returns leaves of two names: a [1, 1] int32 and a [1, 3] float32. */
    InMemoryFedLeaves two_leaves()
    {
        return InMemoryFedLeaves({{"first", {Shape{1, 1}, DType::Int32}},
                                  {"second", {Shape{1, 3}, DType::Float32}}});
    }
} // namespace

/** @brief Tests that the names come back in the order the leaves were given. */
TEST(Small_InMemoryFedLeaves, TheNamesComeBackInTheOrderGiven)
{
    const InMemoryFedLeaves leaves = two_leaves();

    const auto names = leaves.names();

    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "first");
    EXPECT_EQ(names[1], "second");
}

/** @brief Tests that a leaf is answered with the shape and the type it was given. */
TEST(Small_InMemoryFedLeaves, ALeafHoldsTheFactsItWasGiven)
{
    const InMemoryFedLeaves leaves = two_leaves();

    SA_ASSERT_OK_AND_ASSIGN(const LeafFacts facts, leaves.leaf("second"));

    EXPECT_EQ(facts.shape.rank(), 2);
    EXPECT_EQ(facts.shape[1], 3);
    EXPECT_EQ(facts.dtype, DType::Float32);
}

/** @brief Tests that a leaf the step is not fed is refused by its name. */
TEST(Small_InMemoryFedLeaves, AnAbsentLeafIsRefused)
{
    const InMemoryFedLeaves leaves = two_leaves();

    SA_EXPECT_FAILURE(leaves.leaf("third"), StatusCode::InvalidDescription, "'third'");
}
