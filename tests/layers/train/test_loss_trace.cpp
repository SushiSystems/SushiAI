/**************************************************************************/
/* test_loss_trace.cpp                                                  */
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
 * @file test_loss_trace.cpp
 * @brief Tests the values an objective keeps from loss() and the terms of a composite.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/train/loss_trace.hpp>

using namespace SushiAI;
using Train::LossTrace;

namespace
{
    /** @brief Names the suite of the cases that expect the process to stop. */
    class Medium_LossTraceDeathTest : public ::testing::Test {};
} // namespace

/** @brief Tests that a default trace holds no loss, no value and no term. */
TEST(Small_LossTrace, ADefaultTraceHoldsNothing)
{
    const LossTrace trace;

    EXPECT_EQ(trace.loss(), Graph::K_NO_VALUE);
    EXPECT_TRUE(trace.all().empty());
    EXPECT_EQ(trace.term(0).loss(), Graph::K_NO_VALUE);
}

/** @brief Tests that a trace of a loss gives the loss back, and only through loss(). */
TEST(Small_LossTrace, ATraceOfALossReadsAsTheLoss)
{
    const LossTrace trace(Graph::ValueID{7});

    EXPECT_EQ(trace.loss(), Graph::ValueID{7});
    EXPECT_FALSE((std::is_convertible<LossTrace, Graph::ValueID>::value));
}

/** @brief Tests that a kept value is found by its name and the values keep their order. */
TEST(Small_LossTrace, KeptValuesAreFoundByNameInOrder)
{
    LossTrace trace(Graph::ValueID{1});
    trace.add("rows", Graph::ValueID{4});
    trace.add("weights", Graph::ValueID{9});

    EXPECT_TRUE(trace.has("rows"));
    EXPECT_FALSE(trace.has("columns"));
    EXPECT_EQ(trace.at("weights"), Graph::ValueID{9});
    ASSERT_EQ(trace.all().size(), 2u);
    EXPECT_EQ(trace.all().front().first, "rows");
    EXPECT_EQ(trace.all().back().first, "weights");
}

/** @brief Tests that the trace of each term of a composite is found by the term's position. */
TEST(Small_LossTrace, TermsAreFoundByPosition)
{
    LossTrace first(Graph::ValueID{2});
    first.add("rows", Graph::ValueID{5});
    LossTrace trace(Graph::ValueID{8});
    trace.add_term(first);
    trace.add_term(LossTrace(Graph::ValueID{3}));

    EXPECT_EQ(trace.term(0).at("rows"), Graph::ValueID{5});
    EXPECT_EQ(trace.term(1).loss(), Graph::ValueID{3});
    EXPECT_FALSE(trace.term(1).has("rows"));
}

/** @brief Tests that a position past the recorded terms answers a trace that holds nothing. */
TEST(Small_LossTrace, APositionPastTheTermsAnswersAnEmptyTrace)
{
    LossTrace trace(Graph::ValueID{8});
    trace.add_term(LossTrace(Graph::ValueID{3}));

    EXPECT_EQ(trace.term(1).loss(), Graph::K_NO_VALUE);
    EXPECT_TRUE(trace.term(1).all().empty());
}

/** @brief Tests that keeping a name twice stops the process naming it. */
TEST_F(Medium_LossTraceDeathTest, KeepingANameTwiceIsFatalNamingIt)
{
    LossTrace trace;
    trace.add("rows", Graph::ValueID{4});

    SA_EXPECT_FATAL(trace.add("rows", Graph::ValueID{5}), "kept the value 'rows' twice");
}

/** @brief Tests that reading a name that was not kept stops the process listing what is held. */
TEST_F(Medium_LossTraceDeathTest, ReadingAMissingNameIsFatalListingTheHeldOnes)
{
    LossTrace trace;
    trace.add("rows", Graph::ValueID{4});

    SA_EXPECT_FATAL((void)trace.at("columns"), "holds no value 'columns'", "'rows'");
}
