/**************************************************************************/
/* test_step.cpp                                                          */
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
 * @file test_step.cpp
 * @brief Tests what a Step accepts while it traces and when it runs.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "eager_run_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>

using namespace SushiAI;
using Graph::ValueID;

// SaTest stays reachable inside a TEST_F body; see tests/core/eager/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Medium_EagerStep : public SaTest::EngineTest
    {
    };

    /** @brief Holds the step size of the update behind every step. */
    constexpr float K_RATE = 0.05f;

    /** @brief Holds the widths of the two-layer model: 6 inputs, 4 hidden, 3 results. */
    const std::vector<int64_t> K_TWO_LAYERS{6, 4, 3};
} // namespace

/** @brief Tests that asking a step for one state twice returns the one leaf it declared. */
TEST_F(Medium_EagerStep, AStateIsDeclaredOncePerStep)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID first, step.state(loop.model[0].weight_state));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID again, step.state(loop.model[0].weight_state));

    EXPECT_EQ(first, again);
    EXPECT_EQ(step.graph().state_leaf(loop.model[0].weight_state), first);
}

/** @brief Tests that a tensor with no storage cannot be fed as an input. */
TEST_F(Medium_EagerStep, AnEmptyInputIsRefused)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_EXPECT_FAILURE(step.input(SushiBLAS::Tensor{}, "batch"), StatusCode::InvalidDescription,
                      "batch", "no element");
    EXPECT_TRUE(step.graph().has_refusal());
}

/** @brief Tests that only a leaf can be fed: the result of an operation is refused. */
TEST_F(Medium_EagerStep, FeedingTheResultOfAnOperationIsRefused)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID doubled, step.builder().scale(x, 2.0));

    SA_EXPECT_FAILURE(step.feed(doubled, loop.inputs), StatusCode::InvalidDescription,
                      "cannot be fed");
}

/** @brief Tests that keep() refuses a value id the step does not hold. */
TEST_F(Medium_EagerStep, KeepingAnUnknownValueIsRefused)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));

    SA_EXPECT_FAILURE(step.keep(x + 100), StatusCode::InvalidDescription, "does not exist");
}

/** @brief Tests that backward() runs once per step and a second call is refused. */
TEST_F(Medium_EagerStep, BackwardIsCalledOnce)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_OK(step.backward(loss));

    SA_EXPECT_FAILURE(step.backward(loss), StatusCode::InvalidDescription, "called once");
}

/** @brief Tests that read_scalar() runs the step but refuses a value of many elements. */
TEST_F(Medium_EagerStep, ReadScalarRefusesAValueOfManyElements)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID positive, step.builder().relu(x));

    SA_EXPECT_FAILURE(step.read_scalar(positive), StatusCode::InvalidDescription,
                      "holds 48 elements");
    EXPECT_TRUE(step.has_run());
}

/** @brief Tests that backward() hands back the refusal a failed trace of the loss holds. */
TEST_F(Medium_EagerStep, BackwardReturnsTheRefusalOfAFailedTrace)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));
    const Graph::Traced refused = step.builder().matmul(x, x);

    SA_EXPECT_FAILURE(step.backward(refused), StatusCode::InvalidDescription, "MATMUL",
                      "inner extents must agree");
    EXPECT_FALSE(step.has_run());
}

/** @brief Tests that read_scalar() hands back the refusal a failed trace holds. */
TEST_F(Medium_EagerStep, ReadScalarReturnsTheRefusalOfAFailedTrace)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));

    SA_EXPECT_FAILURE(step.read_scalar(step.builder().matmul(x, x)),
                      StatusCode::InvalidDescription, "MATMUL", "inner extents must agree");
    EXPECT_FALSE(step.has_run());
}

/** @brief Tests that backward() refuses a graph holding a refusal whose result was dropped. */
TEST_F(Medium_EagerStep, BackwardRefusesAGraphHoldingARefusal)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_FAILURE(step.builder().matmul(loss, loss), StatusCode::InvalidDescription,
                      "MATMUL");

    SA_EXPECT_FAILURE(step.backward(loss), StatusCode::InvalidDescription,
                      "refused description");
}

/** @brief Tests that finish() runs the step once and a second call changes nothing. */
TEST_F(Medium_EagerStep, FinishRunsTheStepOnce)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_OK(step.keep(loss));
    EXPECT_FALSE(step.has_run());

    SA_EXPECT_OK(step.finish());
    EXPECT_TRUE(step.has_run());
    SA_EXPECT_OK(step.finish());

    EXPECT_EQ(loop.session.step_count(), 1u);
}
