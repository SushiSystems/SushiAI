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
 * @file test_step.cpp
 * @brief Tests what a Step accepts while it traces and when it runs.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "eager_run_support.hpp"
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/traced.hpp>

using namespace SushiAI;
using Graph::ValueID;

namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Names the entry points of a step that hand back the refusal of a failed trace. */
    enum class EntryPoint : std::uint8_t
    {
        Backward,   /**< backward() of the refused value. */
        ReadScalar, /**< read_scalar() of the refused value. */
    };

    /** @brief Names the rows of the entry points; no case needs an engine of its own. */
    class Small_EagerStepRefusal : public ::testing::TestWithParam<EntryPoint>
    {
    };

    /** @brief Calls @p entry on the K_MATMUL of @p x with itself, which the builder refuses. */
    [[nodiscard]] Status call_with_refused_trace(Eager::Step& step, ValueID x, EntryPoint entry)
    {
        const Graph::Traced refused = step.builder().matmul(x, x);
        if (entry == EntryPoint::Backward)
            return step.backward(refused);

        return step.read_scalar(refused).status();
    }

    /** @brief Holds the step size of the update behind every step. */
    constexpr float K_RATE = 0.05f;

    /** @brief Holds the widths of the two-layer model: 6 inputs, 4 hidden, 3 results. */
    const std::vector<int64_t> K_TWO_LAYERS{6, 4, 3};

    /** @brief Holds the weight a blend gives the buffer's current content. */
    constexpr double K_MOMENTUM = 0.1;

    /** @brief Holds the width of the states that share one gradient value. */
    constexpr int64_t K_SHARED_WIDTH = 3;

    /** @brief Binds a buffer shaped like one row of the model input to @p loop and returns its id. */
    [[nodiscard]] Result<Graph::StateID> bind_row_buffer(Support::SmallLoop& loop,
                                                         Graph::StateIdentity& identity)
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor buffer,
                      allocate_tensor(loop.backend.engine(), Shape{K_TWO_LAYERS.front()}));
        const Graph::StateID state = identity.get();
        SA_TRY(loop.session.bind_state(state, buffer, false));
        return state;
    }
} // namespace

/** @brief Tests that asking a step for one state twice returns the one leaf it declared. */
TEST(Small_EagerStep, AStateIsDeclaredOncePerStep)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID first, step.state(loop.model[0].weight_state));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID again, step.state(loop.model[0].weight_state));

    EXPECT_EQ(first, again);
    EXPECT_EQ(step.graph().state_leaf(loop.model[0].weight_state), first);
}

/** @brief Tests that a tensor with no storage cannot be fed as an input. */
TEST(Small_EagerStep, AnEmptyInputIsRefused)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_EXPECT_FAILURE(step.input(SushiBLAS::Tensor{}, "batch"), StatusCode::InvalidDescription,
                      "batch", "no element");
    EXPECT_TRUE(step.graph().has_refusal());
}

/** @brief Tests that only a leaf can be fed: the result of an operation is refused. */
TEST(Small_EagerStep, FeedingTheResultOfAnOperationIsRefused)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID doubled, step.builder().scale(x, 2.0));

    SA_EXPECT_FAILURE(step.feed(doubled, loop.inputs), StatusCode::InvalidDescription,
                      "cannot be fed");
}

/** @brief Tests that keep() refuses a value id the step does not hold. */
TEST(Small_EagerStep, KeepingAnUnknownValueIsRefused)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));

    SA_EXPECT_FAILURE(step.keep(x + 100), StatusCode::InvalidDescription, "does not exist");
}

/** @brief Tests that backward() runs once per step and a second call is refused. */
TEST(Small_EagerStep, BackwardIsCalledOnce)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_OK(step.backward(loss));

    SA_EXPECT_FAILURE(step.backward(loss), StatusCode::InvalidDescription, "called once");
}

/** @brief Tests that read_scalar() runs the step but refuses a value of many elements. */
TEST(Small_EagerStep, ReadScalarRefusesAValueOfManyElements)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID positive, step.builder().relu(x));

    SA_EXPECT_FAILURE(step.read_scalar(positive), StatusCode::InvalidDescription,
                      "holds 48 elements");
    EXPECT_TRUE(step.has_run());
}

/** @brief Tests that an entry point of a step hands back the refusal a failed trace holds. */
TEST_P(Small_EagerStepRefusal, AnEntryPointReturnsTheRefusalOfAFailedTrace)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, step.input(loop.inputs, "inputs"));

    SA_EXPECT_FAILURE(call_with_refused_trace(step, x, GetParam()),
                      StatusCode::InvalidDescription, "MATMUL", "inner extents must agree");
    EXPECT_FALSE(step.has_run());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_EagerStepRefusal,
                         ::testing::Values(EntryPoint::Backward, EntryPoint::ReadScalar));

/** @brief Tests that backward() refuses a graph holding a refusal whose result was dropped. */
TEST(Small_EagerStep, BackwardRefusesAGraphHoldingARefusal)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_FAILURE(step.builder().matmul(loss, loss), StatusCode::InvalidDescription,
                      "MATMUL");

    SA_EXPECT_FAILURE(step.backward(loss), StatusCode::InvalidDescription,
                      "refused description");
}

/** @brief Tests that blending into a state the session does not hold is refused. */
TEST(Small_EagerStep, ABlendIntoAnUnboundStateIsRefused)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));
    Graph::StateIdentity stranger;

    Eager::Step step = loop.session.begin_step();
    const Graph::Traced x = step.input(loop.inputs, "inputs");

    SA_EXPECT_FAILURE(step.blend_buffer(stranger.get(), step.builder().sum_rows(x), K_MOMENTUM),
                      StatusCode::InvalidDescription, "not bound");
}

/** @brief Tests that a source with other extents than the buffer is refused. */
TEST(Small_EagerStep, ABlendOfASourceOfOtherExtentsIsRefused)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));
    Graph::StateIdentity identity;
    SA_ASSERT_OK_AND_ASSIGN(const Graph::StateID buffer, bind_row_buffer(loop, identity));

    Eager::Step step = loop.session.begin_step();
    const Graph::Traced x = step.input(loop.inputs, "inputs");

    SA_EXPECT_FAILURE(step.blend_buffer(buffer, x, K_MOMENTUM), StatusCode::Incompatible,
                      "source");
}

/** @brief Tests that a step that blends a buffer hands the executor trailing work. */
TEST(Small_EagerStep, ABlendGivesTheStepTrailingWork)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));
    Graph::StateIdentity identity;
    SA_ASSERT_OK_AND_ASSIGN(const Graph::StateID buffer, bind_row_buffer(loop, identity));

    Eager::Step step = loop.session.begin_step();
    const Graph::Traced x = step.input(loop.inputs, "inputs");
    SA_EXPECT_OK(step.blend_buffer(buffer, step.builder().sum_rows(x), K_MOMENTUM));
    SA_EXPECT_OK(step.finish());

    ASSERT_EQ(loop.executor.seen().size(), 1u);
    EXPECT_TRUE(loop.executor.seen().front().has_trailing);
}

/** @brief Tests that a step with no update and no blend hands the executor no trailing work. */
TEST(Small_EagerStep, AStepWithNoUpdateAndNoBlendHasNoTrailingWork)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_OK(step.keep(loss));
    SA_EXPECT_OK(step.finish());

    ASSERT_EQ(loop.executor.seen().size(), 1u);
    EXPECT_FALSE(loop.executor.seen().front().has_trailing);
}

/**
 * @brief Tests that two states whose gradient is one value run, and the second gets a copy of
 *        the first one's gradient behind the regions.
 */
TEST(Small_EagerStep, TwoStatesSharingOneGradientValueRunWithoutARefusal)
{
    Support::SmallLoop loop(K_RATE);
    const Shape shape{Support::K_ROWS, K_SHARED_WIDTH};
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor first_tensor,
                            allocate_tensor(loop.backend.engine(), shape));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor second_tensor,
                            allocate_tensor(loop.backend.engine(), shape));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor targets,
                            Support::make_batch(loop.backend.engine(), K_SHARED_WIDTH, 1.75));
    Graph::StateIdentity first_identity;
    Graph::StateIdentity second_identity;
    const Graph::StateID first = first_identity.get();
    const Graph::StateID second = second_identity.get();
    SA_EXPECT_OK(loop.session.bind_state(first, first_tensor, true));
    SA_EXPECT_OK(loop.session.bind_state(second, second_tensor, true));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID a, step.state(first));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID b, step.state(second));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            step.builder().nll_loss(step.builder().add(a, b),
                                                    step.input(targets, "targets")));
    SA_EXPECT_OK(step.backward(loss));
    SA_EXPECT_OK(step.finish());

    // No update was applied, so the only trailing work is the copy one gradient needs.
    ASSERT_EQ(loop.executor.seen().size(), 1u);
    EXPECT_TRUE(loop.executor.seen().front().has_trailing);
}

/** @brief Tests that finish() runs the step once and a second call changes nothing. */
TEST(Small_EagerStep, FinishRunsTheStepOnce)
{
    Support::SmallLoop loop(K_RATE);
    SA_EXPECT_OK(loop.prepare(K_TWO_LAYERS));

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
