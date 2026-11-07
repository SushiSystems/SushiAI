/**************************************************************************/
/* test_step_executor.cpp                                                 */
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
 * @file test_step_executor.cpp
 * @brief Tests the work a Session hands its StepExecutor and how it treats a failed run.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "eager_run_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/in_memory_step_executor.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/session.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::ValueID;

// SaTest stays reachable inside a TEST_F body; see tests/core/eager/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Medium_StepExecutor : public SaTest::EngineTest
    {
    };

    /** @brief Holds the step size of the update behind every step. */
    constexpr float K_RATE = 0.05f;

    /** @brief Holds the widths of the two-layer model: 6 inputs, 4 hidden, 3 results. */
    const std::vector<int64_t> K_TWO_LAYERS{6, 4, 3};

    /** @brief Owns a session over the two-layer model on an in-memory executor. */
    struct Rig
    {
        /** @brief Builds the session on @p backend. */
        explicit Rig(SushiAI::RuntimeExecutionBackend& backend)
            : session(backend, executor, partitioner), update(K_RATE)
        {

        }

        /** @brief Allocates the model and one batch and binds every state to the session. */
        [[nodiscard]] Status prepare(SushiBLAS::Engine& engine)
        {
            SA_TRY_ASSIGN(model, Support::make_model(engine, K_TWO_LAYERS));
            SA_TRY_ASSIGN(inputs, Support::make_batch(engine, K_TWO_LAYERS.front(), 0.25));
            SA_TRY_ASSIGN(targets, Support::make_batch(engine, K_TWO_LAYERS.back(), 1.75));
            return Support::bind_model(session, model, update);
        }

        /** @brief Traces one scoped step and runs it, with the update when @p apply is set. */
        [[nodiscard]] Status step(bool apply)
        {
            Support::TraceOptions options;
            options.scoping = Support::Scoping::PerLayer;
            Eager::Step step = session.begin_step();
            const Graph::Traced loss =
                Support::trace_model(step, model, inputs, targets, options);
            SA_TRY(step.backward(loss));
            return apply ? step.apply(update) : step.finish();
        }

        Eager::InMemoryStepExecutor executor;    /**< The executor under test. */
        Eager::ScopePartitioner partitioner;     /**< Cuts each step at its scopes. */
        Eager::Session session;                  /**< Hands each step to the executor. */
        Support::PlainDescent update;            /**< The update behind each step. */
        std::vector<Support::LinearState> model; /**< The layers, bound to the session. */
        SushiBLAS::Tensor inputs;                /**< The [K_ROWS, 6] batch. */
        SushiBLAS::Tensor targets;               /**< The [K_ROWS, 3] targets. */
    };
} // namespace

/** @brief Tests that each step reaches the executor once, numbered by the steps before it. */
TEST_F(Medium_StepExecutor, EachStepIsOneWorkNumberedByTheStepsBefore)
{
    Rig rig(backend());
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step(false));
    SA_EXPECT_OK(rig.step(false));

    ASSERT_EQ(rig.executor.seen().size(), 2u);
    EXPECT_EQ(rig.executor.seen()[0].run_index, 0u);
    EXPECT_EQ(rig.executor.seen()[1].run_index, 1u);
    EXPECT_GT(rig.executor.seen()[0].regions, 1u);
    EXPECT_FALSE(rig.executor.seen()[0].has_trailing);
}

/** @brief Tests that an applied update arrives as the trailing work, keyed above every region. */
TEST_F(Medium_StepExecutor, TheUpdateIsTheTrailingWorkAboveEveryRegion)
{
    Rig rig(backend());
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step(true));

    ASSERT_EQ(rig.executor.seen().size(), 1u);
    const Eager::InMemoryStepExecutor::Seen& seen = rig.executor.seen()[0];
    EXPECT_TRUE(seen.has_trailing);
    EXPECT_GT(seen.trailing_key, seen.highest_key);
}

/** @brief Tests that a failed run reaches the caller, is discarded and counts no step. */
TEST_F(Medium_StepExecutor, AFailedRunIsDiscardedAndCountsNoStep)
{
    Rig rig(backend());
    SA_EXPECT_OK(rig.prepare(*engine));
    rig.executor.script_execute(Status::from_message(StatusCode::DeviceFailed, []()
    {
        return std::string("the device was lost");
    }));

    SA_EXPECT_FAILURE(rig.step(true), StatusCode::DeviceFailed, "the device was lost");

    EXPECT_EQ(rig.executor.discard_count(), 1u);
    EXPECT_EQ(rig.session.step_count(), 0u);
}
