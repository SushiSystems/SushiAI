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
 * @file test_step_executor.cpp
 * @brief Tests the work a Session hands its StepExecutor and how it treats a failed run.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "eager_run_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/session.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/step_recorder.hpp>

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

    /** @brief Holds what one execute() call was given. */
    struct SeenWork
    {
        std::size_t regions = 0;                         /**< How many regions the step held. */
        Eager::RegionKey highest_key = Eager::K_NO_REGION; /**< The highest region key. */
        const Graph::StepRecorder* trailing = nullptr;   /**< The trailing work, or null. */
        Eager::RegionKey trailing_key = Eager::K_NO_REGION; /**< The trailing work's key. */
        std::uint64_t run_index = 0;                     /**< The steps finished before it. */
    };

    /** @brief Runs nothing; records each work it is given and fails when told to. */
    class RecordingExecutor final : public Eager::StepExecutor
    {
        public:
            /** @brief Records @p work and returns success, or DeviceFailed when told to fail. */
            [[nodiscard]] Status execute(Eager::StepWork& work, Eager::StorageBank&) override
            {
                SeenWork seen;
                seen.regions = work.regions.size();
                for (const Eager::RegionWork& region : work.regions)
                {
                    if (region.order.key > seen.highest_key)
                        seen.highest_key = region.order.key;
                }
                seen.trailing = work.trailing;
                seen.trailing_key = work.trailing_key;
                seen.run_index = work.run_index;
                works.push_back(seen);

                if (!fail)
                    return Status();

                return Status::from_message(StatusCode::DeviceFailed, []()
                {
                    return std::string("the device was lost");
                });
            }

            /** @brief Counts the call and returns success. */
            [[nodiscard]] Status discard() override
            {
                ++discards;
                return Status();
            }

            std::vector<SeenWork> works; /**< What each execute() call was given. */
            std::size_t discards = 0;    /**< How often discard() ran. */
            bool fail = false;           /**< Whether execute() fails. */
    };

    /** @brief Owns a session over the two-layer model on a recording executor. */
    struct Rig
    {
        /** @brief Builds the session on @p engine. */
        explicit Rig(SushiBLAS::Engine& engine)
            : session(engine, executor, partitioner), update(K_RATE)
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

        RecordingExecutor executor;              /**< The executor under test. */
        Eager::ScopePartitioner partitioner;     /**< Cuts each step at its scopes. */
        Eager::Session session;                  /**< Hands each step to the executor. */
        Support::PlainDescent update;            /**< The update behind each step. */
        std::vector<Support::LinearState> model; /**< The layers, bound to the session. */
        SushiBLAS::Tensor inputs;                /**< The [K_ROWS, 6] batch. */
        SushiBLAS::Tensor targets;               /**< The [K_ROWS, 3] targets. */
    };
} // namespace

/** @brief Tests that a default work holds no region, no trailing work and run index 0. */
TEST(Small_StepWork, ADefaultWorkIsEmpty)
{
    const Eager::StepWork work;

    EXPECT_TRUE(work.regions.empty());
    EXPECT_TRUE(work.dropped.empty());
    EXPECT_EQ(work.storage_of_value.size(), 0u);
    EXPECT_EQ(work.trailing, nullptr);
    EXPECT_EQ(work.trailing_key, Eager::K_NO_REGION);
    EXPECT_EQ(work.run_index, 0u);
}

/** @brief Tests that each step reaches the executor once, numbered by the steps before it. */
TEST_F(Medium_StepExecutor, EachStepIsOneWorkNumberedByTheStepsBefore)
{
    Rig rig(*engine);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step(false));
    SA_EXPECT_OK(rig.step(false));

    ASSERT_EQ(rig.executor.works.size(), 2u);
    EXPECT_EQ(rig.executor.works[0].run_index, 0u);
    EXPECT_EQ(rig.executor.works[1].run_index, 1u);
    EXPECT_GT(rig.executor.works[0].regions, 1u);
    EXPECT_EQ(rig.executor.works[0].trailing, nullptr);
}

/** @brief Tests that an applied update arrives as the trailing work, keyed above every region. */
TEST_F(Medium_StepExecutor, TheUpdateIsTheTrailingWorkAboveEveryRegion)
{
    Rig rig(*engine);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step(true));

    ASSERT_EQ(rig.executor.works.size(), 1u);
    const SeenWork& seen = rig.executor.works[0];
    EXPECT_EQ(seen.trailing, &rig.update);
    EXPECT_GT(seen.trailing_key, seen.highest_key);
}

/** @brief Tests that a failed run reaches the caller, is discarded and counts no step. */
TEST_F(Medium_StepExecutor, AFailedRunIsDiscardedAndCountsNoStep)
{
    Rig rig(*engine);
    SA_EXPECT_OK(rig.prepare(*engine));
    rig.executor.fail = true;

    SA_EXPECT_FAILURE(rig.step(true), StatusCode::DeviceFailed, "the device was lost");

    EXPECT_EQ(rig.executor.discards, 1u);
    EXPECT_EQ(rig.session.step_count(), 0u);
}
