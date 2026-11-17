/**************************************************************************/
/* test_trailing_chain.cpp                                                */
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
 * @file test_trailing_chain.cpp
 * @brief Tests that TrailingChain records the copies, the caller's recorder and the blends of a spec.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/trailing_chain.hpp>
#include <SushiAI/eager/trailing_spec.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using Call = Graph::InMemoryStepRecorder::Call;
using Eager::StorageBank;
using Eager::TrailingBlend;
using Eager::TrailingChain;
using Eager::TrailingCopy;
using Eager::TrailingSpec;
using Graph::InMemoryStepRecorder;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Names the tensor the regions wrote and the copy reads. */
    constexpr Eager::StorageID K_GRADIENT = 1;

    /** @brief Names the tensor the copy fills. */
    constexpr Eager::StorageID K_COPIED = 2;

    /** @brief Names the running statistic the blend writes. */
    constexpr Eager::StorageID K_RUNNING = 3;

    /** @brief Names the value the blend reads. */
    constexpr Eager::StorageID K_BATCH = 4;

    /** @brief Holds the tasks the caller's recorder reports. */
    constexpr std::size_t K_INNER_TASKS = 5;

    /** @brief Provides the host backend and a bank of four two-element tensors on it. */
    class Small_TrailingChain : public ::testing::Test
    {
        protected:
            /** @brief Fills the bank: the gradient 1, 2; the copy 0, 0; the running 10, 10; the batch 2, 4. */
            void SetUp() override
            {
                bank = std::make_unique<StorageBank>(backend.engine());
                for (const Eager::StorageID id : {K_GRADIENT, K_COPIED, K_RUNNING, K_BATCH})
                    SA_EXPECT_OK(bank->allocate(id, Shape{2}, DType::Float64));

                SaTest::write_tensor(bank->tensor(K_GRADIENT), std::vector<double>{1.0, 2.0});
                SaTest::write_tensor(bank->tensor(K_COPIED), std::vector<double>{0.0, 0.0});
                SaTest::write_tensor(bank->tensor(K_RUNNING), std::vector<double>{10.0, 10.0});
                SaTest::write_tensor(bank->tensor(K_BATCH), std::vector<double>{2.0, 4.0});
            }

            /** @brief Returns the spec of one copy, the caller's recorder and one blend of weight one half. */
            TrailingSpec full_spec(Graph::StepRecorder* inner) const
            {
                TrailingSpec spec;
                spec.inner = inner;
                spec.copies = {TrailingCopy{K_GRADIENT, K_COPIED}};
                spec.blends = {TrailingBlend{K_RUNNING, K_BATCH, 0.5, 1.0}};
                return spec;
            }

            /** @brief Replays @p tasks once and returns its status. */
            static Status replay_once(StepTasks& tasks)
            {
                SushiRuntime::RunReport report;
                return tasks.replay(report);
            }

            /** @brief The host backend the bank allocates from. */
            HostExecutionBackend backend;

            /** @brief The bank the spec's ids are resolved in. */
            std::unique_ptr<StorageBank> bank;
    };
} // namespace

/** @brief Tests that the chain reports one task per copy, the caller's tasks and two per blend. */
TEST_F(Small_TrailingChain, RecordingReportsTheCopiesTheCallersTasksAndTheBlends)
{
    InMemoryStepRecorder inner(K_INNER_TASKS);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TrailingChain> chain,
                            TrailingChain::make(full_spec(&inner), *bank));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    const Graph::StepRecording step(tasks->recorder(), tasks->runtime_recorder());

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t recorded, chain->record_into_step(step));

    EXPECT_EQ(recorded, 1u + K_INNER_TASKS + 2u);
    EXPECT_EQ(inner.calls(), std::vector<Call>{Call::Record});
}

/** @brief Tests that one replay copies the gradient and blends the batch value into the statistic. */
TEST_F(Small_TrailingChain, OneReplayCopiesTheGradientAndBlendsTheBatchValue)
{
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TrailingChain> chain,
                            TrailingChain::make(full_spec(nullptr), *bank));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    const Graph::StepRecording step(tasks->recorder(), tasks->runtime_recorder());
    SA_EXPECT_OK(chain->record_into_step(step));
    SA_EXPECT_OK(tasks->seal());

    SA_EXPECT_OK(replay_once(*tasks));

    // The blend is (1 - 0.5) * running + 0.5 * 1.0 * batch; every product is exact in binary.
    EXPECT_EQ(SaTest::read_tensor(bank->tensor(K_COPIED)), (std::vector<double>{1.0, 2.0}));
    EXPECT_EQ(SaTest::read_tensor(bank->tensor(K_RUNNING)), (std::vector<double>{6.0, 7.0}));
}

/** @brief Tests that a spec of no copy and no blend records only the caller's tasks. */
TEST_F(Small_TrailingChain, ASpecOfNoCopyAndNoBlendRecordsOnlyTheCallersTasks)
{
    InMemoryStepRecorder inner(K_INNER_TASKS);
    TrailingSpec spec;
    spec.inner = &inner;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TrailingChain> chain, TrailingChain::make(spec, *bank));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    const Graph::StepRecording step(tasks->recorder(), tasks->runtime_recorder());

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t recorded, chain->record_into_step(step));

    EXPECT_EQ(recorded, K_INNER_TASKS);
}

/** @brief Tests that the failure of the caller's recorder is the failure of the chain. */
TEST_F(Small_TrailingChain, TheFailureOfTheCallersRecorderIsTheFailureOfTheChain)
{
    InMemoryStepRecorder inner(K_INNER_TASKS);
    inner.fail_with(Status::from_message(StatusCode::InvalidDescription,
                                         []() { return std::string("scripted recorder failure"); }));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TrailingChain> chain,
                            TrailingChain::make(full_spec(&inner), *bank));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    const Graph::StepRecording step(tasks->recorder(), tasks->runtime_recorder());

    SA_EXPECT_FAILURE(chain->record_into_step(step), StatusCode::InvalidDescription,
                      "scripted recorder failure");
}

/** @brief Tests that the replay hooks of the chain reach the caller's recorder once each. */
TEST_F(Small_TrailingChain, TheReplayHooksReachTheCallersRecorder)
{
    InMemoryStepRecorder inner(K_INNER_TASKS);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TrailingChain> chain,
                            TrailingChain::make(full_spec(&inner), *bank));

    chain->before_replay();
    chain->after_replay();

    EXPECT_EQ(inner.calls(), (std::vector<Call>{Call::BeforeReplay, Call::AfterReplay}));
}
