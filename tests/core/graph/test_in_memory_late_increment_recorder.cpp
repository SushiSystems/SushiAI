/**************************************************************************/
/* test_in_memory_late_increment_recorder.cpp                             */
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
 * @file test_in_memory_late_increment_recorder.cpp
 * @brief Tests that InMemoryLateIncrementRecorder records the late-count task it promises.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_late_increment_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the host backend, a zeroed [4] tensor and a host count of 2. */
    class Small_InMemoryLateIncrementRecorder : public ::testing::Test
    {
        protected:
            /** @brief Allocates the tensor to increment and the host count. */
            void SetUp() override
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(backend.engine(), Shape{4});
                ASSERT_TRUE(made.has_value());
                out = std::move(made).value();
                SaTest::write_tensor(out, std::vector<double>(4, 0.0));

                count = backend.engine().create_tensor({1}, SushiBLAS::Core::DataType::Int32,
                                                       SushiRuntime::Residency::Host);
                count.data_as<std::int32_t>()[0] = 2;
            }

            /** @brief Replays @p tasks once. */
            static Status replay_once(StepTasks& tasks)
            {
                SushiRuntime::RunReport report;
                return tasks.replay(report);
            }

            HostExecutionBackend backend; /**< The adapter the task is recorded and replayed on. */
            SushiBLAS::Tensor out;        /**< The FLOAT32 tensor the task increments. */
            SushiBLAS::Tensor count;      /**< The INT32 host tensor holding the width. */
    };
} // namespace

/** @brief Tests that the recorder reports one task. */
TEST_F(Small_InMemoryLateIncrementRecorder, RecordsOneTask)
{
    Graph::InMemoryLateIncrementRecorder recorder(out, count);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t reported,
                            recorder.record_into_step(Graph::StepRecording(
                                tasks->recorder(), tasks->runtime_recorder())));

    EXPECT_EQ(reported, 1u);
}

/** @brief Tests that a replay adds one to as many leading elements as the count holds. */
TEST_F(Small_InMemoryLateIncrementRecorder, AReplayIncrementsAsManyElementsAsTheCountHolds)
{
    Graph::InMemoryLateIncrementRecorder recorder(out, count);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    SA_EXPECT_OK(recorder.record_into_step(
        Graph::StepRecording(tasks->recorder(), tasks->runtime_recorder())));
    SA_EXPECT_OK(tasks->seal());

    SA_EXPECT_OK(replay_once(*tasks));

    EXPECT_EQ(SaTest::read_tensor(out), (std::vector<double>{1.0, 1.0, 0.0, 0.0}));
}

/** @brief Tests that the task replays after the recorder that made it is gone. */
TEST_F(Small_InMemoryLateIncrementRecorder, TheTaskReplaysAfterItsRecorderIsGone)
{
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    {
        Graph::InMemoryLateIncrementRecorder recorder(out, count);
        SA_EXPECT_OK(recorder.record_into_step(
            Graph::StepRecording(tasks->recorder(), tasks->runtime_recorder())));
    }
    SA_EXPECT_OK(tasks->seal());

    SA_EXPECT_OK(replay_once(*tasks));
    SA_EXPECT_OK(replay_once(*tasks));

    EXPECT_EQ(SaTest::read_tensor(out), (std::vector<double>{2.0, 2.0, 0.0, 0.0}));
}
