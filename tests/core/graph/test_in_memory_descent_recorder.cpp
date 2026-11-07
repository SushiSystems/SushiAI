/**************************************************************************/
/* test_in_memory_descent_recorder.cpp                                    */
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
 * @file test_in_memory_descent_recorder.cpp
 * @brief Tests that InMemoryDescentRecorder records the descent it promises on the host.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_descent_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the host backend the descent is recorded and replayed on. */
    class Small_InMemoryDescentRecorder : public ::testing::Test
    {
        protected:
            /** @brief Allocates a [4] tensor holding @p values on the host engine. */
            [[nodiscard]] SushiBLAS::Tensor vector_of(const std::vector<double>& values)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(backend.engine(), Shape{4});
                EXPECT_TRUE(made.has_value());
                SushiBLAS::Tensor tensor = std::move(made).value();
                SaTest::write_tensor(tensor, values);
                return tensor;
            }

            /** @brief Holds the adapter the tasks are recorded and replayed on. */
            HostExecutionBackend backend;
    };
} // namespace

/** @brief Tests that a fresh recorder has recorded and replayed nothing. */
TEST_F(Small_InMemoryDescentRecorder, StartsWithNoCall)
{
    const Graph::InMemoryDescentRecorder recorder(0.5f);

    EXPECT_EQ(recorder.record_count(), 0u);
    EXPECT_EQ(recorder.replay_count(), 0u);
}

/** @brief Tests that replaying the recorded task subtracts rate times the gradient. */
TEST_F(Small_InMemoryDescentRecorder, TheRecordedTaskSubtractsTheScaledGradient)
{
    SushiBLAS::Tensor parameter = vector_of({1.0, 2.0, 3.0, 4.0});
    const SushiBLAS::Tensor gradient = vector_of({10.0, 20.0, 30.0, 40.0});
    Graph::InMemoryDescentRecorder recorder(0.5f);
    recorder.add(parameter, gradient);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count,
                            recorder.record_into_step(Graph::StepRecording(
                                tasks->recorder(), tasks->runtime_recorder())));
    SA_EXPECT_OK(tasks->seal());
    SushiRuntime::RunReport report;
    SA_EXPECT_OK(tasks->replay(report));

    EXPECT_EQ(count, 1u);
    EXPECT_EQ(SaTest::read_tensor(parameter), (std::vector<double>{-4.0, -8.0, -12.0, -16.0}));
}

/** @brief Tests that the recorder counts the records and the replays it is told of. */
TEST_F(Small_InMemoryDescentRecorder, CountsItsRecordsAndItsReplays)
{
    Graph::InMemoryDescentRecorder recorder(1.0f);
    Graph::StepRecorder& port = recorder;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

    SA_EXPECT_OK(port.record_into_step(
        Graph::StepRecording(tasks->recorder(), tasks->runtime_recorder())));
    port.after_replay();
    port.after_replay();

    EXPECT_EQ(recorder.record_count(), 1u);
    EXPECT_EQ(recorder.replay_count(), 2u);
}
