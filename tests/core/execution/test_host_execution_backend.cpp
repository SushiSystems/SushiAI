/**************************************************************************/
/* test_host_execution_backend.cpp                                        */
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
 * @file test_host_execution_backend.cpp
 * @brief Tests that HostExecutionBackend keeps the StepTasks contract on the host.
 * @author Mustafa Garip
 */

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/random/random.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/step_tasks_contract.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Runs the StepTasks contract against the adapter that needs no runtime. */
    class Small_StepTasksContract : public ::testing::Test
    {
        protected:
            /// @brief The adapter under test; each test makes its own.
            SushiAI::HostExecutionBackend backend;
    };
} // namespace

/** @brief Tests that a recorded task writes its values when the host tasks replay. */
TEST_F(Small_StepTasksContract, ARecordedTaskWritesItsValuesOnReplay)
{
    SaTest::StepTasksContract::expect_recorded_task_writes_its_values_on_replay(backend);
}

/** @brief Tests that a second replay of the host tasks runs the tasks again. */
TEST_F(Small_StepTasksContract, ASecondReplayRunsTheTasksAgain)
{
    SaTest::StepTasksContract::expect_second_replay_runs_the_tasks_again(backend);
}

/** @brief Tests that host hooks run before the tasks, in the order added. */
TEST_F(Small_StepTasksContract, HooksRunBeforeTheTasksInTheOrderAdded)
{
    SaTest::StepTasksContract::expect_hooks_run_before_the_tasks_in_the_order_added(backend);
}

/** @brief Tests that a host recording with no task has a count of zero and replays as success. */
TEST_F(Small_StepTasksContract, ARecordingWithNoTaskReplaysAsSuccess)
{
    SaTest::StepTasksContract::expect_recording_with_no_task_replays_as_success(backend);
}

/** @brief Tests that record_and_run calls its work once on the host. */
TEST_F(Small_StepTasksContract, RecordAndRunRunsItsWorkOnce)
{
    SaTest::StepTasksContract::expect_record_and_run_runs_its_work_once(backend);
}

/** @brief Tests that the host tasks report the origin each task was recorded under. */
TEST_F(Small_StepTasksContract, TheTasksReportTheOriginEachWasRecordedUnder)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target,
                            SushiAI::allocate_tensor(backend.engine(), SushiAI::Shape{4}));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SushiAI::StepTasks> tasks, backend.make_tasks());
    SushiBLAS::RandomOps random(tasks->recorder(), backend.engine().rng());

    {
        const SushiBLAS::TaskRecorder::OriginScope scope = tasks->recorder().origin("node.a");
        random.constant(target, 1.0);
    }
    random.constant(target, 2.0);

    EXPECT_EQ(tasks->task_origins(), (std::vector<std::string>{"node.a", ""}));
}
