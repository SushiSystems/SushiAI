/**************************************************************************/
/* test_runtime_execution_backend.cpp                                     */
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
 * @file test_runtime_execution_backend.cpp
 * @brief Tests that RuntimeExecutionBackend keeps the StepTasks contract on the program's runtime.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/execution/runtime_execution_backend.hpp>
#include <support/step_tasks_contract.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Runs the StepTasks contract against the adapter over the program's runtime. */
    class Medium_StepTasksContract : public SaTest::EngineTest
    {
        protected:
            /// @brief The adapter under test, over the runtime and the engine of the program.
            SushiAI::RuntimeExecutionBackend backend{SaTest::TestEnvironment::runtime(),
                                                     SaTest::TestEnvironment::engine()};
    };
} // namespace

/** @brief Tests that a recorded task writes its values when the runtime tasks replay. */
TEST_F(Medium_StepTasksContract, ARecordedTaskWritesItsValuesOnReplay)
{
    SaTest::StepTasksContract::expect_recorded_task_writes_its_values_on_replay(backend);
}

/** @brief Tests that a second replay of the runtime tasks runs the tasks again. */
TEST_F(Medium_StepTasksContract, ASecondReplayRunsTheTasksAgain)
{
    SaTest::StepTasksContract::expect_second_replay_runs_the_tasks_again(backend);
}

/** @brief Tests that runtime hooks run before the tasks, in the order added. */
TEST_F(Medium_StepTasksContract, HooksRunBeforeTheTasksInTheOrderAdded)
{
    SaTest::StepTasksContract::expect_hooks_run_before_the_tasks_in_the_order_added(backend);
}

/** @brief Tests that a runtime recording with no task has a count of zero and replays as success. */
TEST_F(Medium_StepTasksContract, ARecordingWithNoTaskReplaysAsSuccess)
{
    SaTest::StepTasksContract::expect_recording_with_no_task_replays_as_success(backend);
}

/** @brief Tests that record_and_run calls its work once on the runtime. */
TEST_F(Medium_StepTasksContract, RecordAndRunRunsItsWorkOnce)
{
    SaTest::StepTasksContract::expect_record_and_run_runs_its_work_once(backend);
}
