/**************************************************************************/
/* test_step_recording.cpp                                                */
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
 * @file test_step_recording.cpp
 * @brief Tests that StepRecording hands back the two recorders it was built with.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/graph/step_recording.hpp>

/** @brief Tests that tasks() and graph() return the recorders given, by address. */
TEST(Medium_StepRecording, HandsBackTheRecordersItWasBuiltWith)
{
    SushiBLAS::LateCountHold late_counts(SushiAI::Test::TestEnvironment::runtime());
    SushiRuntime::Graph graph = SushiAI::Test::TestEnvironment::runtime().graph();
    SushiBLAS::ScratchPool scratch(SushiAI::Test::TestEnvironment::runtime());
    SushiBLAS::TaskRecorder tasks(graph.recorder(), scratch, late_counts);

    const SushiAI::Graph::StepRecording recording(tasks, graph.recorder());

    EXPECT_EQ(&recording.tasks(), &tasks);
    EXPECT_EQ(&recording.graph(), &graph.recorder());
}
