/**************************************************************************/
/* test_engine_smoke.cpp                                                  */
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
 * @file test_engine_smoke.cpp
 * @brief Tests that the umbrella header gives an engine and a task recorder that run.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiAI/SushiAI.hpp>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

/**
 * @brief Tests that the umbrella header gives an engine and a task recorder whose graph runs.
 *
 * The value of an axpy is SushiBLAS's own promise and is tested there.
 */
TEST(Medium_EngineSmoke, TheUmbrellaGivesAnEngineAndATaskRecorderThatRun)
{
    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime);

    auto x = engine.create_tensor({4});
    auto y = engine.create_tensor({4});

    sb::LateCountHold late_counts(runtime);
    sr::Graph graph = runtime.graph();
    sb::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), late_counts,
                              engine.mode());
    sb::BLASOps(recorder).axpy(2.0f, x, y);

    EXPECT_TRUE(graph.run().has_value());
    EXPECT_TRUE(runtime.wait().has_value());
}
