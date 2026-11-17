/**************************************************************************/
/* test_run_preparation.cpp                                               */
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
 * @file test_run_preparation.cpp
 * @brief Tests that prepare_run() starts the run where the training file says.
 * @author Mustafa Garip
 */

#include "run_preparation.hpp"

#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>

#include "prepared_run_harness.hpp"
#include "tiny_yolo_files.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
namespace Prepared = SushiAI::Apps::PreparedTest;
namespace TinyYolo = SushiAI::Apps::TinyYoloTest;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_RunPreparation : public SaTest::EngineTest {};
} // namespace

/** @brief Tests that a file with no weights and no resume starts at its seed, batch 0, step 0. */
TEST_F(Medium_RunPreparation, AFileWithNoWeightsStartsAtItsSeedBeforeTheFirstBatch)
{
    const SaTest::ScratchFile checkpoint("sushiai_run_preparation_start.sushi");

    const auto harness = SaTest::require_value(Prepared::make_run_harness(backend(),
                                                                        checkpoint.path()));

    EXPECT_EQ(harness->prepared->start.seed, TinyYolo::K_SEED);
    EXPECT_EQ(harness->prepared->start.batch, 0u);
    EXPECT_EQ(harness->optimizer->step_count(), 0u);
}
