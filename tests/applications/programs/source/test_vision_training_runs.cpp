/**************************************************************************/
/* test_vision_training_runs.cpp                                          */
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
 * @file test_vision_training_runs.cpp
 * @brief Tests the anchor and anchor-free run rows over a prepared run of the tiny_yolo fixture.
 * @author Mustafa Garip
 */

#include "vision_training_runs.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
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
    class Medium_VisionTrainingRuns : public SaTest::EngineTest {};

    /** @brief Holds what a run row returned and the text it printed. */
    struct PrintedRun
    {
        /** @brief What the row returned. */
        Result<Apps::RunSummary> outcome;

        /** @brief What the row wrote to standard output. */
        std::string printed;
    };

    /** @brief Builds the harness with its checkpoint at @p checkpoint, or stops with the refusal. */
    std::unique_ptr<Prepared::RunHarness> built_harness(SushiAI::ExecutionBackend& backend,
                                                        const std::filesystem::path& checkpoint)
    {
        return SaTest::require_value(Prepared::make_run_harness(backend, checkpoint));
    }

    /** @brief Runs the anchor row over @p harness and captures what it prints. */
    PrintedRun anchor_run(SushiAI::ExecutionBackend& backend, Prepared::RunHarness& harness)
    {
        testing::internal::CaptureStdout();
        Result<Apps::RunSummary> outcome = Apps::run_anchor(backend, *harness.prepared);
        return PrintedRun{std::move(outcome), testing::internal::GetCapturedStdout()};
    }

    /** @brief Runs the anchor-free row over @p harness and captures what it prints. */
    PrintedRun free_run(SushiAI::ExecutionBackend& backend, Prepared::RunHarness& harness)
    {
        testing::internal::CaptureStdout();
        Result<Apps::RunSummary> outcome = Apps::run_anchor_free(backend, *harness.prepared);
        return PrintedRun{std::move(outcome), testing::internal::GetCapturedStdout()};
    }
} // namespace

/** @brief Tests that the anchor row takes one update for each batch of each epoch. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorRowTakesOneUpdateForEveryBatchOfEveryEpoch)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_updates.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());

    const PrintedRun run = anchor_run(backend(), *harness);

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_EQ(harness->optimizer->step_count(), TinyYolo::K_STEPS);
}

/** @brief Tests that the anchor row compiles its training step once for all its updates. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorRowCompilesItsTrainingStepOnce)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_compiles.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());

    const PrintedRun run = anchor_run(backend(), *harness);

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_EQ(run.outcome.value().compile_count, 1u);
}

/** @brief Tests that the summary names the seed, the runs of the step and the epoch that follows. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorRowReportsWhereTheRunStands)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_progress.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());

    const PrintedRun run = anchor_run(backend(), *harness);

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_EQ(run.outcome.value().progress.seed, TinyYolo::K_SEED);
    EXPECT_EQ(run.outcome.value().progress.step_runs, TinyYolo::K_STEPS);
    EXPECT_EQ(run.outcome.value().progress.epoch, 1u);
    EXPECT_EQ(run.outcome.value().progress.batch, 0u);
}

/** @brief Tests that the anchor row prints the progress line of the first epoch. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorRowPrintsTheProgressLineOfItsFirstEpoch)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_line.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());

    const PrintedRun run = anchor_run(backend(), *harness);

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_NE(run.printed.find("  epoch   1/1   loss "), std::string::npos) << run.printed;
}

/** @brief Tests that a run which starts from nothing starts the biases from the model file. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorRowStartsItsBiasesFromTheModelFileWhenStartingFromNothing)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_bias_read.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());
    harness->spec.model_path = harness->spec.model_path.parent_path() / "absent_model.json";

    const PrintedRun run = anchor_run(backend(), *harness);

    SA_EXPECT_FAILURE(run.outcome, StatusCode::FileNotFound, "absent_model.json");
}

/** @brief Tests that a run which reads weights leaves the bias start, and so the model file, alone. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorRowLeavesItsBiasesAloneWhenWeightsAreRead)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_bias_skipped.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());
    harness->spec.model_path = harness->spec.model_path.parent_path() / "absent_model.json";
    harness->spec.weights = "weights.sushi";

    const PrintedRun run = anchor_run(backend(), *harness);

    EXPECT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
}

/** @brief Tests that a start batch the data lacks is refused naming the batch. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorRowRefusesAStartBatchTheDataLacks)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_start_batch.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());
    harness->prepared->start.batch = 99;

    const PrintedRun run = anchor_run(backend(), *harness);

    SA_EXPECT_FAILURE(run.outcome, StatusCode::Incompatible, "batch 99");
    EXPECT_EQ(harness->optimizer->step_count(), 0u);
}

/** @brief Tests that the anchor-free row refuses a model whose task is anchor based, training nothing. */
TEST_F(Medium_VisionTrainingRuns, TheAnchorFreeRowRefusesAModelWhoseTaskIsAnchorBased)
{
    const SaTest::ScratchFile checkpoint("sushiai_vision_runs_free_refused.sushi");
    const auto harness = built_harness(backend(), checkpoint.path());

    const PrintedRun run = free_run(backend(), *harness);

    EXPECT_FALSE(run.outcome.has_value());
    EXPECT_EQ(harness->optimizer->step_count(), 0u);
    EXPECT_TRUE(run.printed.empty()) << run.printed;
}
