/**************************************************************************/
/* test_training_program.cpp                                              */
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
 * @file test_training_program.cpp
 * @brief Tests run_training_file over the tiny_yolo fixture: the anchor row and its checkpoint.
 * @author Mustafa Garip
 */

#include "training_program.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include "tiny_yolo_files.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
namespace TinyYolo = SushiAI::Apps::TinyYoloTest;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_TrainingProgram : public SaTest::EngineTest {};

    /** @brief Holds the outcome of a training run and what it printed. */
    struct TrainedRun
    {
        /** @brief What run_training_file returned. */
        Result<Apps::TrainingOutcome> outcome;

        /** @brief What the run wrote to standard output. */
        std::string printed;
    };

    /** @brief Runs @p spec on @p engine and captures its standard output. */
    TrainedRun train_capturing(SushiAI::RuntimeExecutionBackend& backend,
                               const Config::TrainSpec& spec)
    {
        testing::internal::CaptureStdout();
        Result<Apps::TrainingOutcome> outcome = Apps::run_training_file("<test>", spec, backend);
        return TrainedRun{std::move(outcome), testing::internal::GetCapturedStdout()};
    }

    /** @brief Runs the fixture with its checkpoint written to @p checkpoint. */
    TrainedRun train_to(SushiAI::RuntimeExecutionBackend& backend,
                       const std::filesystem::path& checkpoint)
    {
        return train_capturing(backend, TinyYolo::training_spec(checkpoint));
    }
} // namespace

/** @brief Tests that the anchor row takes the two updates of one epoch of two batches. */
TEST_F(Medium_TrainingProgram, TheAnchorRowTakesTwoOptimizerSteps)
{
    const SaTest::ScratchFile checkpoint("sushiai_training_program_steps.sushi");

    const TrainedRun run = train_to(backend(), checkpoint.path());

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_EQ(run.outcome.value().optimizer_steps, TinyYolo::K_STEPS) << run.printed;
}

/** @brief Tests that the anchor row compiles one graph for all its updates. */
TEST_F(Medium_TrainingProgram, TheAnchorRowCompilesOnce)
{
    const SaTest::ScratchFile checkpoint("sushiai_training_program_compiles.sushi");

    const TrainedRun run = train_to(backend(), checkpoint.path());

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_EQ(run.outcome.value().summary.compile_count, 1u) << run.printed;
    EXPECT_NE(run.printed.find("compiles: 1"), std::string::npos) << run.printed;
}

/** @brief Tests that the anchor row writes the checkpoint its file names. */
TEST_F(Medium_TrainingProgram, TheAnchorRowWritesItsCheckpoint)
{
    const SaTest::ScratchFile checkpoint("sushiai_training_program_written.sushi");

    const TrainedRun run = train_to(backend(), checkpoint.path());

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_TRUE(std::filesystem::exists(checkpoint.path()));
}

/** @brief Tests that neither the header nor the checkpoint line counts zero learnable scalars. */
TEST_F(Medium_TrainingProgram, TheAnchorRowPrintsNoZeroScalarCount)
{
    const SaTest::ScratchFile checkpoint("sushiai_training_program_scalars.sushi");

    const TrainedRun run = train_to(backend(), checkpoint.path());

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_EQ(run.printed.find(" 0 learnable scalars"), std::string::npos) << run.printed;
    EXPECT_EQ(run.printed.find("(0 scalars)"), std::string::npos) << run.printed;
}

/** @brief Tests that a checkpoint in a folder that does not exist yet is written there. */
TEST_F(Medium_TrainingProgram, TheRunCreatesTheFolderOfItsCheckpoint)
{
    const SaTest::ScratchDirectory folder("sushiai_training_program_new_folder");
    const std::filesystem::path checkpoint = folder.path() / "nested" / "trained.sushi";

    const TrainedRun run = train_to(backend(), checkpoint);

    EXPECT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_TRUE(std::filesystem::exists(checkpoint));
}

/** @brief Tests that a checkpoint folder that cannot be created fails the run before it trains. */
TEST_F(Medium_TrainingProgram, ACheckpointFolderUnderAFileIsRefusedBeforeTheRunTrains)
{
    const SaTest::ScratchFile blocker("sushiai_training_program_blocker");
    std::ofstream(blocker.path()).put('x');
    const std::filesystem::path checkpoint = blocker.path() / "nested" / "trained.sushi";

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained =
        TinyYolo::train_the_fixture(backend(), checkpoint);
    const std::string printed = testing::internal::GetCapturedStdout();

    ASSERT_FALSE(trained.has_value()) << printed;
    EXPECT_TRUE(SaTest::status_names(trained.status(), StatusCode::InvalidFile,
                                     {"could not create", "the folder of the checkpoint"}));
    EXPECT_TRUE(printed.empty()) << printed;
}

/** @brief Tests that a file without a checkpoint trains the steps of its epoch. */
TEST_F(Medium_TrainingProgram, AFileWithoutACheckpointTrainsItsSteps)
{
    Config::TrainSpec spec = TinyYolo::training_spec("unused.sushi");
    spec.checkpoint.clear();

    const TrainedRun run = train_capturing(backend(), spec);

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_EQ(run.outcome.value().optimizer_steps, TinyYolo::K_STEPS) << run.printed;
}

/** @brief Tests that a file without a checkpoint prints no line about a written file. */
TEST_F(Medium_TrainingProgram, AFileWithoutACheckpointPrintsNoWroteLine)
{
    Config::TrainSpec spec = TinyYolo::training_spec("unused.sushi");
    spec.checkpoint.clear();

    const TrainedRun run = train_capturing(backend(), spec);

    EXPECT_EQ(run.printed.find("wrote "), std::string::npos) << run.printed;
}
