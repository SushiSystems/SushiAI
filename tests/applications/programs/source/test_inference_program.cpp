/**************************************************************************/
/* test_inference_program.cpp                                             */
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
 * @file test_inference_program.cpp
 * @brief Tests run_inference_file over the tiny_yolo fixture: the anchor report.
 * @author Mustafa Garip
 */

#include "inference_program.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <gtest/gtest.h>
#include "tiny_yolo_files.hpp"
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
namespace TinyYolo = SushiAI::Apps::TinyYoloTest;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_InferenceProgram : public SaTest::EngineTest {};

    /** @brief Holds the status of a report and the text it printed. */
    struct PrintedReport
    {
        /** @brief What run_inference_file returned. */
        Status status;

        /** @brief What the report wrote to standard output. */
        std::string printed;
    };

    /**
     * @brief Trains the fixture to @p checkpoint, then reports that checkpoint.
     * @return The status of the report and the text it printed.
     */
    PrintedReport report_after_training(SushiBLAS::Engine& engine,
                                        const std::filesystem::path& checkpoint)
    {
        testing::internal::CaptureStdout();
        const Result<Apps::TrainingOutcome> trained =
            TinyYolo::train_the_fixture(engine, checkpoint);
        testing::internal::GetCapturedStdout();
        EXPECT_TRUE(trained.has_value()) << trained.status().message();

        testing::internal::CaptureStdout();
        const Status reported = TinyYolo::report_the_fixture(engine, checkpoint);
        return PrintedReport{reported, testing::internal::GetCapturedStdout()};
    }
} // namespace

/** @brief Tests that the anchor report succeeds on the checkpoint the run wrote. */
TEST_F(Medium_InferenceProgram, TheAnchorReportSucceedsOnTheCheckpointOfTheRun)
{
    const SaTest::ScratchFile checkpoint("sushiai_inference_program_report.sushi");

    const PrintedReport report = report_after_training(*engine, checkpoint.path());

    SA_EXPECT_OK(report.status);
}

/** @brief Tests that the anchor report prints the heads, the classes and the figures per image. */
TEST_F(Medium_InferenceProgram, TheAnchorReportPrintsItsFiguresOverEveryImage)
{
    const SaTest::ScratchFile checkpoint("sushiai_inference_program_figures.sushi");

    const PrintedReport report = report_after_training(*engine, checkpoint.path());

    EXPECT_NE(report.printed.find("1 heads, 2 classes"), std::string::npos) << report.printed;
    EXPECT_NE(report.printed.find("detections over " + std::to_string(TinyYolo::K_IMAGES)
                                  + " images."),
              std::string::npos)
        << report.printed;
    EXPECT_NE(report.printed.find("Mean average precision at IoU 0.5"), std::string::npos)
        << report.printed;
    EXPECT_NE(report.printed.find("of " + std::to_string(TinyYolo::K_IMAGES) + " images"),
              std::string::npos)
        << report.printed;
}

/** @brief Tests that the anchor report compiles one graph. */
TEST_F(Medium_InferenceProgram, TheAnchorReportCompilesOneGraph)
{
    const SaTest::ScratchFile checkpoint("sushiai_inference_program_compiles.sushi");

    const PrintedReport report = report_after_training(*engine, checkpoint.path());

    EXPECT_NE(report.printed.find("compiles: 1"), std::string::npos) << report.printed;
}

/** @brief Tests that a checkpoint path that names no file fails the report. */
TEST_F(Medium_InferenceProgram, ACheckpointThatDoesNotExistFailsTheReport)
{
    const SaTest::ScratchFile checkpoint("sushiai_inference_program_absent.sushi");

    testing::internal::CaptureStdout();
    const Status reported = TinyYolo::report_the_fixture(*engine, checkpoint.path());
    testing::internal::GetCapturedStdout();

    EXPECT_FALSE(reported.is_ok());
}
