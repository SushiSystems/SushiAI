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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_inference_program.cpp
 * @brief Tests run_inference_file over the tiny_yolo fixture: the anchor report.
 * @author Mustafa Garip
 */

#include "inference_program.hpp"

#include <cstddef>
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
} // namespace

/** @brief Tests that the anchor report reads the checkpoint the run wrote, over every image. */
TEST_F(Medium_InferenceProgram, TheAnchorReportReadsTheCheckpointOfTheRun)
{
    const SaTest::ScratchFile checkpoint("sushiai_inference_program_report.sushi");

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained =
        TinyYolo::train_the_fixture(*engine, checkpoint.path());
    testing::internal::GetCapturedStdout();
    ASSERT_TRUE(trained.has_value()) << trained.status().message();

    testing::internal::CaptureStdout();
    const Status reported = TinyYolo::report_the_fixture(*engine, checkpoint.path());
    const std::string printed = testing::internal::GetCapturedStdout();

    SA_EXPECT_OK(reported);
    EXPECT_NE(printed.find("1 heads, 2 classes"), std::string::npos) << printed;
    EXPECT_NE(printed.find("detections over " + std::to_string(TinyYolo::K_IMAGES) + " images."),
              std::string::npos)
        << printed;
    EXPECT_NE(printed.find("Mean average precision at IoU 0.5"), std::string::npos) << printed;
    EXPECT_NE(printed.find("of " + std::to_string(TinyYolo::K_IMAGES) + " images"),
              std::string::npos)
        << printed;
    EXPECT_NE(printed.find("compiles: 1"), std::string::npos) << printed;
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
