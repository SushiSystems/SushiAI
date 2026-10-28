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
 * @file test_training_program.cpp
 * @brief Tests run_training_file over the tiny_yolo fixture: the anchor row and its checkpoint.
 * @author Mustafa Garip
 */

#include "training_program.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include "registries.hpp"
#include "tiny_yolo_files.hpp"
#include "training_setup.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
namespace TinyYolo = SushiAI::Apps::TinyYoloTest;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_TrainingProgram : public SaTest::EngineTest {};
} // namespace

/** @brief Tests that the anchor row trains two updates in one compile and a checkpoint lands. */
TEST_F(Medium_TrainingProgram, TheAnchorRowTrainsTwoStepsAndWritesACheckpoint)
{
    const SaTest::ScratchFile checkpoint("sushiai_training_program_train.sushi");

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained =
        TinyYolo::train_the_fixture(*engine, checkpoint.path());
    const std::string printed = testing::internal::GetCapturedStdout();

    ASSERT_TRUE(trained.has_value()) << trained.status().message() << printed;
    EXPECT_EQ(trained.value().optimizer_steps, TinyYolo::K_STEPS) << printed;
    EXPECT_EQ(trained.value().summary.compile_count, 1u) << printed;
    EXPECT_TRUE(std::filesystem::exists(checkpoint.path()));
    EXPECT_EQ(printed.find(" 0 learnable scalars"), std::string::npos) << printed;
    EXPECT_EQ(printed.find("(0 scalars)"), std::string::npos) << printed;
    EXPECT_NE(printed.find("compiles: 1"), std::string::npos) << printed;
}

/** @brief Tests that a checkpoint in a folder that does not exist yet is written there. */
TEST_F(Medium_TrainingProgram, TheRunCreatesTheFolderOfItsCheckpoint)
{
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() / "sushiai_training_program_new_folder";
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    const std::filesystem::path checkpoint = folder / "nested" / "trained.sushi";

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained =
        TinyYolo::train_the_fixture(*engine, checkpoint);
    const std::string printed = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(trained.has_value()) << trained.status().message() << printed;
    EXPECT_TRUE(std::filesystem::exists(checkpoint));
    std::filesystem::remove_all(folder, ignored);
}

/** @brief Tests that a checkpoint folder that cannot be created fails the run before it trains. */
TEST_F(Medium_TrainingProgram, ACheckpointFolderUnderAFileIsRefusedBeforeTheRunTrains)
{
    const SaTest::ScratchFile blocker("sushiai_training_program_blocker");
    std::ofstream(blocker.path()).put('x');
    const std::filesystem::path checkpoint = blocker.path() / "nested" / "trained.sushi";

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained =
        TinyYolo::train_the_fixture(*engine, checkpoint);
    const std::string printed = testing::internal::GetCapturedStdout();

    ASSERT_FALSE(trained.has_value()) << printed;
    EXPECT_TRUE(SaTest::status_names(trained.status(), StatusCode::InvalidFile,
                                     {"could not create", "the folder of the checkpoint"}));
    EXPECT_TRUE(printed.empty()) << printed;
}

/** @brief Tests that a file without a checkpoint trains and prints no line about a written file. */
TEST_F(Medium_TrainingProgram, AFileWithoutACheckpointTrainsAndWritesNothing)
{
    Config::TrainSpec spec = TinyYolo::training_spec("unused.sushi");
    spec.checkpoint.clear();

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained =
        Apps::run_training_file("<test>", spec, *engine);
    const std::string printed = testing::internal::GetCapturedStdout();

    ASSERT_TRUE(trained.has_value()) << trained.status().message() << printed;
    EXPECT_EQ(trained.value().optimizer_steps, TinyYolo::K_STEPS) << printed;
    EXPECT_EQ(printed.find("wrote "), std::string::npos) << printed;
}

/** @brief Tests that a fresh anchor run starts the head's biases from the stride of its level. */
TEST_F(Medium_TrainingProgram, AFreshAnchorRunStartsTheObjectnessAndClassBiasesFromTheStride)
{
    // The fixture's head is 8 by 8 over a 16 by 16 input: stride 2, two anchors, two classes.
    constexpr std::int64_t K_SLOTS = 2;
    constexpr std::int64_t K_CHANNELS = Vision::Detect::Anchor::K_FIRST_CLASS_ENTRY + 2;
    // Two SGD updates at rate 0.001 move a bias by far less than this.
    constexpr double K_TOLERANCE = 0.5;
    const double objectness = std::log(8.0 / (320.0 * 320.0));
    const double classes = std::log(0.6 / (2.0 - 0.99));

    const SaTest::ScratchFile checkpoint("sushiai_training_program_bias.sushi");
    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained =
        TinyYolo::train_the_fixture(*engine, checkpoint.path());
    testing::internal::GetCapturedStdout();
    ASSERT_TRUE(trained.has_value()) << trained.status().message();

    const Config::ModelSpec model_spec =
        SaTest::require_value(Config::load_model(TinyYolo::fixture_directory() / "model.json"));
    const Config::TrainSpec spec = TinyYolo::training_spec(checkpoint.path());
    const Apps::Registries registries = Apps::compose_registries().value();
    const std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, spec, model_spec, *engine, TinyYolo::K_SEED)
            .value();
    SA_EXPECT_OK(IO::load_checkpoint(checkpoint.path(), setup->state()));

    const Vision::Detect::Anchor::AnchorTask task =
        Vision::Detect::Anchor::read_anchor_task(setup->given().task).value();
    const std::vector<Vision::Training::AnchorBiasHead> heads =
        Vision::Training::derive_anchor_bias_heads(model_spec, setup->given().input_sample, task,
                                                   setup->given().outputs)
            .value();
    ASSERT_EQ(heads.size(), 1u);
    EXPECT_EQ(heads[0].stride, 2);

    const SushiBLAS::Tensor* const bias = setup->state().tensor(heads[0].bias).value();
    const float* const values = bias->data_as<float>();
    for (std::int64_t slot = 0; slot < K_SLOTS; ++slot)
    {
        const float* const entries = values + slot * K_CHANNELS;
        EXPECT_NEAR(entries[Vision::Detect::Anchor::K_OBJECTNESS_ENTRY], objectness, K_TOLERANCE);
        EXPECT_NEAR(entries[Vision::Detect::Anchor::K_FIRST_CLASS_ENTRY], classes, K_TOLERANCE);
    }
}
