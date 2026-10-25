/**************************************************************************/
/* test_anchor_run.cpp                                                    */
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
 * @file test_anchor_run.cpp
 * @brief Tests the anchor run row and the anchor inference report over a YOLO-layout fixture.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "inference_program.hpp"
#include "registries.hpp"
#include "training_setup.hpp"
#include "training_program.hpp"
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_AnchorRun : public SaTest::EngineTest {};

    /** @brief Holds the images of the fixture: two batches of two. */
    constexpr std::size_t K_IMAGES = 4;

    /** @brief Holds the seed the training file states. */
    constexpr std::uint64_t K_SEED = 7;

    /** @brief Holds the updates of the run: one epoch of two batches. */
    constexpr std::uint64_t K_STEPS = 2;

    /** @brief Returns the directory of the fixture: the model file and the YOLO-layout dataset. */
    std::filesystem::path fixture_directory()
    {
#ifdef SA_TEST_FIXTURE_DIR
        return std::filesystem::path(SA_TEST_FIXTURE_DIR) / "tiny_yolo";
#else
        return {};
#endif
    }

    /** @brief Returns the data block both files state: the fixture read as a detection task. */
    std::string data_block()
    {
        return R"({"reader": "yolo_txt", "task": "detection", "root": ")"
               + fixture_directory().generic_string()
               + R"(", "split": "train", "batch_size": 2, "input": [16, 16],
                   "max_objects": 8, "classes": 2})";
    }

    /** @brief Returns the training file: the anchor objective, one epoch, @p checkpoint. */
    std::string training_text(const std::filesystem::path& checkpoint)
    {
        return R"({
            "sushiai_train": 3,
            "model": "model.json",
            "objective": {"type": "anchor"},
            "data": )" + data_block() + R"(,
            "optimizer": {"type": "sgd", "momentum": 0.9, "learning_rate": 0.001},
            "epochs": 1,
            "seed": 7,
            "checkpoint": ")" + checkpoint.generic_string() + R"("
        })";
    }

    /**
     * @brief Runs `sushiai train` over the fixture through the assembly the program runs.
     * @return The outcome of the run, or the first failure of the setup, the run or the save.
     */
    Result<Apps::TrainingOutcome> train_the_fixture(SushiBLAS::Engine& engine,
                                                    const std::filesystem::path& checkpoint)
    {
        const Config::TrainSpec spec = Config::parse_train(
            Config::parse_json(training_text(checkpoint), "<test>"), "<test>",
            fixture_directory());
        return Apps::run_training_file("<test>", spec, engine);
    }

    /**
     * @brief Runs `sushiai inference` over the fixture through the assembly the program runs,
     *        which prints the report of the task type of the model file.
     * @return The first failure of the build, the load or the report.
     */
    Status report_the_fixture(SushiBLAS::Engine& engine, const std::filesystem::path& checkpoint)
    {
        const std::filesystem::path model_path = fixture_directory() / "model.json";
        const Config::ModelSpec model_spec = Config::load_model(model_path);
        const Config::DataSpec data_spec = Config::parse_data(
            Config::parse_json(R"({"sushiai_data": 1, "data": )" + data_block() + "}", "<test>"),
            "<test>");
        return Apps::run_inference_file(model_path, checkpoint, model_spec, data_spec, engine);
    }

} // namespace

/** @brief Tests that the anchor row trains two updates in one compile and a checkpoint lands. */
TEST_F(Medium_AnchorRun, TheAnchorRowTrainsTwoStepsAndWritesACheckpoint)
{
    const SaTest::ScratchFile checkpoint("sushiai_anchor_run_train.sushi");

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained = train_the_fixture(*engine, checkpoint.path());
    const std::string printed = testing::internal::GetCapturedStdout();

    ASSERT_TRUE(trained.has_value()) << trained.status().message() << printed;
    EXPECT_EQ(trained.value().optimizer_steps, K_STEPS) << printed;
    EXPECT_EQ(trained.value().summary.compile_count, 1u) << printed;
    EXPECT_TRUE(std::filesystem::exists(checkpoint.path()));
}

/** @brief Tests that the anchor report reads the checkpoint the run wrote, over every image. */
TEST_F(Medium_AnchorRun, TheAnchorReportReadsTheCheckpointOfTheRun)
{
    const SaTest::ScratchFile checkpoint("sushiai_anchor_run_report.sushi");

    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained = train_the_fixture(*engine, checkpoint.path());
    testing::internal::GetCapturedStdout();
    ASSERT_TRUE(trained.has_value()) << trained.status().message();

    testing::internal::CaptureStdout();
    const Status reported = report_the_fixture(*engine, checkpoint.path());
    const std::string printed = testing::internal::GetCapturedStdout();

    SA_EXPECT_OK(reported);
    EXPECT_NE(printed.find("1 heads, 2 classes"), std::string::npos) << printed;
    EXPECT_NE(printed.find("detections over " + std::to_string(K_IMAGES) + " images."),
              std::string::npos)
        << printed;
    EXPECT_NE(printed.find("Mean average precision at IoU 0.5"), std::string::npos) << printed;
    EXPECT_NE(printed.find("of " + std::to_string(K_IMAGES) + " images"), std::string::npos)
        << printed;
}

/** @brief Tests that a fresh anchor run starts the head's biases from the stride of its level. */
TEST_F(Medium_AnchorRun, AFreshAnchorRunStartsTheObjectnessAndClassBiasesFromTheStride)
{
    // The fixture's head is 8 by 8 over a 16 by 16 input: stride 2, two anchors, two classes.
    constexpr std::int64_t K_SLOTS = 2;
    constexpr std::int64_t K_CHANNELS = Vision::Detect::Anchor::K_FIRST_CLASS_ENTRY + 2;
    // Two SGD updates at rate 0.001 move a bias by far less than this.
    constexpr double K_TOLERANCE = 0.5;
    const double objectness = std::log(8.0 / (320.0 * 320.0));
    const double classes = std::log(0.6 / (2.0 - 0.99));

    const SaTest::ScratchFile checkpoint("sushiai_anchor_run_bias.sushi");
    testing::internal::CaptureStdout();
    const Result<Apps::TrainingOutcome> trained = train_the_fixture(*engine, checkpoint.path());
    testing::internal::GetCapturedStdout();
    ASSERT_TRUE(trained.has_value()) << trained.status().message();

    const Config::ModelSpec model_spec = Config::load_model(fixture_directory() / "model.json");
    const Config::TrainSpec spec = Config::parse_train(
        Config::parse_json(training_text(checkpoint.path()), "<test>"), "<test>",
        fixture_directory());
    const Apps::Registries registries = Apps::compose_registries().value();
    const std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, spec, model_spec, *engine, K_SEED).value();
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
