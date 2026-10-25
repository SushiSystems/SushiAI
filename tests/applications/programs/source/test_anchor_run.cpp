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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include "inference_run.hpp"
#include "registries.hpp"
#include "training_run.hpp"
#include "training_setup.hpp"
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/data_fit.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/train/evaluation_step.hpp>

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
     * @brief Runs `sushiai train` over the fixture as the program does, up to the checkpoint.
     * @param summary Takes what the run reports.
     * @param steps Takes the updates the optimizer has taken.
     * @return The first failure of the setup, the run or the save.
     */
    Status train_the_fixture(SushiBLAS::Engine& engine, const std::filesystem::path& checkpoint,
                             Apps::RunSummary& summary, std::uint64_t& steps)
    {
        const Config::TrainSpec spec = Config::parse_train(
            Config::parse_json(training_text(checkpoint), "<test>"), "<test>",
            fixture_directory());
        const Config::ModelSpec model_spec = Config::load_model(spec.model_path);

        SA_TRY_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
        SA_TRY_ASSIGN(const std::unique_ptr<Apps::TrainingSetup> setup,
                      Apps::TrainingSetup::make(registries, spec, model_spec, engine, K_SEED));
        SA_TRY_ASSIGN(const std::unique_ptr<Optim::Optimizer> optimizer,
                      Optim::make_optimizer(engine, spec.optimizer_type, setup->parameters(),
                                            spec.optimizer_attributes, "the optimizer block"));

        Apps::PreparedRun prepared{
            engine, spec, setup->dataset(), setup->given(), setup->objective(), setup->built(),
            setup->state(), setup->pool(), *optimizer, setup->take_updates(), nullptr, {},
        };
        SA_TRY_ASSIGN(summary,
                      Apps::run_training(registries.runs, spec.objective_type, prepared));
        steps = optimizer->step_count();
        return IO::save_run_checkpoint(spec.checkpoint, setup->state(), *optimizer,
                                       summary.progress);
    }

    /**
     * @brief Runs `sushiai inference` over the fixture as the program does, through the report
     *        table, and prints the report of the task type of the model file.
     * @return The first failure of the build, the load or the report.
     */
    Status report_the_fixture(SushiBLAS::Engine& engine, const std::filesystem::path& checkpoint)
    {
        const std::filesystem::path model_path = fixture_directory() / "model.json";
        const Config::ModelSpec model_spec = Config::load_model(model_path);
        const Config::DataSpec data_spec = Config::parse_data(
            Config::parse_json(R"({"sushiai_data": 1, "data": )" + data_block() + "}", "<test>"),
            "<test>");

        SA_TRY_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
        SA_TRY_ASSIGN(const Config::Value fitted,
                      Config::fit_data_to_model(data_spec.attributes, model_spec,
                                                model_path.string()));
        SA_TRY_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                      Data::make_dataset(registries.datasets, engine, data_spec.type, fitted,
                                         "the data block"));

        SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(registries.nodes, model_spec));
        SA_TRY_ASSIGN(const Shape batch_shape,
                      Train::batch_shape(dataset->batch_size(), dataset->sample_shape()));
        Train::GraphOptions options;
        options.dtype = model_spec.input_dtype;
        SA_TRY_ASSIGN(const Train::TracedStep traced,
                      Train::build_evaluation_graph(model, batch_shape, options));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        SA_TRY(IO::load_checkpoint(checkpoint, state));
        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                      Train::EvaluationStep::make(engine, traced, state, pool));

        Apps::PreparedInference prepared{
            model_path, checkpoint, model_spec, model.size(), *dataset, *step,
            traced.outputs.all(), engine, &model, &state,
        };
        const std::string& task =
            model_spec.task.at("type", "the task section").as_string("the task type");
        return Apps::report_inference(registries.inferences, task, prepared);
    }

} // namespace

/** @brief Tests that the anchor row trains two updates in one compile and a checkpoint lands. */
TEST_F(Medium_AnchorRun, TheAnchorRowTrainsTwoStepsAndWritesACheckpoint)
{
    const SaTest::ScratchFile checkpoint("sushiai_anchor_run_train.sushi");
    Apps::RunSummary summary;
    std::uint64_t steps = 0;

    testing::internal::CaptureStdout();
    const Status trained = train_the_fixture(*engine, checkpoint.path(), summary, steps);
    const std::string printed = testing::internal::GetCapturedStdout();

    SA_EXPECT_OK(trained);
    EXPECT_EQ(steps, K_STEPS) << printed;
    EXPECT_EQ(summary.compile_count, 1u) << printed;
    EXPECT_TRUE(std::filesystem::exists(checkpoint.path()));
}

/** @brief Tests that the anchor report reads the checkpoint the run wrote, over every image. */
TEST_F(Medium_AnchorRun, TheAnchorReportReadsTheCheckpointOfTheRun)
{
    const SaTest::ScratchFile checkpoint("sushiai_anchor_run_report.sushi");
    Apps::RunSummary summary;
    std::uint64_t steps = 0;

    testing::internal::CaptureStdout();
    const Status trained = train_the_fixture(*engine, checkpoint.path(), summary, steps);
    testing::internal::GetCapturedStdout();
    ASSERT_TRUE(trained.is_ok()) << trained.message();

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
