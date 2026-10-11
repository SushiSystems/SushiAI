/**************************************************************************/
/* test_program_runs.cpp                                                  */
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
 * @file test_program_runs.cpp
 * @brief Tests the run table, the fallback run, the report table and the start of a run.
 * @author Mustafa Garip
 * @see tests/integration/README.md
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <common/classifier_objective.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

#include "inference_run.hpp"
#include "run_start.hpp"
#include "trainer_run.hpp"
#include "training_run.hpp"

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_ProgramRuns : public SaTest::EngineTest {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 8;

    /** @brief Holds the epochs of the classification run. */
    constexpr std::int64_t K_EPOCHS = 2;

    /** @brief Holds the compile count the stub of a row reports. */
    constexpr std::size_t K_ROW_COUNT = 7;

    /** @brief Holds the compile count the stub of the fallback reports. */
    constexpr std::size_t K_FALLBACK_COUNT = 9;

    /**
     * @struct Pieces
     * @brief Holds what `sushiai train` builds before a run, over one engine.
     */
    struct Pieces
    {
        /** @brief Builds the pool on @p engine; make_pieces() fills the rest. */
        explicit Pieces(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief The model the graph is traced from. */
        MLP model = MLP(NN::Linear{K_FEATURES, 12}, NN::ReLU{}, NN::Linear{12, K_CLASSES});

        /** @brief The training graph of the model. */
        std::optional<Train::TrainingGraph> built;

        /** @brief The parameters of the model. */
        std::optional<NN::ModelState> state;

        /** @brief The pool the graph's tensors live in. */
        Graph::TensorPool pool;

        /** @brief The objective the graph was traced with. */
        std::optional<Train::CrossEntropyObjective> objective;

        /** @brief The optimizer over the parameters of the state. */
        std::unique_ptr<Optim::Optimizer> optimizer;

        /** @brief The batches of one epoch. */
        std::unique_ptr<Data::SyntheticClassification> dataset;

        /** @brief The training file; a test sets the keys it needs. */
        Config::TrainSpec spec;

        /** @brief The context a run of a domain reads; the fallback run reads none of it. */
        Train::ObjectiveContext given;

        /** @brief The probe trace a run of a domain reads; the fallback run reads none of it. */
        Train::TracedStep probe;
    };

    /**
     * @brief Builds the pieces of a classification run with initialised parameters.
     * @return The pieces, or the first failure of a trace, a bind, the optimizer or the data.
     */
    Result<std::unique_ptr<Pieces>> make_pieces(SushiBLAS::Engine& engine)
    {
        std::unique_ptr<Pieces> pieces = std::make_unique<Pieces>(engine);

        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(pieces->model, K_BATCH, K_FEATURES, K_CLASSES));
        pieces->built.emplace(std::move(built));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, pieces->model));
        pieces->state.emplace(std::move(state));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(*pieces->built, *pieces->state, pieces->pool));
        SA_TRY(NN::initialize(engine, *pieces->state));

        Optim::SGDOptions options;
        options.learning_rate = 0.1;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                      Optim::SGD::make(engine, std::move(slots), options));
        pieces->optimizer = std::move(optimizer);
        pieces->objective.emplace(SaTest::classifier_objective(*pieces->built));

        Data::SyntheticOptions data;
        data.batch_size = K_BATCH;
        data.batches = K_BATCHES;
        data.features = K_FEATURES;
        data.classes = K_CLASSES;
        SA_TRY_ASSIGN(pieces->dataset, Data::SyntheticClassification::make(engine, data));

        pieces->spec.epochs = K_EPOCHS;
        return std::move(pieces);
    }

    /** @brief Returns the run `sushiai train` would hand to the table for @p pieces. */
    Apps::PreparedRun prepare(Pieces& pieces, SushiBLAS::Engine& engine)
    {
        return Apps::PreparedRun{
            engine,         pieces.spec,       *pieces.dataset, pieces.given,
            *pieces.objective, pieces.probe,   *pieces.built,   *pieces.state,
            pieces.pool,    *pieces.optimizer, {},              nullptr,
        };
    }

    /** @brief Stands for the run of a row; reports K_ROW_COUNT and trains nothing. */
    Result<Apps::RunSummary> run_of_a_row(Apps::PreparedRun&)
    {
        Apps::RunSummary summary;
        summary.compile_count = K_ROW_COUNT;
        return summary;
    }

    /** @brief Stands for the fallback run; reports K_FALLBACK_COUNT and trains nothing. */
    Result<Apps::RunSummary> run_of_the_fallback(Apps::PreparedRun&)
    {
        Apps::RunSummary summary;
        summary.compile_count = K_FALLBACK_COUNT;
        return summary;
    }

    /** @brief Stands for a report; prints nothing and succeeds. */
    Status report_nothing(Apps::PreparedInference&)
    {
        return Status();
    }

    /** @brief Returns the number after each "loss " of @p text, in order. */
    std::vector<double> losses_of(const std::string& text)
    {
        const std::string key = "loss ";
        std::vector<double> losses;
        for (std::size_t at = text.find(key); at != std::string::npos; at = text.find(key, at))
        {
            at += key.size();
            losses.push_back(std::strtod(text.c_str() + at, nullptr));
        }

        return losses;
    }
} // namespace

/** @brief Tests that a row runs for its type, the fallback for any other, and neither twice. */
TEST_F(Integration_ProgramRuns, TheRunTablePicksARowOrTheFallback)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> pieces, make_pieces(*engine));
    Apps::PreparedRun prepared = prepare(*pieces, *engine);

    Apps::RunRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::RunEntry{"anchor", &run_of_a_row}));
    SA_EXPECT_FAILURE(registry.add(Apps::RunEntry{"anchor", &run_of_the_fallback}),
                      StatusCode::InvalidDescription, "'anchor'", "already registered");

    SA_EXPECT_FAILURE(Apps::run_training(registry, "cross_entropy", prepared),
                      StatusCode::InvalidDescription, "'cross_entropy'", "'anchor'");

    SA_EXPECT_OK(registry.set_fallback(&run_of_the_fallback));
    SA_EXPECT_FAILURE(registry.set_fallback(&run_of_a_row), StatusCode::InvalidDescription,
                      "already registered");

    SA_ASSERT_OK_AND_ASSIGN(const Apps::RunSummary of_row,
                            Apps::run_training(registry, "anchor", prepared));
    SA_ASSERT_OK_AND_ASSIGN(const Apps::RunSummary of_fallback,
                            Apps::run_training(registry, "cross_entropy", prepared));
    EXPECT_EQ(of_row.compile_count, K_ROW_COUNT);
    EXPECT_EQ(of_fallback.compile_count, K_FALLBACK_COUNT);
}

/** @brief Tests that the fallback run trains a classifier: one compile and a falling loss. */
TEST_F(Integration_ProgramRuns, AClassificationRunTrainsThroughTheFallback)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> pieces, make_pieces(*engine));
    Apps::PreparedRun prepared = prepare(*pieces, *engine);

    Apps::RunRegistry registry;
    SA_EXPECT_OK(Apps::register_runs(registry));
    ASSERT_NE(registry.fallback(), nullptr);

    testing::internal::CaptureStdout();
    const Result<Apps::RunSummary> summary =
        Apps::run_training(registry, "cross_entropy", prepared);
    const std::string printed = testing::internal::GetCapturedStdout();

    SA_EXPECT_OK(summary);
    ASSERT_TRUE(summary.has_value());
    EXPECT_EQ(summary.value().compile_count, 1u);
    EXPECT_EQ(pieces->optimizer->step_count(),
              static_cast<std::uint64_t>(K_EPOCHS * K_BATCHES));

    // The run prints the loss of the last batch of the first and of the last epoch.
    const std::vector<double> losses = losses_of(printed);
    ASSERT_EQ(losses.size(), static_cast<std::size_t>(K_EPOCHS)) << printed;
    EXPECT_LT(losses.back(), losses.front()) << printed;
}

/** @brief Tests that a report runs for its task type and an unknown type names the known. */
TEST_F(Integration_ProgramRuns, TheReportTableRefusesAnUnknownTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> pieces, make_pieces(*engine));
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(pieces->model, Shape{K_BATCH, K_FEATURES}));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(*engine, traced, *pieces->state, pool));

    const std::filesystem::path model_path("model.json");
    const std::filesystem::path weights_path("model.sushi");
    const Config::ModelSpec model{};
    const std::vector<NN::ModelOutputs::Entry> outputs;
    Apps::PreparedInference prepared{
        model_path, weights_path, model, 3, *pieces->dataset, *step, outputs, *engine,
    };

    Apps::InferenceRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::InferenceEntry{"classification", &report_nothing}));
    SA_EXPECT_FAILURE(registry.add(Apps::InferenceEntry{"classification", &report_nothing}),
                      StatusCode::InvalidDescription, "'classification'", "already registered");

    SA_EXPECT_OK(Apps::report_inference(registry, "classification", prepared));
    SA_EXPECT_FAILURE(Apps::report_inference(registry, "pose", prepared),
                      StatusCode::InvalidDescription, "model.json", "'pose'",
                      "'classification'");
}

/** @brief Tests that a run starts initialised, from weights, or resumed with its run state. */
TEST_F(Integration_ProgramRuns, ARunStartsFromWhatItsFileStates)
{
    const SaTest::ScratchFile file("sushiai_program_runs_resume.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> trained, make_pieces(*engine));
    Apps::PreparedRun prepared = prepare(*trained, *engine);
    Apps::RunRegistry registry;
    SA_EXPECT_OK(Apps::register_runs(registry));

    testing::internal::CaptureStdout();
    const Result<Apps::RunSummary> summary =
        Apps::run_training(registry, "cross_entropy", prepared);
    (void)testing::internal::GetCapturedStdout();
    SA_EXPECT_OK(summary);
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *trained->state, *trained->optimizer,
                                         *engine));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> fresh, make_pieces(*engine));
    EXPECT_EQ(Apps::describe_start(fresh->spec), "initialised");
    SA_EXPECT_OK(Apps::start_state(fresh->spec, *engine, *fresh->state, *fresh->optimizer));
    EXPECT_EQ(fresh->optimizer->step_count(), 0u);

    fresh->spec.weights = file.path();
    EXPECT_EQ(Apps::describe_start(fresh->spec), file.path().string());
    SA_EXPECT_OK(Apps::start_state(fresh->spec, *engine, *fresh->state, *fresh->optimizer));
    EXPECT_EQ(fresh->optimizer->step_count(), 0u);

    fresh->spec.weights.clear();
    fresh->spec.resume = file.path();
    EXPECT_EQ(Apps::describe_start(fresh->spec), "resumed from " + file.path().string());
    SA_EXPECT_OK(Apps::start_state(fresh->spec, *engine, *fresh->state, *fresh->optimizer));
    EXPECT_EQ(fresh->optimizer->step_count(), trained->optimizer->step_count());
    EXPECT_EQ(fresh->optimizer->step_count(),
              static_cast<std::uint64_t>(K_EPOCHS * K_BATCHES));
}
