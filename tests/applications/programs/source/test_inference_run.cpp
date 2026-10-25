/**************************************************************************/
/* test_inference_run.cpp                                                 */
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
 * @file test_inference_run.cpp
 * @brief Tests the report table of inference_run and the task it refuses.
 * @author Mustafa Garip
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
#include "inference_run.hpp"
#include "run_start.hpp"
#include "trainer_run.hpp"
#include "training_run.hpp"
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    class Medium_InferenceRun : public SaTest::EngineTest {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;

    constexpr int64_t K_FEATURES = 6;

    constexpr int64_t K_CLASSES = 4;

    constexpr int64_t K_BATCHES = 8;

    /** @brief Holds the epochs of the classification run. */
    constexpr std::int64_t K_EPOCHS = 2;

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

    /** @brief Stands for a report; prints nothing and succeeds. */
    Status report_nothing(Apps::PreparedInference&)
    {
        return Status();
    }

} // namespace

/** @brief Tests that a report runs for its task type and an unknown type names the known. */
TEST_F(Medium_InferenceRun, TheReportTableRefusesAnUnknownTask)
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
