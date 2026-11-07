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
    Result<std::unique_ptr<Pieces>> make_pieces(SushiAI::ExecutionBackend& backend)
    {
        SushiBLAS::Engine& engine = backend.engine();
        std::unique_ptr<Pieces> pieces = std::make_unique<Pieces>(engine);

        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(pieces->model, K_BATCH, K_FEATURES, K_CLASSES));
        pieces->built.emplace(std::move(built));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, pieces->model));
        pieces->state.emplace(std::move(state));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, *pieces->built, *pieces->state,
                                             pieces->pool));
        SA_TRY(NN::initialize(backend, *pieces->state));

        Optim::SGDOptions options;
        options.learning_rate = 0.1;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                      Optim::SGD::make(backend, std::move(slots), options));
        pieces->optimizer = std::move(optimizer);
        pieces->objective.emplace(SaTest::classifier_objective(*pieces->built));

        Data::SyntheticOptions data;
        data.batch_size = K_BATCH;
        data.batches = K_BATCHES;
        data.features = K_FEATURES;
        data.classes = K_CLASSES;
        SA_TRY_ASSIGN(pieces->dataset, Data::SyntheticClassification::make(backend, data));

        pieces->spec.epochs = K_EPOCHS;
        return std::move(pieces);
    }

    /** @brief Stands for a report; prints nothing and succeeds. */
    Status report_nothing(Apps::PreparedInference&)
    {
        return Status();
    }

    /** @brief Stands for a report that fails, so a run of it can be told from a refusal. */
    Status report_failure(Apps::PreparedInference&)
    {
        SA_FAIL_IF(true, StatusCode::DeviceFailed, "the classification row ran");
        return Status();
    }

    /**
     * @struct InferenceCase
     * @brief Holds what a report of the table runs over, on one engine.
     */
    struct InferenceCase
    {
        /** @brief Builds the pool on @p engine; make_case() fills the rest. */
        explicit InferenceCase(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief The pieces of a classification run. */
        std::unique_ptr<Pieces> pieces;

        /** @brief The evaluation graph the step runs. */
        std::optional<Train::TracedStep> traced;

        /** @brief The pool the evaluation step lives in. */
        Graph::TensorPool pool;

        /** @brief The evaluation step over the state of the pieces. */
        std::unique_ptr<Train::EvaluationStep> step;

        /** @brief The path of the model file the report names. */
        std::filesystem::path model_path = "model.json";

        /** @brief The path of the weights file the report names. */
        std::filesystem::path weights_path = "model.sushi";

        /** @brief The model specification the report reads. */
        Config::ModelSpec model;

        /** @brief The outputs of the model; none. */
        std::vector<NN::ModelOutputs::Entry> outputs;

        /** @brief The prepared report over the members above. */
        std::optional<Apps::PreparedInference> prepared;
    };

    /** @brief Builds the case on @p engine, or returns the first failure of its pieces. */
    Result<std::unique_ptr<InferenceCase>> make_case(SushiAI::ExecutionBackend& backend)
    {
        std::unique_ptr<InferenceCase> made = std::make_unique<InferenceCase>(backend.engine());
        SA_TRY_ASSIGN(made->pieces, make_pieces(backend));
        SA_TRY_ASSIGN(Train::TracedStep traced,
                      Train::build_evaluation_graph(made->pieces->model,
                                                    Shape{K_BATCH, K_FEATURES}));
        made->traced.emplace(std::move(traced));
        SA_TRY_ASSIGN(made->step, Train::EvaluationStep::make(backend, *made->traced,
                                                              *made->pieces->state, made->pool));
        made->prepared.emplace(Apps::PreparedInference{
            made->model_path, made->weights_path, made->model, 3, *made->pieces->dataset,
            *made->step, made->outputs, backend,
        });
        return std::move(made);
    }

} // namespace

/** @brief Tests that a second row for one task type is refused as already registered. */
TEST(Small_InferenceRun, ASecondRowForOneTypeIsRefused)
{
    Apps::InferenceRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::InferenceEntry{"classification", &report_nothing}));

    SA_EXPECT_FAILURE(registry.add(Apps::InferenceEntry{"classification", &report_nothing}),
                      StatusCode::InvalidDescription, "'classification'", "already registered");
}

/** @brief Tests that a report runs the row of its task type. */
TEST_F(Medium_InferenceRun, TheReportRunsTheRowOfItsType)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<InferenceCase> made, make_case(backend()));
    Apps::InferenceRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::InferenceEntry{"classification", &report_failure}));

    SA_EXPECT_FAILURE(Apps::report_inference(registry, "classification", *made->prepared),
                      StatusCode::DeviceFailed, "the classification row ran");
}

/** @brief Tests that an unknown task type is refused, naming the model file and the known type. */
TEST_F(Medium_InferenceRun, AnUnknownTaskTypeIsRefusedNamingTheKnownOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<InferenceCase> made, make_case(backend()));
    Apps::InferenceRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::InferenceEntry{"classification", &report_nothing}));

    SA_EXPECT_FAILURE(Apps::report_inference(registry, "pose", *made->prepared),
                      StatusCode::InvalidDescription, "model.json", "'pose'",
                      "'classification'");
}
