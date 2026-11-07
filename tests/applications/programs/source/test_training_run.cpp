/**************************************************************************/
/* test_training_run.cpp                                                  */
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
 * @file test_training_run.cpp
 * @brief Tests the run table of training_run: the row a task picks, the fallback, and a
 *        classification run.
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
    class Medium_TrainingRun : public SaTest::EngineTest {};

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

    /** @brief Returns the run `sushiai train` would hand to the table for @p pieces. */
    Apps::PreparedRun prepare(Pieces& pieces)
    {
        return Apps::PreparedRun{
            pieces.spec,       *pieces.dataset, pieces.given,
            *pieces.objective, *pieces.built,  *pieces.state,   pieces.pool,
            *pieces.optimizer, {},             nullptr,
        };
    }

    /** @brief Stands for the run of a row; reports K_ROW_COUNT and trains nothing. */
    Result<Apps::RunSummary> run_of_a_row(SushiAI::ExecutionBackend&, Apps::PreparedRun&)
    {
        Apps::RunSummary summary;
        summary.compile_count = K_ROW_COUNT;
        return summary;
    }

    /** @brief Stands for the fallback run; reports K_FALLBACK_COUNT and trains nothing. */
    Result<Apps::RunSummary> run_of_the_fallback(SushiAI::ExecutionBackend&, Apps::PreparedRun&)
    {
        Apps::RunSummary summary;
        summary.compile_count = K_FALLBACK_COUNT;
        return summary;
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

    /**
     * @struct RunCase
     * @brief Holds the pieces of a run and the run `sushiai train` would hand to the table.
     */
    struct RunCase
    {
        /** @brief The pieces the run is made of. */
        std::unique_ptr<Pieces> pieces;

        /** @brief The run over the pieces. */
        std::optional<Apps::PreparedRun> prepared;
    };

    /** @brief Builds the run case on @p engine, or returns the first failure of its pieces. */
    Result<std::unique_ptr<RunCase>> make_case(SushiAI::ExecutionBackend& backend)
    {
        std::unique_ptr<RunCase> made = std::make_unique<RunCase>();
        SA_TRY_ASSIGN(made->pieces, make_pieces(backend));
        made->prepared.emplace(prepare(*made->pieces));
        return std::move(made);
    }

    /**
     * @struct ClassificationRun
     * @brief Holds the pieces, the summary and the printed text of a run through the fallback.
     */
    struct ClassificationRun
    {
        /** @brief The pieces the run trained. */
        std::unique_ptr<Pieces> pieces;

        /** @brief What the fallback run returned. */
        Result<Apps::RunSummary> summary;

        /** @brief What the run wrote to standard output. */
        std::string printed;
    };

    /** @brief Trains a classification run through the fallback of the run table on @p engine. */
    ClassificationRun run_classification(SushiAI::ExecutionBackend& backend)
    {
        std::unique_ptr<Pieces> pieces = make_pieces(backend).value();
        Apps::PreparedRun prepared = prepare(*pieces);
        Apps::RunRegistry registry;
        SA_EXPECT_OK(Apps::register_runs(registry));

        testing::internal::CaptureStdout();
        Result<Apps::RunSummary> summary =
            Apps::run_training(registry, "cross_entropy", backend, prepared);
        std::string printed = testing::internal::GetCapturedStdout();
        return ClassificationRun{std::move(pieces), std::move(summary), std::move(printed)};
    }

} // namespace

/** @brief Tests that a second row for one task type is refused as already registered. */
TEST(Small_TrainingRun, ASecondRowForOneTypeIsRefused)
{
    Apps::RunRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::RunEntry{"anchor", &run_of_a_row}));

    SA_EXPECT_FAILURE(registry.add(Apps::RunEntry{"anchor", &run_of_the_fallback}),
                      StatusCode::InvalidDescription, "'anchor'", "already registered");
}

/** @brief Tests that a run of a task type with a row runs that row. */
TEST_F(Medium_TrainingRun, ARunRunsTheRowOfItsType)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RunCase> made, make_case(backend()));
    Apps::RunRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::RunEntry{"anchor", &run_of_a_row}));
    SA_EXPECT_OK(registry.set_fallback(&run_of_the_fallback));

    SA_ASSERT_OK_AND_ASSIGN(const Apps::RunSummary of_row,
                            Apps::run_training(registry, "anchor", backend(), *made->prepared));

    EXPECT_EQ(of_row.compile_count, K_ROW_COUNT);
}

/** @brief Tests that a run of a task type with no row runs the fallback. */
TEST_F(Medium_TrainingRun, ARunOfAnotherTypeRunsTheFallback)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RunCase> made, make_case(backend()));
    Apps::RunRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::RunEntry{"anchor", &run_of_a_row}));
    SA_EXPECT_OK(registry.set_fallback(&run_of_the_fallback));

    SA_ASSERT_OK_AND_ASSIGN(const Apps::RunSummary of_fallback,
                            Apps::run_training(registry, "cross_entropy", backend(), *made->prepared));

    EXPECT_EQ(of_fallback.compile_count, K_FALLBACK_COUNT);
}

/** @brief Tests that an unknown type without a fallback is refused, naming the known type. */
TEST_F(Medium_TrainingRun, AnUnknownTypeWithoutAFallbackIsRefusedNamingTheKnownOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RunCase> made, make_case(backend()));
    Apps::RunRegistry registry;
    SA_EXPECT_OK(registry.add(Apps::RunEntry{"anchor", &run_of_a_row}));

    SA_EXPECT_FAILURE(Apps::run_training(registry, "cross_entropy", backend(), *made->prepared),
                      StatusCode::InvalidDescription, "'cross_entropy'", "'anchor'");
}

/** @brief Tests that the fallback run compiles once and takes every step of its epochs. */
TEST_F(Medium_TrainingRun, AClassificationRunCompilesOnceAndTakesEveryStep)
{
    const ClassificationRun run = run_classification(backend());

    SA_EXPECT_OK(run.summary);
    ASSERT_TRUE(run.summary.has_value());
    EXPECT_EQ(run.summary.value().compile_count, 1u);
    EXPECT_EQ(run.pieces->optimizer->step_count(),
              static_cast<std::uint64_t>(K_EPOCHS * K_BATCHES));
}

/** @brief Tests that the fallback run prints a loss that falls from its first epoch to its last. */
TEST_F(Medium_TrainingRun, AClassificationRunPrintsAFallingLoss)
{
    const ClassificationRun run = run_classification(backend());

    // The run prints the loss of the last batch of the first and of the last epoch.
    const std::vector<double> losses = losses_of(run.printed);
    ASSERT_EQ(losses.size(), static_cast<std::size_t>(K_EPOCHS)) << run.printed;
    EXPECT_LT(losses.back(), losses.front()) << run.printed;
}
