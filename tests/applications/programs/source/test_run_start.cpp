/**************************************************************************/
/* test_run_start.cpp                                                     */
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
 * @file test_run_start.cpp
 * @brief Tests that a run starts from what its file states.
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
    class Medium_RunStart : public SaTest::EngineTest {};

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

    /** @brief Returns the run `sushiai train` would hand to the table for @p pieces. */
    Apps::PreparedRun prepare(Pieces& pieces, SushiAI::ExecutionBackend& backend)
    {
        return Apps::PreparedRun{
            backend,         pieces.spec,       *pieces.dataset, pieces.given,
            *pieces.objective, *pieces.built,  *pieces.state,   pieces.pool,
            *pieces.optimizer, {},             nullptr,
        };
    }

    /**
     * @brief Trains a run for K_EPOCHS epochs and saves it with its run state to @p file.
     * @return The optimizer's step count after the run, or the first failure.
     */
    Result<std::uint64_t> train_and_save(SushiAI::ExecutionBackend& backend,
                                         const std::filesystem::path& file)
    {
        SA_TRY_ASSIGN(const std::unique_ptr<Pieces> trained, make_pieces(backend));
        Apps::PreparedRun prepared = prepare(*trained, backend);
        Apps::RunRegistry registry;
        SA_TRY(Apps::register_runs(registry));

        testing::internal::CaptureStdout();
        const Result<Apps::RunSummary> summary =
            Apps::run_training(registry, "cross_entropy", prepared);
        (void)testing::internal::GetCapturedStdout();
        SA_TRY(summary.status());
        SA_TRY(IO::save_run_checkpoint(file, *trained->state, *trained->optimizer));
        return trained->optimizer->step_count();
    }

} // namespace

/** @brief Tests that a spec without files describes a fresh start. */
TEST(Small_RunStart, ASpecWithoutFilesDescribesAnInitialisedStart)
{
    const Config::TrainSpec spec;

    EXPECT_EQ(Apps::describe_start(spec), "initialised");
}

/** @brief Tests that a spec with weights describes the path of the weights. */
TEST(Small_RunStart, ASpecWithWeightsDescribesTheirPath)
{
    Config::TrainSpec spec;
    spec.weights = "weights.sushi";

    EXPECT_EQ(Apps::describe_start(spec), spec.weights.string());
}

/** @brief Tests that a spec with a resume file describes the resume. */
TEST(Small_RunStart, ASpecWithAResumeFileDescribesTheResume)
{
    Config::TrainSpec spec;
    spec.resume = "run.sushi";

    EXPECT_EQ(Apps::describe_start(spec), "resumed from " + spec.resume.string());
}

/** @brief Tests that a start without files leaves the optimizer at step 0. */
TEST_F(Medium_RunStart, AStartWithoutFilesLeavesTheOptimizerAtStepZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> fresh, make_pieces(backend()));

    SA_EXPECT_OK(Apps::start_state(fresh->spec, backend(), *fresh->state, *fresh->optimizer));

    EXPECT_EQ(fresh->optimizer->step_count(), 0u);
}

/** @brief Tests that a start from weights loads them and leaves the optimizer at step 0. */
TEST_F(Medium_RunStart, AStartFromWeightsLeavesTheOptimizerAtStepZero)
{
    const SaTest::ScratchFile file("sushiai_program_runs_weights.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t trained_steps,
                            train_and_save(backend(), file.path()));
    ASSERT_GT(trained_steps, 0u);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> fresh, make_pieces(backend()));
    fresh->spec.weights = file.path();

    SA_EXPECT_OK(Apps::start_state(fresh->spec, backend(), *fresh->state, *fresh->optimizer));

    EXPECT_EQ(fresh->optimizer->step_count(), 0u);
}

/** @brief Tests that a resumed start restores the step count of the saved run. */
TEST_F(Medium_RunStart, AResumedStartRestoresTheStepCountOfTheSavedRun)
{
    const SaTest::ScratchFile file("sushiai_program_runs_resume.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t trained_steps,
                            train_and_save(backend(), file.path()));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> fresh, make_pieces(backend()));
    fresh->spec.resume = file.path();

    SA_EXPECT_OK(Apps::start_state(fresh->spec, backend(), *fresh->state, *fresh->optimizer));

    EXPECT_EQ(fresh->optimizer->step_count(), trained_steps);
    EXPECT_EQ(fresh->optimizer->step_count(),
              static_cast<std::uint64_t>(K_EPOCHS * K_BATCHES));
}
