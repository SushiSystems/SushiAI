/**************************************************************************/
/* test_epoch_loop.cpp                                                    */
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
 * @file test_epoch_loop.cpp
 * @brief Tests run_epochs, the loop the trainer run and the cluster run share.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

#include "epoch_loop.hpp"
#include "training_run.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    class Medium_EpochLoop : public SaTest::EngineTest {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    /** @brief Holds the batches of one epoch. */
    constexpr std::size_t K_BATCHES = 3;

    /**
     * @struct Pieces
     * @brief Holds what a PreparedRun names, over one engine; no case trains it.
     */
    struct Pieces
    {
        /** @brief Builds the pool on @p engine; make_pieces() fills the rest. */
        explicit Pieces(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief The model the graph is traced from. */
        MLP model = MLP(NN::Linear{4, 6}, NN::ReLU{}, NN::Linear{6, 2});

        /** @brief The training graph of the model. */
        std::optional<Train::TrainingGraph> built;

        /** @brief The parameters of the model. */
        std::optional<NN::ModelState> state;

        /** @brief The pool the graph's tensors live in. */
        Graph::TensorPool pool;

        /** @brief The objective the graph was traced with. */
        std::optional<Train::CrossEntropyObjective> objective;

        /** @brief The optimizer, whose rate the schedule sets. */
        std::unique_ptr<Optim::Optimizer> optimizer;

        /** @brief The batches of one epoch. */
        std::unique_ptr<Data::SyntheticClassification> dataset;

        /** @brief The training file; a case sets its epochs. */
        Config::TrainSpec spec;

        /** @brief The context no case reads. */
        Train::ObjectiveContext given;

        /** @brief The schedule a case may hand the run. */
        Optim::RateSchedule schedule;
    };

    /** @brief Builds the pieces of a run of @p epochs epochs of K_BATCHES batches. */
    Result<std::unique_ptr<Pieces>> make_pieces(SushiBLAS::Engine& engine, std::int64_t epochs)
    {
        std::unique_ptr<Pieces> pieces = std::make_unique<Pieces>(engine);

        SA_TRY_ASSIGN(Train::TrainingGraph built, Train::build_classifier(pieces->model, 2, 4, 2));
        pieces->built.emplace(std::move(built));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, pieces->model));
        pieces->state.emplace(std::move(state));
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(*pieces->built, *pieces->state, pieces->pool));

        Optim::SGDOptions options;
        options.learning_rate = 0.1;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                      Optim::SGD::make(engine, std::move(slots), options));
        pieces->optimizer = std::move(optimizer);
        pieces->objective.emplace(SaTest::classifier_objective(*pieces->built));

        Data::SyntheticOptions data;
        data.batch_size = 2;
        data.batches = static_cast<std::int64_t>(K_BATCHES);
        data.features = 4;
        data.classes = 2;
        SA_TRY_ASSIGN(pieces->dataset, Data::SyntheticClassification::make(engine, data));

        Optim::StepSchedule schedule;
        schedule.base_rate = 0.5;
        pieces->schedule = schedule;
        pieces->spec.epochs = epochs;
        return std::move(pieces);
    }

    /** @brief Returns the run over @p pieces, starting at @p start. */
    Apps::PreparedRun prepare(Pieces& pieces, SushiBLAS::Engine& engine,
                              const Optim::RateSchedule* schedule, IO::RunProgress start)
    {
        return Apps::PreparedRun{
            engine,         pieces.spec,       *pieces.dataset, pieces.given,
            *pieces.objective, *pieces.built,  *pieces.state,   pieces.pool,
            *pieces.optimizer, {},             schedule,        start,
        };
    }
} // namespace

/** @brief Tests the epochs a progress line is printed for. */
TEST(Small_EpochLoop, ReportsTheFirstEveryFifthAndTheLastEpoch)
{
    EXPECT_TRUE(Apps::is_reported_epoch(1, 12));
    EXPECT_FALSE(Apps::is_reported_epoch(2, 12));
    EXPECT_TRUE(Apps::is_reported_epoch(5, 12));
    EXPECT_TRUE(Apps::is_reported_epoch(10, 12));
    EXPECT_FALSE(Apps::is_reported_epoch(11, 12));
    EXPECT_TRUE(Apps::is_reported_epoch(12, 12));
}

/** @brief Tests that a resumed run starts at its batch and epoch, then reads every batch. */
TEST_F(Medium_EpochLoop, StartsAtTheSavedBatchAndEntersEachEpoch)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> pieces, make_pieces(*engine, 2));
    IO::RunProgress start;
    start.epoch = 4;
    start.batch = 2;
    const Apps::PreparedRun run = prepare(*pieces, *engine, &pieces->schedule, start);

    std::vector<std::int64_t> entered;
    std::vector<std::size_t> stepped;
    std::vector<double> rates;
    std::vector<std::int64_t> reported;
    Apps::EpochSteps steps;
    steps.enter_epoch = [&entered](std::int64_t epoch_index)
    {
        entered.push_back(epoch_index);
    };
    steps.run_step = [&](std::size_t index)
    {
        stepped.push_back(index);
        rates.push_back(pieces->optimizer->learning_rate());
        return Status();
    };
    steps.report = [&reported](std::int64_t epoch, std::int64_t)
    {
        reported.push_back(epoch);
    };
    SA_EXPECT_OK(Apps::run_epochs(run, steps));

    EXPECT_EQ(entered, (std::vector<std::int64_t>{4, 5}));
    EXPECT_EQ(stepped, (std::vector<std::size_t>{2, 0, 1, 2}));
    EXPECT_EQ(rates, (std::vector<double>(4, 0.5)));
    EXPECT_EQ(reported, (std::vector<std::int64_t>{1, 2}));
}

/** @brief Tests that the first failed update ends the loop with its status. */
TEST_F(Medium_EpochLoop, StopsAtTheFirstFailedUpdate)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Pieces> pieces, make_pieces(*engine, 2));
    const Apps::PreparedRun run = prepare(*pieces, *engine, nullptr, IO::RunProgress{});

    std::size_t updates = 0;
    Apps::EpochSteps steps;
    steps.run_step = [&updates](std::size_t index) -> Status
    {
        ++updates;
        SA_FAIL_IF(index == 1, StatusCode::DeviceFailed, "update {} failed", index);
        return Status();
    };
    steps.report = [](std::int64_t, std::int64_t)
    {

    };
    SA_EXPECT_FAILURE(Apps::run_epochs(run, steps), StatusCode::DeviceFailed, "update 1");
    EXPECT_EQ(updates, 2u);
}
