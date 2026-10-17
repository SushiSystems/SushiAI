/**************************************************************************/
/* test_training_step.cpp                                                 */
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
 * @file test_training_step.cpp
 * @brief Tests how TrainingStep places the optimizer, splits at a host stage and counts runs.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Train::OptimizerPlacement;
using Train::TrainingStep;
using Train::TrainingStepOptions;

namespace
{
    /** @brief Names the classifier every case compiles. */
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    /** @brief Samples per step. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Input features of the classifier. */
    constexpr std::int64_t K_FEATURES = 6;

    /** @brief Hidden units of the classifier. */
    constexpr std::int64_t K_HIDDEN = 8;

    /** @brief Classes the classifier ends in. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Provides the engine each case compiles its step on. */
    class Medium_TrainingStep : public SaTest::EngineTest
    {

    };

    /** @brief Holds the traced classifier, its state, its pool and its parameter slots. */
    struct ClassifierParts
    {
        /** @brief Traces, binds and initialises the classifier; sets @c ready on success. */
        void prepare(SushiBLAS::Engine& engine)
        {
            SA_ASSERT_OK_AND_ASSIGN(built,
                                    Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));
            SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(engine, model));
            pool = std::make_unique<Graph::TensorPool>(engine);
            SA_ASSERT_OK_AND_ASSIGN(slots, Train::bind_parameters(built, *state, *pool));
            SA_EXPECT_OK(NN::initialize(engine, *state));
            ready = true;
        }

        /** @brief Returns the options of a step whose host reads the logits. */
        TrainingStepOptions host_stage_options() const
        {
            TrainingStepOptions options;
            options.host_stage.reads = {built.spec.logits};
            return options;
        }

        MLP model{NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{},
                  NN::Linear{K_HIDDEN, K_CLASSES}}; /**< @brief Holds the classifier. */
        Train::TrainingGraph built;                 /**< @brief Holds its training graph. */
        std::optional<NN::ModelState> state;        /**< @brief Holds its parameters. */
        std::unique_ptr<Graph::TensorPool> pool;    /**< @brief Holds the step's values. */
        std::vector<Optim::ParameterSlot> slots;    /**< @brief Pairs parameters and gradients. */
        bool ready = false;                         /**< @brief Holds whether prepare() ended. */
    };
} // namespace

/** @brief Tests that an InPlan step refuses an optimizer whose replay would repeat its scalars. */
TEST_F(Medium_TrainingStep, InPlanRefusesAnOptimizerThatIsNotReplayInvariant)
{
    ClassifierParts parts;
    parts.prepare(*engine);
    ASSERT_TRUE(parts.ready);

    Optim::AdamWOptions adam;
    adam.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, parts.slots, adam));
    ASSERT_FALSE(optimizer->is_replay_invariant());

    SA_EXPECT_FAILURE(TrainingStep::make(*engine, parts.built.graph, *parts.pool, *optimizer, {}),
                      StatusCode::InvalidDescription, "cannot be compiled into the step",
                      "PerStep");
}

/** @brief Tests that a PerStep step accepts such an optimizer and keeps it out of the plan. */
TEST_F(Medium_TrainingStep, PerStepKeepsTheOptimizerOutOfThePlan)
{
    ClassifierParts parts;
    parts.prepare(*engine);
    ASSERT_TRUE(parts.ready);

    Optim::AdamWOptions adam;
    adam.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, parts.slots, adam));
    TrainingStepOptions options;
    options.optimizer_placement = OptimizerPlacement::PerStep;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TrainingStep> step,
        TrainingStep::make(*engine, parts.built.graph, *parts.pool, *optimizer, {}, options));

    EXPECT_FALSE(step->optimizer_is_in_plan());
    EXPECT_EQ(&step->optimizer(), optimizer.get());
    SA_EXPECT_OK(step->run());
    SA_EXPECT_OK(step->run());
    EXPECT_EQ(step->step().run_count(), 2u);
}

/** @brief Tests that the default step records the optimizer and compiles one part. */
TEST_F(Medium_TrainingStep, TheDefaultStepHoldsTheOptimizerInOnePart)
{
    ClassifierParts parts;
    parts.prepare(*engine);
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, parts.slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TrainingStep> step,
        TrainingStep::make(*engine, parts.built.graph, *parts.pool, *optimizer, {}));

    EXPECT_TRUE(step->optimizer_is_in_plan());
    EXPECT_FALSE(step->has_host_stage());
    EXPECT_EQ(step->lead(), nullptr);
    SA_EXPECT_OK(step->run());
    EXPECT_EQ(step->step().run_count(), 1u);
}

/** @brief Tests that a host stage splits the step and the host work runs once between parts. */
TEST_F(Medium_TrainingStep, AHostStageRunsTheHostWorkBetweenTheTwoParts)
{
    ClassifierParts parts;
    parts.prepare(*engine);
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, parts.slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TrainingStep> step,
                            TrainingStep::make(*engine, parts.built.graph, *parts.pool,
                                               *optimizer, {}, parts.host_stage_options()));
    ASSERT_TRUE(step->has_host_stage());
    ASSERT_NE(step->lead(), nullptr);

    int calls = 0;
    std::size_t lead_runs_seen = 0;
    std::size_t rest_runs_seen = 0;
    SA_EXPECT_OK(step->run([&]()
    {
        ++calls;
        lead_runs_seen = step->lead()->run_count();
        rest_runs_seen = step->step().run_count();
        return Status();
    }));

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(lead_runs_seen, 1u);
    EXPECT_EQ(rest_runs_seen, 0u);
    EXPECT_EQ(step->step().run_count(), 1u);
}

/** @brief Tests that restoring a run count sets it on both parts of a split step. */
TEST_F(Medium_TrainingStep, RestoringTheRunCountSetsEveryPart)
{
    ClassifierParts parts;
    parts.prepare(*engine);
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, parts.slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TrainingStep> step,
                            TrainingStep::make(*engine, parts.built.graph, *parts.pool,
                                               *optimizer, {}, parts.host_stage_options()));

    SA_EXPECT_OK(step->restore_run_count(5));

    ASSERT_NE(step->lead(), nullptr);
    EXPECT_EQ(step->lead()->run_count(), 5u);
    EXPECT_EQ(step->step().run_count(), 5u);
}
