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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
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
        void prepare(SushiAI::ExecutionBackend& backend)
        {
            SA_ASSERT_OK_AND_ASSIGN(built,
                                    Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));
            SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(backend.engine(), model));
            pool = std::make_unique<Graph::TensorPool>(backend.engine());
            SA_ASSERT_OK_AND_ASSIGN(slots, Train::bind_parameters(backend, built, *state, *pool));
            SA_EXPECT_OK(NN::initialize(backend, *state));
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

    /**
     * @brief Records nothing and notes whether the optimizer had recorded before it, and how
     *        often each replay hook ran.
     */
    class OrderProbe final : public Graph::StepRecorder
    {
        public:
            /** @brief Watches @p optimizer, which outlives the probe. */
            explicit OrderProbe(const Optim::Optimizer& optimizer) noexcept
                : optimizer_(optimizer)
            {

            }

            OrderProbe(const OrderProbe&) = delete;
            OrderProbe& operator=(const OrderProbe&) = delete;
            OrderProbe(OrderProbe&&) = delete;
            OrderProbe& operator=(OrderProbe&&) = delete;
            ~OrderProbe() override = default;

            /** @brief Notes whether the optimizer is already in the step; records no task. */
            Result<std::size_t> record_into_step(const Graph::StepRecording&) override
            {
                after_optimizer = optimizer_.is_in_compiled_step();
                ++records;
                return std::size_t{0};
            }

            /** @brief Counts the replays about to run. */
            void before_replay() override
            {
                ++replays_begun;
            }

            /** @brief Counts the replays that finished. */
            void after_replay() override
            {
                ++replays_ended;
            }

            bool after_optimizer = false; /**< @brief Whether the optimizer recorded first. */
            int records = 0;              /**< @brief The record_into_step() calls. */
            int replays_begun = 0;        /**< @brief The before_replay() calls. */
            int replays_ended = 0;        /**< @brief The after_replay() calls. */

        private:
            /// @brief The optimizer whose recording the probe looks at.
            const Optim::Optimizer& optimizer_;
    };
} // namespace

/** @brief Tests that binding gives the loss its own tensor before any step is compiled. */
TEST_F(Medium_TrainingStep, Regression_BindingGivesTheLossATensorBeforeTheStepIsCompiled)
{
    ClassifierParts parts;
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    ASSERT_TRUE(parts.pool->is_bound(parts.built.spec.loss));
    EXPECT_NE(parts.pool->tensor(parts.built.spec.loss).data(), nullptr);
}

/** @brief Tests that an InPlan step refuses an optimizer whose replay would repeat its scalars. */
TEST_F(Medium_TrainingStep, InPlanRefusesAnOptimizerThatIsNotReplayInvariant)
{
    ClassifierParts parts;
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    Optim::AdamWOptions adam;
    adam.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend(), parts.slots, adam));
    ASSERT_FALSE(optimizer->is_replay_invariant());

    SA_EXPECT_FAILURE(TrainingStep::make(backend(), parts.built.graph, *parts.pool, *optimizer, {}),
                      StatusCode::InvalidDescription, "cannot be compiled into the step",
                      "PerStep");
}

/** @brief Tests that a PerStep step accepts such an optimizer and keeps it out of the plan. */
TEST_F(Medium_TrainingStep, PerStepKeepsTheOptimizerOutOfThePlan)
{
    ClassifierParts parts;
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    Optim::AdamWOptions adam;
    adam.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend(), parts.slots, adam));
    TrainingStepOptions options;
    options.optimizer_placement = OptimizerPlacement::PerStep;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TrainingStep> step,
        TrainingStep::make(backend(), parts.built.graph, *parts.pool, *optimizer, {}, options));

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
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), parts.slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TrainingStep> step,
        TrainingStep::make(backend(), parts.built.graph, *parts.pool, *optimizer, {}));

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
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), parts.slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TrainingStep> step,
                            TrainingStep::make(backend(), parts.built.graph, *parts.pool,
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
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), parts.slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TrainingStep> step,
                            TrainingStep::make(backend(), parts.built.graph, *parts.pool,
                                               *optimizer, {}, parts.host_stage_options()));

    SA_EXPECT_OK(step->restore_run_count(5));

    ASSERT_NE(step->lead(), nullptr);
    EXPECT_EQ(step->lead()->run_count(), 5u);
    EXPECT_EQ(step->step().run_count(), 5u);
}

/**
 * @brief Tests that before_update records ahead of the optimizer and after_update behind it,
 *        and that both follow every replay.
 */
TEST_F(Medium_TrainingStep, BeforeAndAfterUpdateRecordAroundTheOptimizer)
{
    ClassifierParts parts;
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), parts.slots, Optim::SGDOptions{}));
    OrderProbe before(*optimizer);
    OrderProbe after(*optimizer);
    TrainingStepOptions options;
    options.before_update = &before;
    options.after_update = &after;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TrainingStep> step,
                            TrainingStep::make(backend(), parts.built.graph, *parts.pool,
                                               *optimizer, {}, options));

    EXPECT_EQ(before.records, 1);
    EXPECT_EQ(after.records, 1);
    EXPECT_FALSE(before.after_optimizer);
    EXPECT_TRUE(after.after_optimizer);

    SA_EXPECT_OK(step->run());
    SA_EXPECT_OK(step->run());
    EXPECT_EQ(before.replays_begun, 2);
    EXPECT_EQ(before.replays_ended, 2);
    EXPECT_EQ(after.replays_begun, 2);
    EXPECT_EQ(after.replays_ended, 2);
}

/** @brief Tests that after_update is refused beside an optimizer stepped after the replay. */
TEST_F(Medium_TrainingStep, AfterUpdateIsRefusedBesideAPerStepOptimizer)
{
    ClassifierParts parts;
    parts.prepare(backend());
    ASSERT_TRUE(parts.ready);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), parts.slots, Optim::SGDOptions{}));
    OrderProbe after(*optimizer);
    TrainingStepOptions options;
    options.optimizer_placement = OptimizerPlacement::PerStep;
    options.after_update = &after;

    SA_EXPECT_FAILURE(TrainingStep::make(backend(), parts.built.graph, *parts.pool, *optimizer,
                                         {}, options),
                      StatusCode::InvalidDescription, "after_update", "InPlan");
    EXPECT_EQ(after.records, 0);
}
