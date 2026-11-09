/**************************************************************************/
/* test_static_memory.cpp                                                 */
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
 * @file test_static_memory.cpp
 * @brief Tests that the memory planner packs buffers without allocating or changing a number.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::MemoryPlan;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 16;
    constexpr int64_t K_HIDDEN = 64;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 3;

    /** @brief The synthetic problem every run below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        options.seed = 0xA11'0C8'2026ull;
        return options;
    }

    /**
     * @struct Outcome
     * @brief One trained model, plus what its memory cost.
     */
    struct Outcome
    {
        std::vector<std::vector<unsigned char>> parameters;
        std::vector<std::string> names;
        std::vector<double> losses;
        std::size_t allocated_bytes = 0;
        std::size_t unpacked_bytes = 0;
        std::size_t arena_count = 0;
    };

    /** @brief Returns fusion options with fusion off, the graph the planner's saving is about. */
    Graph::FusionOptions unfused()
    {
        Graph::FusionOptions options;
        options.enabled = false;
        return options;
    }

    /**
     * @brief Trains the same MLP, with the planner on or off.
     * @return The trained model and its memory figures, or the first failure of a library call.
     * @see tests/layers/train/README.md
     */
    [[nodiscard]] Result<Outcome> train(SushiAI::ExecutionBackend& backend, bool packed,
                                        Graph::FusionOptions fusion = {})
    {
        MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES, DType::Float32,
                                              fusion));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), model));
        TensorPool pool(backend.engine());
        if (!packed)
            SA_TRY(pool.adopt(MemoryPlan{}));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, built, state, pool));
        SA_TRY(NN::initialize(backend, state));

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                      Optim::AdamW::make(backend, slots, adam));

        Train::TrainerOptions options;
        options.track_metrics = false;
        const auto objective = SaTest::classifier_objective(built);
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(backend, built, pool, objective, *optimizer,
                                           options));

        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(backend, data_options()));

        Outcome run;
        for (int epoch = 0; epoch < 3; ++epoch)
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                SA_TRY(dataset->read(batch, dataset_batch));
                SA_TRY_ASSIGN(const Train::StepMetrics metrics,
                              trainer->run_step(dataset_batch));
                run.losses.push_back(metrics.loss);
            }

        for (const Optim::ParameterSlot& slot : slots)
        {
            run.names.push_back(slot.name);
            run.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }

        run.allocated_bytes = pool.allocated_bytes();
        run.unpacked_bytes = pool.unpacked_bytes();
        run.arena_count = pool.has_plan() ? pool.arenas().arena_count() : 0;
        return run;
    }

    /** @brief Trains the model on a fresh Engine and returns what it cost. */
    void train_on_fresh_engine(bool packed, Graph::FusionOptions fusion, Outcome& outcome)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
        SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);
        SA_ASSERT_OK_AND_ASSIGN(outcome, train(backend, packed, fusion));
    }

    /** @brief Provides the Engine the packed ZERO buffer case records on. */
    class Medium_StaticMemory : public SaTest::EngineTest {};

    /** @brief Replays a trained step over one batch and expects the allocations to hold. */
    class Medium_StaticMemoryReplay : public ::testing::TestWithParam<std::size_t>
    {
        protected:
            /** @brief Builds the trainer, takes the first step and records the baseline. */
            void SetUp() override
            {
                engine = std::make_unique<SushiBLAS::Engine>(SaTest::TestEnvironment::runtime());
                backend = std::make_unique<RuntimeExecutionBackend>(
                    SaTest::TestEnvironment::runtime(), *engine);

                SA_ASSERT_OK_AND_ASSIGN(
                    built, Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));
                SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(*engine, model));
                pool = std::make_unique<TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(slots,
                                        Train::bind_parameters(*backend, built, *state, *pool));
                SA_EXPECT_OK(NN::initialize(*backend, *state));

                ASSERT_TRUE(pool->has_plan());
                ASSERT_GT(pool->arenas().arena_count(), 0u)
                    << "nothing was packed, so this proves nothing";

                SA_ASSERT_OK_AND_ASSIGN(
                    optimizer, Optim::AdamW::make(*backend, slots, Optim::AdamWOptions{}));

                Train::TrainerOptions options;
                options.track_metrics = false;
                objective = std::make_unique<Train::CrossEntropyObjective>(
                    SaTest::classifier_objective(built));
                SA_ASSERT_OK_AND_ASSIGN(
                    trainer, Train::Trainer::make(*backend, built, *pool, *objective, *optimizer,
                                                  options));
                SA_ASSERT_OK_AND_ASSIGN(
                    dataset, Data::SyntheticClassification::make(*backend, data_options()));

                step_over(0);
                baseline = live_allocations();
                bytes_baseline = pool->allocated_bytes();
            }

            /** @brief Returns the live allocations of the runtime. */
            static std::size_t live_allocations()
            {
                return SaTest::TestEnvironment::runtime().advanced().live_allocation_count();
            }

            /** @brief Runs one step over batch @p index and waits for the device. */
            void step_over(std::size_t index)
            {
                Data::Batch batch;
                ASSERT_TRUE(SaTest::status_is_ok(dataset->read(index, batch)));
                SA_EXPECT_OK(trainer->run_step(batch));
                SaTest::TestEnvironment::runtime().wait().value();
            }

            /** @brief Replays batch @p index and expects the allocations to stay as they were. */
            void expect_replay_allocates_nothing(std::size_t index)
            {
                step_over(index);

                // A compiled step must replay into the memory it already has.
                EXPECT_EQ(live_allocations(), baseline)
                    << "a replay changed the number of live buffers";
                EXPECT_EQ(pool->allocated_bytes(), bytes_baseline);
            }

            MLP model{NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES}};
            std::unique_ptr<SushiBLAS::Engine> engine;                  /**< @brief Owns the device state. */
            std::unique_ptr<RuntimeExecutionBackend> backend;           /**< @brief Runs the steps. */
            Train::TrainingGraph built;                                 /**< @brief The graph. */
            std::optional<NN::ModelState> state;                        /**< @brief The parameters. */
            std::unique_ptr<TensorPool> pool;                           /**< @brief The values. */
            std::vector<Optim::ParameterSlot> slots;                    /**< @brief The slots. */
            std::unique_ptr<Optim::AdamW> optimizer;                    /**< @brief The optimizer. */
            std::unique_ptr<Train::CrossEntropyObjective> objective;    /**< @brief The objective. */
            std::unique_ptr<Train::Trainer> trainer;                    /**< @brief The trainer. */
            std::unique_ptr<Data::SyntheticClassification> dataset;     /**< @brief The data. */
            std::size_t baseline = 0;                                   /**< @brief Buffers after step one. */
            std::size_t bytes_baseline = 0;                             /**< @brief Pool bytes after step one. */
    };

    /** @brief Holds a packed and an unpacked run of the fused model. */
    class Medium_StaticMemoryAgreement : public ::testing::Test
    {
        protected:
            /** @brief Trains the model with the planner on and off. */
            void SetUp() override
            {
                ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(false, {}, unplanned));
                ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(true, {}, planned));
            }

            Outcome unplanned;   /**< @brief Holds the run with no plan. */
            Outcome planned;     /**< @brief Holds the run with the plan. */
    };
}

/**
 * @brief Tests that four replays of one batch after the first step leave the allocations unchanged.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_StaticMemoryReplay, ATrainingLoopAllocatesNothingAfterTheFirstStep)
{
    expect_replay_allocates_nothing(GetParam());
    expect_replay_allocates_nothing(GetParam());
    expect_replay_allocates_nothing(GetParam());
    expect_replay_allocates_nothing(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Batches, Medium_StaticMemoryReplay, ::testing::Values(0u, 1u, 2u));

/** @brief Tests that fusion removes intermediates a packed unfused graph still holds. */
TEST_F(Medium_StaticMemory, FusionRemovesIntermediates)
{
    Outcome fused;
    Outcome planned;
    ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(true, {}, fused));
    ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(true, unfused(), planned));

    EXPECT_LT(fused.unpacked_bytes, planned.unpacked_bytes)
        << "fusion did not remove any intermediate at all";
}

/** @brief Tests that the fused graph allocates fewer bytes than an unplanned, unfused one. */
TEST_F(Medium_StaticMemory, FusionAloneBeatsDoingNothing)
{
    Outcome fused;
    Outcome loose;
    ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(true, {}, fused));
    ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(false, unfused(), loose));

    EXPECT_LT(fused.allocated_bytes, loose.allocated_bytes)
        << "fused " << fused.allocated_bytes << " against an unplanned, unfused "
        << loose.allocated_bytes;
}

/** @brief Tests that the fused graph allocates no more than the packed unfused one. */
TEST_F(Medium_StaticMemory, FusionIsNeverWorseThanPackingAlone)
{
    Outcome fused;
    Outcome planned;
    ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(true, {}, fused));
    ASSERT_NO_FATAL_FAILURE(train_on_fresh_engine(true, unfused(), planned));

    // Never worse than packed-but-unfused, hence <=; see tests/layers/train/README.md.
    EXPECT_LE(fused.allocated_bytes, planned.allocated_bytes)
        << "fused " << fused.allocated_bytes << " against an unfused-but-packed "
        << planned.allocated_bytes;
}

/** @brief Tests that packing changes no loss. */
TEST_F(Medium_StaticMemoryAgreement, PackingChangesNoLoss)
{
    ASSERT_EQ(planned.losses.size(), unplanned.losses.size());
    EXPECT_EQ(planned.losses, unplanned.losses) << "packing changed a loss";
}

/** @brief Tests that packing changes no parameter byte. */
TEST_F(Medium_StaticMemoryAgreement, PackingChangesNoParameterByte)
{
    ASSERT_EQ(planned.names, unplanned.names);
    EXPECT_EQ(planned.parameters, unplanned.parameters) << "packing changed a parameter";
}

/**
 * @brief Tests that a packed ZERO buffer corrupted between replays is refilled.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_StaticMemory, APackedZeroBufferIsRefilledOnEveryReplay)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID zeros = builder.zeros(Shape{4, 3});
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sum, builder.add(zeros, x));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.adopt(MemoryPlan::build(graph).value()));
    SA_EXPECT_OK(pool.materialise_leaves(graph));

    const std::vector<double> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    SaTest::write_tensor(pool.tensor(x), input);

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Graph::CompiledStep> step,
                            Graph::CompiledStep::compile(backend(), graph, pool));

    SA_EXPECT_OK(step->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(sum)), input);

    // Stand in for "a later value was given these bytes".
    SaTest::write_tensor(pool.tensor(zeros), std::vector<double>(12, 999.0));

    SA_EXPECT_OK(step->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(sum)), input)
        << "the ZERO buffer was not refilled, so its bytes cannot be reused by anything";
}
