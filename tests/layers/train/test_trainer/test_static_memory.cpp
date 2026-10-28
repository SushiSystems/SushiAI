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
 * @file test_static_memory.cpp
 * @brief Tests that the memory planner packs buffers without allocating or changing a number.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
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
    class Medium_StaticMemory : public SaTest::EngineTest {};

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
    [[nodiscard]] Result<Outcome> train(SushiBLAS::Engine& engine, bool packed,
                                        Graph::FusionOptions fusion = {})
    {
        MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES, DType::Float32,
                                              fusion));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        if (!packed)
            SA_TRY(pool.adopt(MemoryPlan{}));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_TRY(NN::initialize(engine, state));

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                      Optim::AdamW::make(engine, slots, adam));

        Train::TrainerOptions options;
        options.track_metrics = false;
        const auto objective = SaTest::classifier_objective(built);
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(engine, built, pool, objective, *optimizer,
                                           options));

        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(engine, data_options()));

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
}

/**
 * @brief Tests that twelve steps after the first leave the live allocation count unchanged.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_StaticMemory, ATrainingLoopAllocatesNothingAfterTheFirstStep)
{
    const auto live_allocations = [this]
    {
        return engine->runtime().advanced().live_allocation_count();
    };

    MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    ASSERT_TRUE(pool.has_plan());
    ASSERT_GT(pool.arenas().arena_count(), 0u) << "nothing was packed, so this proves nothing";

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, Optim::AdamWOptions{}));

    Train::TrainerOptions options;
    options.track_metrics = false;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Train::Trainer> trainer,
        Train::Trainer::make(*engine, built, pool, objective, *optimizer, options));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options()));

    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_EXPECT_OK(trainer->run_step(dataset_batch));
    engine->runtime().wait().value();

    const std::size_t baseline = live_allocations();
    const std::size_t bytes_baseline = pool.allocated_bytes();

    for (int step = 1; step <= 12; ++step)
    {
        ASSERT_TRUE(SaTest::status_is_ok(
            dataset->read(static_cast<std::size_t>(step) % K_BATCHES, dataset_batch)));
        SA_EXPECT_OK(trainer->run_step(dataset_batch));
        engine->runtime().wait().value();

        EXPECT_EQ(live_allocations(), baseline)
            << "step " << step << " changed the number of live buffers; a compiled step "
            << "must replay into the memory it already has";
        EXPECT_EQ(pool.allocated_bytes(), bytes_baseline);
    }
}

/**
 * @brief Tests that packing allocates fewer bytes than no plan on the unfused graph.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_StaticMemory, PackingShrinksTheFootprintOfTheSameModel)
{
    SushiBLAS::Engine unplanned_engine(SaTest::TestEnvironment::runtime());
    SA_ASSERT_OK_AND_ASSIGN(const Outcome unplanned, train(unplanned_engine, false, unfused()));

    SushiBLAS::Engine planned_engine(SaTest::TestEnvironment::runtime());
    SA_ASSERT_OK_AND_ASSIGN(const Outcome planned, train(planned_engine, true, unfused()));

    EXPECT_EQ(unplanned.arena_count, 0u);
    EXPECT_GT(planned.arena_count, 0u);

    // Both pools built the same tensors, so the unpacked figure is the same.
    EXPECT_EQ(planned.unpacked_bytes, unplanned.unpacked_bytes);
    EXPECT_EQ(unplanned.allocated_bytes, unplanned.unpacked_bytes);

    // The planned pool holds fewer bytes than the unplanned one.
    EXPECT_LT(planned.allocated_bytes, unplanned.allocated_bytes)
        << "planned " << planned.allocated_bytes << " bytes against an unplanned "
        << unplanned.allocated_bytes;
}

/**
 * @brief Tests that fusion removes intermediates and never allocates more than packing alone.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_StaticMemory, FusingTakesTheBuffersPackingUsedToOverlap)
{
    SushiBLAS::Engine fused_engine(SaTest::TestEnvironment::runtime());
    SA_ASSERT_OK_AND_ASSIGN(const Outcome fused, train(fused_engine, true));

    SushiBLAS::Engine planned_engine(SaTest::TestEnvironment::runtime());
    SA_ASSERT_OK_AND_ASSIGN(const Outcome planned, train(planned_engine, true, unfused()));

    SushiBLAS::Engine loose_engine(SaTest::TestEnvironment::runtime());
    SA_ASSERT_OK_AND_ASSIGN(const Outcome loose, train(loose_engine, false, unfused()));

    EXPECT_LT(fused.unpacked_bytes, planned.unpacked_bytes)
        << "fusion did not remove any intermediate at all";

    // Fusion on its own beats doing nothing, which is the claim worth pinning.
    EXPECT_LT(fused.allocated_bytes, loose.allocated_bytes)
        << "fused " << fused.allocated_bytes << " against an unplanned, unfused "
        << loose.allocated_bytes;

    // Never worse than packed-but-unfused, hence <=; see tests/layers/train/README.md.
    EXPECT_LE(fused.allocated_bytes, planned.allocated_bytes)
        << "fused " << fused.allocated_bytes << " against an unfused-but-packed "
        << planned.allocated_bytes;
}

/**
 * @brief Tests that packed and unpacked training agree byte for byte.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_StaticMemory, PackedAndUnpackedTrainingAgreeByteForByte)
{
    SushiBLAS::Engine unplanned_engine(SaTest::TestEnvironment::runtime());
    SA_ASSERT_OK_AND_ASSIGN(const Outcome unplanned, train(unplanned_engine, false));

    SushiBLAS::Engine planned_engine(SaTest::TestEnvironment::runtime());
    SA_ASSERT_OK_AND_ASSIGN(const Outcome planned, train(planned_engine, true));

    ASSERT_EQ(planned.names, unplanned.names);
    ASSERT_EQ(planned.losses.size(), unplanned.losses.size());

    for (std::size_t step = 0; step < planned.losses.size(); ++step)
        EXPECT_EQ(planned.losses[step], unplanned.losses[step])
            << "packing changed the loss at step " << step;

    for (std::size_t i = 0; i < planned.parameters.size(); ++i)
    {
        const std::size_t at = SaTest::first_difference(planned.parameters[i],
                                                        unplanned.parameters[i]);
        EXPECT_EQ(at, planned.parameters[i].size())
            << "packing changed parameter '" << planned.names[i] << "' from byte " << at;
    }
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
                            Graph::CompiledStep::compile(*engine, graph, pool));

    SA_EXPECT_OK(step->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(sum)), input);

    // Stand in for "a later value was given these bytes".
    SaTest::write_tensor(pool.tensor(zeros), std::vector<double>(12, 999.0));

    SA_EXPECT_OK(step->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(sum)), input)
        << "the ZERO buffer was not refilled, so its bytes cannot be reused by anything";
}
