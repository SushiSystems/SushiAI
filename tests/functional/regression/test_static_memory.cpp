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
 * @see tests/functional/regression/README.md
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::MemoryPlan;
using Graph::TensorPool;
using Graph::ValueId;

namespace
{
    class Regression_StaticMemory : public SaTest::EngineTest {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t BATCH = 8;
    constexpr int64_t K_FEATURES = 16;
    constexpr int64_t K_HIDDEN = 64;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t K_BATCHES = 3;

    /** @brief The synthetic problem every run below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = CLASSES;
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

    /**
     * @brief Trains the same MLP, with the planner on or off.
     * @see tests/functional/regression/README.md
     */
    Outcome train(SushiBLAS::Engine& engine, bool packed, Graph::FusionOptions fusion = {})
    {
        Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, CLASSES});
        Train::TrainingGraph built = Train::build_classifier(
            model, BATCH, K_FEATURES, CLASSES, DType::Float32, fusion);

        TensorPool pool(engine);
        if (!packed) pool.adopt(MemoryPlan{});

        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        Optim::AdamW optimizer(engine, slots, adam);

        Train::TrainerOptions options;
        options.track_accuracy = false;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        const Data::SyntheticClassification dataset(engine, data_options());

        Outcome run;
        for (int epoch = 0; epoch < 3; ++epoch)
            for (std::size_t batch = 0; batch < dataset.batch_count(); ++batch)
                run.losses.push_back(trainer.run_step(dataset.batch(batch)).loss);

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
 * @brief Tests that twelve steps after the first leave the context's live allocation count unchanged.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_StaticMemory, ATrainingLoopAllocatesNothingAfterTheFirstStep)
{
    const auto live_allocations = [this]
    {
        return static_cast<std::int32_t>(engine->get_context().live_usm_count());
    };

    Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, CLASSES});
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, K_FEATURES, CLASSES);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    ASSERT_TRUE(pool.has_plan());
    ASSERT_GT(pool.arenas().arena_count(), 0u) << "nothing was packed, so this proves nothing";

    Optim::AdamW optimizer(*engine, slots, Optim::AdamWOptions{});

    Train::TrainerOptions options;
    options.track_accuracy = false;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    const Data::SyntheticClassification dataset(*engine, data_options());

    trainer.run_step(dataset.batch(0));
    engine->get_context().wait_all();

    const std::int32_t baseline = live_allocations();
    const std::size_t bytes_baseline = pool.allocated_bytes();

    for (int step = 1; step <= 12; ++step)
    {
        trainer.run_step(dataset.batch(static_cast<std::size_t>(step) % K_BATCHES));
        engine->get_context().wait_all();

        EXPECT_EQ(live_allocations(), baseline)
            << "step " << step << " allocated " << (live_allocations() - baseline)
            << " buffer(s); a compiled step must replay into the memory it already has";
        EXPECT_EQ(pool.allocated_bytes(), bytes_baseline);
    }
}

/** @brief The unfused graph - the one the planner's saving is a claim about. */
Graph::FusionOptions unfused()
{
    Graph::FusionOptions options;
    options.enabled = false;
    return options;
}

/**
 * @brief Tests that packing allocates fewer bytes than no plan on the unfused graph.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_StaticMemory, PackingShrinksTheFootprintOfTheSameModel)
{
    SushiBLAS::Engine unplanned_engine(SaTest::TestEnvironment::context());
    const Outcome unplanned = train(unplanned_engine, false, unfused());

    SushiBLAS::Engine planned_engine(SaTest::TestEnvironment::context());
    const Outcome planned = train(planned_engine, true, unfused());

    EXPECT_EQ(unplanned.arena_count, 0u);
    EXPECT_GT(planned.arena_count, 0u);

    // Both pools built the same tensors, so the counterfactual is identical...
    EXPECT_EQ(planned.unpacked_bytes, unplanned.unpacked_bytes);
    EXPECT_EQ(unplanned.allocated_bytes, unplanned.unpacked_bytes);

    // ...and the planned one actually holds less of it.
    EXPECT_LT(planned.allocated_bytes, unplanned.allocated_bytes)
        << "planned " << planned.allocated_bytes << " bytes against an unplanned "
        << unplanned.allocated_bytes;
}

/**
 * @brief Tests that fusion removes intermediates and never allocates more than packing alone.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_StaticMemory, FusingTakesTheBuffersPackingUsedToOverlap)
{
    SushiBLAS::Engine fused_engine(SaTest::TestEnvironment::context());
    const Outcome fused = train(fused_engine, true);

    SushiBLAS::Engine planned_engine(SaTest::TestEnvironment::context());
    const Outcome planned = train(planned_engine, true, unfused());

    SushiBLAS::Engine loose_engine(SaTest::TestEnvironment::context());
    const Outcome loose = train(loose_engine, false, unfused());

    EXPECT_LT(fused.unpacked_bytes, planned.unpacked_bytes)
        << "fusion did not remove any intermediate at all";

    // Fusion on its own beats doing nothing, which is the claim worth pinning.
    EXPECT_LT(fused.allocated_bytes, loose.allocated_bytes)
        << "fused " << fused.allocated_bytes << " against an unplanned, unfused "
        << loose.allocated_bytes;

    // Never worse than packed-but-unfused, hence <=; see tests/functional/regression/README.md.
    EXPECT_LE(fused.allocated_bytes, planned.allocated_bytes)
        << "fused " << fused.allocated_bytes << " against an unfused-but-packed "
        << planned.allocated_bytes;
}

/**
 * @brief Tests that packed and unpacked training agree byte for byte.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_StaticMemory, PackedAndUnpackedTrainingAgreeByteForByte)
{
    SushiBLAS::Engine unplanned_engine(SaTest::TestEnvironment::context());
    const Outcome unplanned = train(unplanned_engine, false);

    SushiBLAS::Engine planned_engine(SaTest::TestEnvironment::context());
    const Outcome planned = train(planned_engine, true);

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
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_StaticMemory, APackedZeroBufferIsRefilledOnEveryReplay)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId zeros = builder.zeros(Shape{4, 3});
    const ValueId sum = builder.add(zeros, x);

    TensorPool pool(*engine);
    pool.adopt(MemoryPlan::build(graph));
    pool.materialise_leaves(graph);

    const std::vector<double> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    SaTest::write_tensor(pool.tensor(x), input);

    Graph::CompiledStep step(*engine, graph, pool);

    step.run();
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(sum)), input);

    // Stand in for "a later value was given these bytes".
    SaTest::write_tensor(pool.tensor(zeros), std::vector<double>(12, 999.0));

    step.run();
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(sum)), input)
        << "the ZERO buffer was not refilled, so its bytes cannot be reused by anything";
}
