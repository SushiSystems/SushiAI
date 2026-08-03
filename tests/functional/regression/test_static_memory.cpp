/**************************************************************************/
/* test_static_memory.cpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Four guarantees of the static memory planner whose failure would be silent -
// the loop still converges, the run still reports success, and nobody notices:
//
//  1. A training loop allocates NOTHING after the first step. Counted, not
//     eyeballed: every Storage holds a counted reference to the context's
//     allocator, so the allocator's reference count is the live-allocation
//     count plus a constant, and one extra buffer moves it by exactly one.
//  2. Packing shrinks the footprint. A pass that quietly stopped packing would
//     leave every other test in this suite green. Measured on the *unfused*
//     graph, because fusion reaches the same bytes by a different route -
//     deleting the intermediates rather than overlapping them - and a planner
//     test that let fusion do its work first would be measuring the catalog.
//  3. Packing changes no number. If it does, something is aliasing that should
//     not be, and the right response is to find it rather than to re-record a
//     reference.
//  4. A buffer the planner packed is REWRITTEN every replay. A ZERO buffer that
//     is filled once and then merely read holds zeros for exactly one step and
//     then whatever overwrote it - which trains, badly, for ever.

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
    constexpr int64_t FEATURES = 16;
    constexpr int64_t HIDDEN = 64;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 3;

    /** @brief The synthetic problem every run below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = BATCHES;
        options.features = FEATURES;
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
     *
     * "Off" is a pool that has already adopted the empty plan, which covers no
     * value and asks for no arena - the exact behaviour the library had before
     * this pass existed, and therefore the only honest control.
     *
     * @param engine The engine to train on.
     * @param packed Whether to let bind_parameters build a real plan.
     * @param fusion What the fusion pass is allowed to do. It is a real
     *        confound here and not a detail: a fused subtree's intermediates
     *        stop being values at all, so fusion removes exactly the buffers
     *        packing used to overlap. Every claim below therefore has to say
     *        which graph it is about.
     * @return The trained parameters, the loss trajectory and the footprint.
     */
    Outcome train(SushiBLAS::Engine& engine, bool packed, Graph::FusionOptions fusion = {})
    {
        Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
        Train::TrainingGraph built = Train::build_classifier(
            model, BATCH, FEATURES, CLASSES, DType::FLOAT32, fusion);

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

// =========================================================================
// 1. Zero allocation in the steady state
// =========================================================================

TEST_F(Regression_StaticMemory, ATrainingLoopAllocatesNothingAfterTheFirstStep)
{
    // The headline property of the whole pass, counted rather than claimed.
    // The baseline is taken AFTER the first step, so a one-time lazy cache
    // inside the runtime is tolerated while a per-step allocation is not - a
    // per-step buffer would still train correctly and would only show up as a
    // slowly growing process.
    const auto live_allocations = [this]
    {
        return engine->get_context().get_allocator()->get_ref_count();
    };

    Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

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
        trainer.run_step(dataset.batch(static_cast<std::size_t>(step) % BATCHES));
        engine->get_context().wait_all();

        EXPECT_EQ(live_allocations(), baseline)
            << "step " << step << " allocated " << (live_allocations() - baseline)
            << " buffer(s); a compiled step must replay into the memory it already has";
        EXPECT_EQ(pool.allocated_bytes(), bytes_baseline);
    }
}

// =========================================================================
// 2. The saving is real
// =========================================================================

/** @brief The unfused graph - the one the planner's saving is a claim about. */
Graph::FusionOptions unfused()
{
    Graph::FusionOptions options;
    options.enabled = false;
    return options;
}

TEST_F(Regression_StaticMemory, PackingShrinksTheFootprintOfTheSameModel)
{
    // Deliberately measured on the **unfused** graph. Fusion is not a confound
    // to be worked around here, it is a different mechanism reaching the same
    // bytes: a fused subtree's intermediates stop being values, so they need no
    // slot rather than a shared one. Planning what fusion already deleted would
    // make this test a statement about the fusion catalog rather than about the
    // planner, and it would go quiet the day a new catalog entry landed.
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

TEST_F(Regression_StaticMemory, FusingTakesTheBuffersPackingUsedToOverlap)
{
    // The interaction, stated rather than left to be rediscovered, because on
    // the fused graph the planner's own report reads like a broken planner:
    // every remaining activation's live range intersects every other's, so it
    // overlaps *nothing* and prints "0% saved". It is not broken. The buffers
    // it used to overlap have stopped existing - a fused subtree's
    // intermediates are not values any more - which is the better outcome of
    // the two and is why the arena has nothing left to do.
    //
    // `unpacked_bytes` is the direct measure of that: it is what the same
    // tensors would cost one allocation each, so a drop there is buffers
    // disappearing rather than being shared.
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

    // Against the packed-but-unfused graph it is never worse - and on this
    // model it is exactly equal, because what fusion deletes here happens to be
    // precisely what the planner was overlapping. Written as <= rather than
    // == on purpose: the equality is a property of this model's shape, not of
    // the two passes, and pinning it would fail on the next catalog entry for
    // no reason anyone should have to investigate.
    EXPECT_LE(fused.allocated_bytes, planned.allocated_bytes)
        << "fused " << fused.allocated_bytes << " against an unfused-but-packed "
        << planned.allocated_bytes;
}

// =========================================================================
// 3. Packing changes no number
// =========================================================================

TEST_F(Regression_StaticMemory, PackedAndUnpackedTrainingAgreeByteForByte)
{
    // If these ever differ, two tensors are sharing bytes they should not - and
    // the fix is to find which, never to relax the comparison. Bytes rather
    // than a tolerance for the reason test_determinism.cpp gives: a tolerance
    // cannot tell a rounding difference from a real overlap.
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

// =========================================================================
// 4. A packed buffer is rewritten every replay
// =========================================================================

TEST_F(Regression_StaticMemory, APackedZeroBufferIsRefilledOnEveryReplay)
{
    // ZERO is the one operation whose output nothing else writes, so it is the
    // one that would work perfectly if it were filled once outside the step -
    // right up until the planner handed its bytes to a later value. The buffer
    // is corrupted from the host between two replays here, which is exactly
    // what a reused arena slot does, and the second replay must still see zeros.
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
