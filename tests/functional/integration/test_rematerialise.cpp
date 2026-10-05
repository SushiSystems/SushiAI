/****************************************************************/
/* test_rematerialise.cpp                                       */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The half of the IM2COL rematerialisation pass that the host-lane tier cannot
// reach. Unit_Rematerialise proves the rewrite has the shape it should; this
// proves the rewritten graph computes the same numbers.
//
// "The pass ran and the graph still executes" would be worth nothing here. The
// pass deletes a stored value and recomputes it, and every way that can go
// wrong - a clone reading a different operand, a clone whose attributes were
// rebuilt from the geometry instead of copied - produces a column matrix of
// exactly the right shape holding the wrong pixels. So the assertion is
// equality against the same graph run without the pass, and it is exact
// equality rather than a tolerance: the clone recomputes the same operation
// from the same inputs, and SushiRuntime's `sushiruntime_deterministic_fp`
// contract is what makes "the same" mean bit-for-bit.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/rematerialise.hpp>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    class Integration_Rematerialise : public SaTest::EngineTest {};

    /**
     * @brief The image the convolution below reads: [1, 5, 4, 2].
     *
     * Deliberately not square. IM2COL's output extents live in attribute slots
     * 8 and 9, and the way a clone goes wrong is by re-deriving them instead of
     * copying them - which can yield the transpose of the real window grid,
     * with the same number of windows. On a square image that mistake is
     * invisible; on 5x4 it is a different gather.
     */
    std::vector<double> image_values()
    {
        std::vector<double> v(1 * 5 * 4 * 2);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = std::sin(static_cast<double>(i) * 0.7) * 1.5;
        return v;
    }

    /** @brief The 3x3x2 -> 4 filter bank, as an [18, 4] matrix. */
    std::vector<double> weight_values()
    {
        std::vector<double> v(18 * 4);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = std::cos(static_cast<double>(i) * 0.31) * 0.4;
        return v;
    }

    /** @brief A one-hot target over the 20 window positions and 4 classes. */
    std::vector<double> target_values()
    {
        std::vector<double> v(20 * 4, 0.0);
        for (std::size_t row = 0; row < 20; ++row)
            v[row * 4 + (row % 4)] = 1.0;
        return v;
    }

    /** @brief What one run of the convolution produces. */
    struct Outcome
    {
        std::vector<double> loss;
        std::vector<double> weight_gradient;
        std::size_t nodes = 0;
        std::size_t im2col_nodes = 0;
        Graph::Lifetime cols_lifetime = Graph::Lifetime::ACTIVATION;
        std::size_t planned_bytes = 0;
    };

    /**
     * @brief Builds, differentiates, optionally rematerialises, runs.
     *
     * One function for both arms so the two graphs differ in exactly the thing
     * under test. FLOAT64 throughout, as every numeric autograd test here is,
     * so a difference cannot be hidden under rounding.
     */
    Outcome run_convolution(SushiBLAS::Engine& engine, bool rematerialise)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.input(Shape{1, 5, 4, 2}, "x", DType::FLOAT64);
        const ValueId w = builder.parameter(Shape{18, 4}, "w", DType::FLOAT64);

        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueId cols = builder.im2col(x, g);
        const ValueId out = builder.matmul(cols, w);
        const ValueId target = builder.input(graph.value(out).shape, "target", DType::FLOAT64);
        const ValueId loss = builder.nll_loss(builder.log_softmax(out), target);
        graph.finalise();

        const std::array<ValueId, 1> wrt = {w};
        const std::vector<ValueId> grads = Autograd::differentiate(graph, loss,
            SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));
        const ValueId dw = grads.front();

        if (rematerialise) Graph::rematerialise_im2col(graph);

        Outcome outcome;
        outcome.nodes = graph.node_count();
        outcome.cols_lifetime = graph.value(cols).lifetime;
        for (const Graph::Node& node : graph.nodes())
            if (node.op == Ops::IM2COL.id) ++outcome.im2col_nodes;

        const Graph::MemoryPlan plan = Graph::MemoryPlan::build(graph);
        outcome.planned_bytes = plan.arena_bytes();

        TensorPool pool(engine);
        pool.adopt(plan);
        pool.materialise_leaves(graph);

        write_tensor(pool.tensor(x), image_values());
        write_tensor(pool.tensor(w), weight_values());
        write_tensor(pool.tensor(target), target_values());

        CompiledStep step(engine, graph, pool);
        step.run();

        outcome.loss = read_tensor(pool.tensor(loss));
        outcome.weight_gradient = read_tensor(pool.tensor(dw));
        return outcome;
    }
}

TEST_F(Integration_Rematerialise, TheRewrittenGraphComputesTheSameGradient)
{
    const Outcome stored = run_convolution(*engine, false);
    const Outcome recomputed = run_convolution(*engine, true);

    // First: the pass actually did something. A pass that silently no-ops
    // passes the comparison below trivially.
    ASSERT_EQ(stored.im2col_nodes, std::size_t{1});
    ASSERT_EQ(recomputed.im2col_nodes, std::size_t{2}) << "no recomputation was inserted";
    ASSERT_EQ(recomputed.nodes, stored.nodes + 1);
    ASSERT_EQ(stored.cols_lifetime, Graph::Lifetime::ACTIVATION);
    ASSERT_EQ(recomputed.cols_lifetime, Graph::Lifetime::SCRATCH);

    // Then: it changed nothing about the answer. Exact, not near - the clone
    // recomputes the same operation from the same operands in the same order.
    ASSERT_EQ(recomputed.loss.size(), stored.loss.size());
    for (std::size_t i = 0; i < stored.loss.size(); ++i)
        EXPECT_EQ(recomputed.loss[i], stored.loss[i]) << "loss element " << i;

    ASSERT_EQ(recomputed.weight_gradient.size(), stored.weight_gradient.size());
    ASSERT_EQ(recomputed.weight_gradient.size(), std::size_t{18 * 4});

    // A gradient of all zeros would compare equal without proving anything.
    double largest = 0.0;
    for (double value : stored.weight_gradient)
        largest = std::max(largest, std::abs(value));
    ASSERT_GT(largest, 1e-6) << "the reference gradient is empty; the comparison is vacuous";

    for (std::size_t i = 0; i < stored.weight_gradient.size(); ++i)
        EXPECT_EQ(recomputed.weight_gradient[i], stored.weight_gradient[i])
            << "weight gradient element " << i;
}

TEST_F(Integration_Rematerialise, TheColumnMatrixLeavesTheActivationArena)
{
    // The reason the pass exists, stated against the planner rather than
    // assumed from the lifetime field alone: the planner is what turns the
    // shortened live range into an arena decision, and it has to agree.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 5, 4, 2}, "x", DType::FLOAT64);
    const ValueId w = builder.parameter(Shape{18, 4}, "w", DType::FLOAT64);

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;
    g.pad_top = 1;  g.pad_left = 1;

    const ValueId cols = builder.im2col(x, g);
    const ValueId out = builder.matmul(cols, w);
    const ValueId target = builder.input(graph.value(out).shape, "target", DType::FLOAT64);
    const ValueId loss = builder.nll_loss(builder.log_softmax(out), target);
    graph.finalise();

    const std::array<ValueId, 1> wrt = {w};
    (void)Autograd::differentiate(graph, loss,
        SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    const Graph::MemoryPlan before = Graph::MemoryPlan::build(graph);
    ASSERT_TRUE(before.is_packed(cols));
    EXPECT_EQ(before.arenas()[before.placement(cols).arena].lifetime,
              Graph::Lifetime::ACTIVATION);

    Graph::rematerialise_im2col(graph);

    const Graph::MemoryPlan after = Graph::MemoryPlan::build(graph);
    ASSERT_TRUE(after.is_packed(cols));
    EXPECT_EQ(after.arenas()[after.placement(cols).arena].lifetime, Graph::Lifetime::SCRATCH)
        << "the planner still packs the column matrix with the activations";
}
