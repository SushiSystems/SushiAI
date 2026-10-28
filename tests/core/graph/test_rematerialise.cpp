/**************************************************************************/
/* test_rematerialise.cpp                                                 */
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
 * @file test_rematerialise.cpp
 * @brief Tests that the IM2COL rematerialisation pass leaves the computed numbers unchanged.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/rematerialise.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    class Medium_Rematerialise : public SaTest::EngineTest {};

    /**
     * @brief Returns the non-square [1, 5, 4, 2] image the convolution reads; see
     *        tests/core/graph/README.md.
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
        Graph::Lifetime cols_lifetime = Graph::Lifetime::Activation;
        std::size_t planned_bytes = 0;
    };

    /**
     * @brief Builds, differentiates, optionally rematerialises, runs.
     *
     * One function for both arms so the two graphs differ in exactly the thing
     * under test. FLOAT64 throughout, as every numeric autograd test here is,
     * so a difference cannot be hidden under rounding. The results go to @p outcome.
     */
    void run_convolution(SushiBLAS::Engine& engine, bool rematerialise, Outcome& outcome)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{1, 5, 4, 2}, "x", DType::Float64);
        const ValueID w = builder.parameter(Shape{18, 4}, "w", DType::Float64);

        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        SA_ASSERT_OK_AND_ASSIGN(const ValueID cols, builder.im2col(x, g));
        SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.matmul(cols, w));
        const ValueID target = builder.input(graph.value(out).shape, "target", DType::Float64);
        SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                                builder.nll_loss(builder.log_softmax(out), target));
        SA_EXPECT_OK(graph.finalise());

        const std::array<ValueID, 1> wrt = {w};
        const std::vector<ValueID> grads = Autograd::differentiate(graph, loss,
            SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).value();
        const ValueID dw = grads.front();

        if (rematerialise) Graph::rematerialise_im2col(graph);

        outcome.nodes = graph.node_count();
        outcome.cols_lifetime = graph.value(cols).lifetime;
        for (const Graph::Node& node : graph.nodes())
            if (node.op == Ops::K_IM2COL.id) ++outcome.im2col_nodes;

        const Graph::MemoryPlan plan = Graph::MemoryPlan::build(graph).value();
        outcome.planned_bytes = plan.arena_bytes();

        TensorPool pool(engine);
        SA_EXPECT_OK(pool.adopt(plan));
        SA_EXPECT_OK(pool.materialise_leaves(graph));

        write_tensor(pool.tensor(x), image_values());
        write_tensor(pool.tensor(w), weight_values());
        write_tensor(pool.tensor(target), target_values());

        SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledStep> step,
                                CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());

        outcome.loss = read_tensor(pool.tensor(loss));
        outcome.weight_gradient = read_tensor(pool.tensor(dw));
    }
}

TEST_F(Medium_Rematerialise, TheRewrittenGraphComputesTheSameGradient)
{
    Outcome stored;
    ASSERT_NO_FATAL_FAILURE(run_convolution(*engine, false, stored));
    Outcome recomputed;
    ASSERT_NO_FATAL_FAILURE(run_convolution(*engine, true, recomputed));

    // The pass must have inserted a recomputation; see tests/core/graph/README.md.
    ASSERT_EQ(stored.im2col_nodes, std::size_t{1});
    ASSERT_EQ(recomputed.im2col_nodes, std::size_t{2}) << "no recomputation was inserted";
    ASSERT_EQ(recomputed.nodes, stored.nodes + 1);
    ASSERT_EQ(stored.cols_lifetime, Graph::Lifetime::Activation);
    ASSERT_EQ(recomputed.cols_lifetime, Graph::Lifetime::Scratch);

    // Equality is exact, not within a tolerance; see tests/core/graph/README.md.
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

TEST_F(Medium_Rematerialise, TheColumnMatrixLeavesTheActivationArena)
{
    // The planner must agree with the lifetime; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 5, 4, 2}, "x", DType::Float64);
    const ValueID w = builder.parameter(Shape{18, 4}, "w", DType::Float64);

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;
    g.pad_top = 1;  g.pad_left = 1;

    SA_ASSERT_OK_AND_ASSIGN(const ValueID cols, builder.im2col(x, g));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.matmul(cols, w));
    const ValueID target = builder.input(graph.value(out).shape, "target", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss, builder.nll_loss(builder.log_softmax(out), target));
    SA_EXPECT_OK(graph.finalise());

    const std::array<ValueID, 1> wrt = {w};
    SA_EXPECT_OK(Autograd::differentiate(graph, loss,
        SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    SA_ASSERT_OK_AND_ASSIGN(const Graph::MemoryPlan before, Graph::MemoryPlan::build(graph));
    ASSERT_TRUE(before.is_packed(cols));
    EXPECT_EQ(before.arenas()[before.placement(cols).arena].lifetime,
              Graph::Lifetime::Activation);

    Graph::rematerialise_im2col(graph);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::MemoryPlan after, Graph::MemoryPlan::build(graph));
    ASSERT_TRUE(after.is_packed(cols));
    EXPECT_EQ(after.arenas()[after.placement(cols).arena].lifetime, Graph::Lifetime::Scratch)
        << "the planner still packs the column matrix with the activations";
}
