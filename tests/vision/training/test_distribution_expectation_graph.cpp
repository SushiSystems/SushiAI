/**************************************************************************/
/* test_distribution_expectation_graph.cpp                                */
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

/**
 * @file test_distribution_expectation_graph.cpp
 * @brief Tests the expectation graph's nodes, and on an engine its agreement with the host decode.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/detect/free/distribution_expectation.hpp>
#include <SushiAI/vision/training/distribution_expectation_graph.hpp>

using namespace SushiAI;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using Vision::Training::emit_distribution_expectation;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the bins per side of the cases. */
    constexpr std::int64_t K_BINS = 4;

    /** @brief Holds the box output of one image on a 1x2 grid, 32 values. */
    const Shape K_BOX_SHAPE{1, 1, 2, 4 * K_BINS};

    /** @brief Seeds the logits of the second anchor. */
    constexpr std::uint64_t K_LOGIT_SEED = 20261025;

    /** @brief Bounds the error of a float32 expectation below 4 against the double host decode. */
    constexpr double K_VALUE_TOLERANCE = 1e-5;

    /** @brief Counts the nodes of @p graph whose operation is @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }
} // namespace

/** @brief Tests that one bin per side returns the box itself and emits nothing. */
TEST(Medium_DistributionExpectationGraph, OneBinReturnsTheBoxAndEmitsNothing)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID box = builder.input(Shape{1, 2, 2, 4}, "box");
    const std::size_t before = graph.node_count();

    const ValueID expected = emit_distribution_expectation(builder, graph, box, Graph::K_NO_VALUE);

    EXPECT_EQ(expected, box);
    EXPECT_EQ(graph.node_count(), before);
}

/** @brief Tests that four bins give a [B, H, W, 4] value from one softmax and one product. */
TEST(Medium_DistributionExpectationGraph, FourBinsGiveFourSidesFromOneSoftmaxAndOneProduct)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID box = builder.input(K_BOX_SHAPE, "box", DType::Float64);
    const ValueID projection = builder.input(Shape{K_BINS, 1}, "projection", DType::Float64);

    const ValueID expected = emit_distribution_expectation(builder, graph, box, projection);

    EXPECT_EQ(graph.value(expected).shape, (Shape{1, 1, 2, 4}));
    EXPECT_EQ(graph.value(expected).dtype, DType::Float64);
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_EXP.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 1u);
}

namespace
{
    class Medium_DistributionExpectationGraphEndToEnd : public SaTest::EngineTest {};
} // namespace

/** @brief Tests that the engine's expectations equal expect_distances() on the same logits. */
TEST_F(Medium_DistributionExpectationGraphEndToEnd, TheGraphAgreesWithTheHostDecode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID box = builder.input(K_BOX_SHAPE, "box");
    const ValueID projection = builder.input(Shape{K_BINS, 1}, "projection");
    const ValueID expected = emit_distribution_expectation(builder, graph, box, projection);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    Result<std::unique_ptr<CompiledStep>> compiled = CompiledStep::compile(*engine, graph, pool);
    SA_EXPECT_OK(compiled);
    ASSERT_TRUE(compiled.has_value());
    const std::unique_ptr<CompiledStep> step = std::move(compiled).value();

    // The first anchor holds the hand distributions: means 1.5, 2, 1 and 1.5.
    std::vector<double> logits = {0.0,         0.0,         0.0,         0.0,
                                  std::log(1), std::log(2), std::log(3), std::log(4),
                                  std::log(4), std::log(3), std::log(2), std::log(1),
                                  5.0,         5.0,         5.0,         5.0};
    std::vector<double> noise(static_cast<std::size_t>(4 * K_BINS));
    SaTest::HostNoise(K_LOGIT_SEED).fill(noise);
    logits.insert(logits.end(), noise.begin(), noise.end());
    const std::vector<double> bins = {0.0, 1.0, 2.0, 3.0};
    SaTest::write_tensor(pool.tensor(box), logits);
    SaTest::write_tensor(pool.tensor(projection), bins);

    SA_EXPECT_OK(step->run());

    // The host decodes the float32 values the tensor holds.
    const std::vector<float> stored(logits.begin(), logits.end());
    std::vector<float> host(8, 0.0f);
    Vision::Detect::Free::expect_distances(stored, K_BINS, host);
    const std::vector<double> device = SaTest::read_tensor(pool.tensor(expected));
    ASSERT_EQ(device.size(), host.size());
    for (std::size_t i = 0; i < host.size(); ++i)
        EXPECT_NEAR(device[i], static_cast<double>(host[i]), K_VALUE_TOLERANCE) << "side " << i;
    EXPECT_NEAR(device[1], 2.0, K_VALUE_TOLERANCE);
    EXPECT_NEAR(device[2], 1.0, K_VALUE_TOLERANCE);
}
