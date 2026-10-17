/**************************************************************************/
/* test_head_delta_loss.cpp                                               */
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
 * @file test_head_delta_loss.cpp
 * @brief Tests the head delta loss: its graph, and on an engine the gradient -scale * delta at the
 *        head.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/training/head_delta_loss.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueID;
using Vision::Training::emit_head_delta_loss;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::TensorPool;

namespace
{
    /** @brief Bounds the error allowed on the factor the SCALE node stores. */
    constexpr double K_ATTRIBUTE_TOLERANCE = 1e-12;

    /** @brief Holds the factor the tests pass, one over a batch of four. */
    constexpr double K_SCALE = 0.25;

    /** @brief Returns a head of two samples on a 3x3 grid with seven channels, 126 elements. */
    Shape head_shape()
    {
        return Shape{2, 3, 3, 7};
    }

    /** @brief Returns the operation names of nodes [from, end). */
    std::vector<std::string> op_names(const Graph::Graph& graph, std::size_t from)
    {
        std::vector<std::string> names;
        for (std::size_t i = from; i < graph.node_count(); ++i)
            names.emplace_back(Ops::name_of(graph.nodes()[i].op));
        return names;
    }
} // namespace

/** @brief Tests that the emitter appends MUL, RESHAPE, SUM_ROWS and SCALE, wired in that order. */
TEST(Medium_HeadDeltaLoss, EmitsFourNodesEndingInAScaleOfOneElement)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta");
    const std::size_t leaves = graph.node_count();

    const ValueID loss = emit_head_delta_loss(builder, head, delta, K_SCALE);

    ASSERT_EQ(op_names(graph, leaves), (std::vector<std::string>{
        "sushiai.mul",
        "sushiai.reshape",
        "sushiai.sum_rows",
        "sushiai.scale",
    }));

    const Graph::Node& product = graph.nodes()[leaves];
    const Graph::Node& column = graph.nodes()[leaves + 1];
    const Graph::Node& sum = graph.nodes()[leaves + 2];
    const Graph::Node& root = graph.nodes()[leaves + 3];

    EXPECT_EQ(product.inputs, (std::vector<ValueID>{delta, head}));
    EXPECT_EQ(column.inputs, (std::vector<ValueID>{product.outputs.at(0)}));
    EXPECT_EQ(sum.inputs, (std::vector<ValueID>{column.outputs.at(0)}));
    EXPECT_EQ(root.inputs, (std::vector<ValueID>{sum.outputs.at(0)}));
    EXPECT_EQ(root.outputs, (std::vector<ValueID>{loss}));

    EXPECT_EQ(graph.value(column.outputs.at(0)).shape, (Shape{126, 1}));
    EXPECT_EQ(graph.value(loss).shape, (Shape{1}));
    EXPECT_NEAR(root.attributes.get(0), -K_SCALE, K_ATTRIBUTE_TOLERANCE);
}

/** @brief Tests that the result carries the element type of the head. */
TEST(Medium_HeadDeltaLoss, TheResultHasTheElementTypeOfTheHead)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head", DType::Float64);
    const ValueID delta = builder.input(head_shape(), "delta", DType::Float64);

    const ValueID loss = emit_head_delta_loss(builder, head, delta, K_SCALE);

    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
}

/** @brief Tests that autograd takes the result as a root and returns a head-shaped gradient. */
TEST(Medium_HeadDeltaLoss, TheResultIsARootAutogradDifferentiates)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta");
    const ValueID loss = emit_head_delta_loss(builder, head, delta, K_SCALE);

    const std::vector<ValueID> wrt = {head};
    std::vector<ValueID> gradients;
    ASSERT_NO_THROW(gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_NE(gradients[0], Graph::K_NO_VALUE);
    EXPECT_EQ(graph.value(gradients[0]).shape, head_shape());
    EXPECT_FALSE(graph.value(delta).requires_grad);
}

/** @brief Tests that a delta of another shape throws, naming MUL and both shapes. */
TEST(Medium_HeadDeltaLoss, ADeltaOfAnotherShapeThrowsNamingBothShapes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(Shape{2, 3, 3, 6}, "delta");
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_ERROR((void)emit_head_delta_loss(builder, head, delta, K_SCALE),
                    "MUL needs equally shaped operands", "[2, 3, 3, 6]", "[2, 3, 3, 7]");
    EXPECT_EQ(graph.node_count(), leaves);
}

/** @brief Tests that a delta of another element type throws, naming MUL and both types. */
TEST(Medium_HeadDeltaLoss, ADeltaOfAnotherElementTypeThrowsNamingBothTypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID head = builder.parameter(head_shape(), "head");
    const ValueID delta = builder.input(head_shape(), "delta", DType::Float64);
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_ERROR((void)emit_head_delta_loss(builder, head, delta, K_SCALE),
                    "MUL needs operands of one element type", "float32", "float64");
    EXPECT_EQ(graph.node_count(), leaves);
}

namespace
{
    class Medium_HeadDeltaLossEndToEnd : public SaTest::EngineTest {};

    /** @brief Holds the factor under test, one over the batch of two. */
    constexpr double K_ENGINE_SCALE = 0.5;

    /** @brief Seeds the head's values. */
    constexpr std::uint64_t K_HEAD_SEED = 20261009;

    /** @brief Seeds the delta's values. */
    constexpr std::uint64_t K_DELTA_SEED = 20261010;

    /** @brief Bounds the absolute error of each float32 gradient element. */
    constexpr double K_GRADIENT_TOLERANCE = 1e-6;

    /** @brief Bounds the absolute error of the float32 sum of 126 products below 1. */
    constexpr double K_VALUE_TOLERANCE = 1e-4;

    /** @brief Holds the graph of one head and its delta, compiled, with host copies of both. */
    struct DeltaLossCase
    {
        /**
         * @brief Builds, differentiates and compiles the loss, then writes the seeded leaves.
         * @param engine Must outlive this object.
         */
        explicit DeltaLossCase(SushiBLAS::Engine& engine) : pool(engine)
        {
            const GraphBuilder builder(graph);
            const Shape shape{2, 3, 3, 7};

            head = builder.parameter(shape, "head");
            delta = builder.input(shape, "delta");
            loss = emit_head_delta_loss(builder, head, delta, K_ENGINE_SCALE);

            const std::vector<ValueID> wrt = {head};
            gradient = Autograd::differentiate(
                graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).at(0);

            SA_EXPECT_OK(pool.materialise_leaves(graph));
            Result<std::unique_ptr<CompiledStep>> compiled =
                CompiledStep::compile(engine, graph, pool);
            SA_EXPECT_OK(compiled);
            if (compiled.has_value())
                step = std::move(compiled).value();

            const std::size_t count = static_cast<std::size_t>(shape.num_elements());
            std::vector<double> head_noise(count);
            std::vector<double> delta_noise(count);
            SaTest::HostNoise(K_HEAD_SEED).fill(head_noise);
            SaTest::HostNoise(K_DELTA_SEED).fill(delta_noise);

            // Every third delta is zero, as most entries of a real head's delta are.
            for (std::size_t i = 0; i < count; i += 3)
                delta_noise[i] = 0.0;

            SaTest::write_tensor(pool.tensor(head), head_noise);
            SaTest::write_tensor(pool.tensor(delta), delta_noise);

            // The tensors hold float32, so the expectation starts from the rounded values.
            for (std::size_t i = 0; i < count; ++i)
            {
                head_values.push_back(static_cast<double>(static_cast<float>(head_noise[i])));
                delta_values.push_back(static_cast<double>(static_cast<float>(delta_noise[i])));
            }
        }

        Graph::Graph graph;                      /**< @brief Holds the graph under test. */
        TensorPool pool;                         /**< @brief Holds the tensors of the graph. */
        std::unique_ptr<CompiledStep> step;      /**< @brief Holds the step, null when refused. */
        ValueID head = Graph::K_NO_VALUE;        /**< @brief Holds the head, a parameter. */
        ValueID delta = Graph::K_NO_VALUE;       /**< @brief Holds the delta, an input leaf. */
        ValueID loss = Graph::K_NO_VALUE;        /**< @brief Holds the emitted root. */
        ValueID gradient = Graph::K_NO_VALUE;    /**< @brief Holds the gradient at the head. */
        std::vector<double> head_values;         /**< @brief Holds the head as the tensor has it. */
        std::vector<double> delta_values;        /**< @brief Holds the delta the tensor holds. */
    };
} // namespace

/** @brief Tests that the gradient at the head is -scale * delta, element by element, to 1e-6. */
TEST_F(Medium_HeadDeltaLossEndToEnd, TheGradientAtTheHeadIsMinusScaleTimesDelta)
{
    DeltaLossCase model(*engine);
    ASSERT_NE(model.step, nullptr);
    ASSERT_NE(model.gradient, Graph::K_NO_VALUE);

    SA_EXPECT_OK(model.step->run());

    const std::vector<double> gradient = SaTest::read_tensor(model.pool.tensor(model.gradient));
    ASSERT_EQ(gradient.size(), model.delta_values.size());

    for (std::size_t i = 0; i < gradient.size(); ++i)
        EXPECT_NEAR(gradient[i], -K_ENGINE_SCALE * model.delta_values[i], K_GRADIENT_TOLERANCE)
            << "element " << i;
}

/** @brief Tests that the root's value is -scale * sum(delta * head), summed in double, to 1e-4. */
TEST_F(Medium_HeadDeltaLossEndToEnd, TheValueIsMinusScaleTimesTheSumOfProducts)
{
    DeltaLossCase model(*engine);
    ASSERT_NE(model.step, nullptr);

    SA_EXPECT_OK(model.step->run());

    double sum = 0.0;
    for (std::size_t i = 0; i < model.delta_values.size(); ++i)
        sum += model.delta_values[i] * model.head_values[i];

    const std::vector<double> value = SaTest::read_tensor(model.pool.tensor(model.loss));
    ASSERT_EQ(value.size(), 1u);
    EXPECT_NEAR(value[0], -K_ENGINE_SCALE * sum, K_VALUE_TOLERANCE);
}
