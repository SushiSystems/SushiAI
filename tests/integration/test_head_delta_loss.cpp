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
 * @brief Tests on an engine that the delta loss has the gradient -scale * delta at the head.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/training/head_delta_loss.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using Vision::Training::emit_head_delta_loss;

namespace
{
    class Integration_HeadDeltaLoss : public SaTest::EngineTest {};

    /** @brief Holds the factor under test, one over the batch of two. */
    constexpr double K_SCALE = 0.5;

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
            loss = emit_head_delta_loss(builder, head, delta, K_SCALE);

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
TEST_F(Integration_HeadDeltaLoss, TheGradientAtTheHeadIsMinusScaleTimesDelta)
{
    DeltaLossCase model(*engine);
    ASSERT_NE(model.step, nullptr);
    ASSERT_NE(model.gradient, Graph::K_NO_VALUE);

    SA_EXPECT_OK(model.step->run());

    const std::vector<double> gradient = SaTest::read_tensor(model.pool.tensor(model.gradient));
    ASSERT_EQ(gradient.size(), model.delta_values.size());

    for (std::size_t i = 0; i < gradient.size(); ++i)
        EXPECT_NEAR(gradient[i], -K_SCALE * model.delta_values[i], K_GRADIENT_TOLERANCE)
            << "element " << i;
}

/** @brief Tests that the root's value is -scale * sum(delta * head), summed in double, to 1e-4. */
TEST_F(Integration_HeadDeltaLoss, TheValueIsMinusScaleTimesTheSumOfProducts)
{
    DeltaLossCase model(*engine);
    ASSERT_NE(model.step, nullptr);

    SA_EXPECT_OK(model.step->run());

    double sum = 0.0;
    for (std::size_t i = 0; i < model.delta_values.size(); ++i)
        sum += model.delta_values[i] * model.head_values[i];

    const std::vector<double> value = SaTest::read_tensor(model.pool.tensor(model.loss));
    ASSERT_EQ(value.size(), 1u);
    EXPECT_NEAR(value[0], -K_SCALE * sum, K_VALUE_TOLERANCE);
}
