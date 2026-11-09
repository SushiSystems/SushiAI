/**************************************************************************/
/* test_loss.cpp                                                          */
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
 * @file test_loss.cpp
 * @brief Tests the nodes CrossEntropyLoss emits and the operands it refuses.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Graph::ValueID;
using NN::CrossEntropyLoss;
namespace SaTest = SushiAI::Test;

/** @brief Tests that the loss is a log-softmax of the logits read by an NLL against the target. */
TEST(Small_CrossEntropyLoss, TheLossIsTheLikelihoodOfALogSoftmax)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID logits = builder.input(Shape{4, 3}, "logits");
    const ValueID one_hot = builder.input(Shape{4, 3}, "one_hot");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            CrossEntropyLoss{}.forward(builder, logits, one_hot));

    ASSERT_EQ(graph.node_count(), nodes_before + 2u);
    const Graph::Node& softmax = graph.nodes()[nodes_before];
    const Graph::Node& likelihood = graph.nodes()[nodes_before + 1u];
    EXPECT_EQ(softmax.op, Ops::K_LOG_SOFTMAX.id);
    EXPECT_EQ(softmax.inputs, (std::vector<ValueID>{logits}));
    EXPECT_EQ(likelihood.op, Ops::K_NLL_LOSS.id);
    EXPECT_EQ(likelihood.inputs, (std::vector<ValueID>{softmax.outputs[0], one_hot}));
    EXPECT_EQ(likelihood.outputs, (std::vector<ValueID>{loss}));
}

/** @brief Tests that the loss holds one element of the logits' element type. */
TEST(Small_CrossEntropyLoss, TheLossIsOneElementOfTheLogitsType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID logits = builder.input(Shape{4, 3}, "logits", DType::Float64);
    const ValueID one_hot = builder.input(Shape{4, 3}, "one_hot", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            CrossEntropyLoss{}.forward(builder, logits, one_hot));

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
}

/** @brief Tests that logits and a target of two shapes are refused with the node's name. */
TEST(Small_CrossEntropyLoss, OperandsOfTwoShapesAreRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID logits = builder.input(Shape{4, 3}, "logits");
    const ValueID one_hot = builder.input(Shape{4, 2}, "one_hot");

    SA_EXPECT_FAILURE(CrossEntropyLoss{}.forward(builder, logits, one_hot),
                      StatusCode::InvalidDescription, "NLL_LOSS", "[4, 3]", "[4, 2]");
}

namespace
{
    /** @brief Provides the Engine the loss values are computed on. */
    class Medium_CrossEntropyLossValues : public SaTest::EngineTest
    {
    };

    /**
     * @brief Counts the roundings on the longest path to the loss: the four terms of a row's
     * normaliser, the exponential and the logarithm, the product with the target, the sum over the
     * four classes and the mean over three rows; 16 covers the 12 and leaves slack for the device's
     * exponential.
     */
    constexpr std::size_t K_LOSS_ROUNDED_OPERATIONS = 16;
}

/** @brief Tests that the loss is the mean over the batch of the per-sample cross-entropy. */
TEST_F(Medium_CrossEntropyLossValues, TheLossIsTheMeanOverTheBatchNotTheSum)
{
    constexpr std::size_t K_ROWS = 3;
    constexpr std::size_t K_CLASSES = 4;
    const std::vector<double> logits{1.0, 2.0, 0.5, -1.0, 0.0, 0.25, 3.0, -2.0, 1.5, 1.5, 0.0, 0.5};
    const std::vector<double> targets{0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0};

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{3, 4}, "logits");
    const ValueID t = builder.input(Shape{3, 4}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss, CrossEntropyLoss{}.forward(builder, x, t));
    Graph::TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), logits);
    SaTest::write_tensor(pool.tensor(t), targets);
    std::unique_ptr<Graph::CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, Graph::CompiledStep::compile(backend(), graph, pool));
    SA_EXPECT_OK(step->run());

    double expected = 0.0;
    for (std::size_t row = 0; row < K_ROWS; ++row)
    {
        double normaliser = 0.0;
        for (std::size_t c = 0; c < K_CLASSES; ++c)
            normaliser += std::exp(logits[row * K_CLASSES + c]);
        for (std::size_t c = 0; c < K_CLASSES; ++c)
            expected -= targets[row * K_CLASSES + c] *
                        (logits[row * K_CLASSES + c] - std::log(normaliser));
    }
    expected /= static_cast<double>(K_ROWS);

    const std::vector<double> read = SaTest::read_tensor(pool.tensor(loss));
    ASSERT_EQ(read.size(), 1u);
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<float>(read[0], expected, K_LOSS_ROUNDED_OPERATIONS)));
}
