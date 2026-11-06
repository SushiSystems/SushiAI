/**************************************************************************/
/* test_hand_values.cpp                                                   */
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
 * @file test_hand_values.cpp
 * @brief Tests every lowered operation against values worked out by hand.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /**
     * @brief Bounds the absolute error of a float result; 2048 unit roundoffs of float, the
     * smallest power of two that covers 1e-4.
     */
    constexpr double K_FLOAT_LOOSE_TOLERANCE = 2048.0 * TestSupport::unit_roundoff<float>();

    /**
     * @brief Bounds the absolute error of a float result; 256 unit roundoffs of float, the smallest
     * power of two that covers 1e-5.
     */
    constexpr double K_FLOAT_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /**
     * @brief Bounds the absolute error of a float result; 32 unit roundoffs of float, the smallest
     * power of two that covers 1e-6.
     */
    constexpr double K_FLOAT_TIGHT_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /**
     * @brief Bounds the absolute error of a double result; 16777216 unit roundoffs of double, the
     * smallest power of two that covers 1e-9.
     */
    constexpr double K_DOUBLE_LOOSE_TOLERANCE = 16777216.0 * TestSupport::unit_roundoff<double>();

    /**
     * @brief Bounds the absolute error of a double result; 16384 unit roundoffs of double, the
     * smallest power of two that covers 1e-12.
     */
    constexpr double K_DOUBLE_TOLERANCE = 16384.0 * TestSupport::unit_roundoff<double>();

    class Medium_Forward : public SaTest::EngineTest {};

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Medium_ForwardDeathTest = Medium_Forward;

    /** @brief Lowers a graph, runs it once, and hands back the pool. */
    class OneShot
    {
        public:
            OneShot(SushiBLAS::Engine& engine, const Graph::Graph& graph)
                : pool(engine)
            {
                SA_EXPECT_OK(pool.materialise_leaves(graph));
                graph_ = &graph;
                engine_ = &engine;
            }

            /** @brief Compiles the graph; call after every leaf has been written. */
            void compile()
            {
                SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine_, *graph_, pool));
            }

            TensorPool pool;
            std::unique_ptr<CompiledStep> step;

        private:
            const Graph::Graph* graph_;
            SushiBLAS::Engine* engine_;
    };

    /** @brief Compares every element of @p got with @p expected to within @p absolute. */
    void expect_elements(const std::vector<double>& got, const std::vector<double>& expected,
                         double absolute)
    {
        ASSERT_EQ(got.size(), expected.size());

        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], absolute) << "at " << i;
    }

    /** @brief Returns @p function applied to every element of @p values. */
    template <typename Function>
    std::vector<double> mapped(const std::vector<double>& values, Function function)
    {
        std::vector<double> result(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            result[i] = function(values[i]);
        return result;
    }

    /** @brief Returns the sum of exp over @p count elements of @p values from @p first. */
    double exponential_mass(const std::vector<double>& values, std::size_t first,
                            std::size_t count)
    {
        double mass = 0.0;
        for (std::size_t i = first; i < first + count; ++i)
            mass += std::exp(values[i]);
        return mass;
    }
}

/** @brief Tests that a linear layer and the ReLU after it equal the values worked out by hand. */
TEST_F(Medium_Forward, LinearLayerMatchesHandComputedValues)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2}, "x");
    const ValueID w = builder.input(Shape{3, 2}, "w");
    const ValueID b = builder.input(Shape{3}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID y,
                            builder.bias_add(builder.matmul(x, w, false, true), b));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID z, builder.relu(y));

    OneShot run(*engine, graph);

    // X = [[1, 2], [3, 4]], W = [[1, 0], [0, 1], [1, 1]], b = [-2, -0.5, 1]
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(w), std::vector<double>{1, 0, 0, 1, 1, 1});
    write_tensor(run.pool.tensor(b), std::vector<double>{-2.0, -0.5, 1.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    // X * W^T = [[1, 2, 3], [3, 4, 7]], then + b.
    expect_elements(read_tensor(run.pool.tensor(y)),
                    std::vector<double>{-1.0, 1.5, 4.0, 1.0, 3.5, 8.0}, K_FLOAT_TOLERANCE);
    expect_elements(read_tensor(run.pool.tensor(z)),
                    std::vector<double>{0.0, 1.5, 4.0, 1.0, 3.5, 8.0}, K_FLOAT_TOLERANCE);
}

/** @brief Tests that sum_rows gives the sum of each column. */
TEST_F(Medium_Forward, SumRowsProducesColumnSums)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{3, 2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sums, builder.sum_rows(x));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 4, 5, 6});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(sums)), std::vector<double>{9.0, 12.0},
                    K_FLOAT_TOLERANCE);
}

/** @brief Tests that a reshape keeps the row-major linear order of the elements. */
TEST_F(Medium_Forward, ReshapeReinterpretsRowMajorLinearOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{3, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.reshape(x, Shape{2, 6}));

    OneShot run(*engine, graph);
    const std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    write_tensor(run.pool.tensor(x), values);

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(y)), values, K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that a layout convert to NCHW puts the channels outermost. */
TEST_F(Medium_Forward, LayoutConvertPermutesChannelsToTheFront)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // N=1, H=1, W=2, C=3: small enough to hand-trace every element's move.
    const ValueID x = builder.input(Shape{1, 1, 2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.layout_convert(x, Graph::TensorLayout::NCHW));

    OneShot run(*engine, graph);

    // NHWC, channels innermost: (w=0,c=0..2) = 0,1,2; (w=1,c=0..2) = 3,4,5.
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3, 4, 5});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    // For each c, the two W positions in order; see tests/core/graph/README.md.
    expect_elements(read_tensor(run.pool.tensor(y)), std::vector<double>{0, 3, 1, 4, 2, 5},
                    K_FLOAT_TOLERANCE);
}

/** @brief Tests that add writes the sum of its two operands. */
TEST_F(Medium_Forward, AddMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{2, 2}, "a");
    const ValueID b = builder.input(Shape{2, 2}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sum, builder.add(a, b));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(a), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(b), std::vector<double>{0.5, -1.0, 2.0, 0.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(sum)), std::vector<double>{1.5, 1.0, 5.0, 4.0},
                    K_FLOAT_TOLERANCE);
}

/** @brief Tests that sub writes the first operand minus the second. */
TEST_F(Medium_Forward, SubMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{2, 2}, "a");
    const ValueID b = builder.input(Shape{2, 2}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID difference, builder.sub(a, b));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(a), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(b), std::vector<double>{0.5, -1.0, 2.0, 0.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(difference)),
                    std::vector<double>{0.5, 3.0, 1.0, 4.0}, K_FLOAT_TOLERANCE);
}

/** @brief Tests that mul writes the elementwise product of its two operands. */
TEST_F(Medium_Forward, MulMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{2, 2}, "a");
    const ValueID b = builder.input(Shape{2, 2}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID product, builder.mul(a, b));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(a), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(b), std::vector<double>{0.5, -1.0, 2.0, 0.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(product)),
                    std::vector<double>{0.5, -2.0, 6.0, 0.0}, K_FLOAT_TOLERANCE);
}

/** @brief Tests that scale multiplies every element by its constant. */
TEST_F(Medium_Forward, ScaleMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{2, 2}, "a");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID scaled, builder.scale(a, -2.5));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(a), std::vector<double>{1, 2, 3, 4});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(scaled)),
                    std::vector<double>{-2.5, -5.0, -7.5, -10.0}, K_FLOAT_TOLERANCE);
}

/** @brief Tests that exp writes e raised to each element. */
TEST_F(Medium_Forward, ExpMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID b = builder.input(Shape{2, 2}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID exponent, builder.exp(b));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(b), std::vector<double>{0.5, -1.0, 2.0, 0.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(exponent)),
                    std::vector<double>{std::exp(0.5), std::exp(-1.0), std::exp(2.0), 1.0},
                    K_FLOAT_TOLERANCE);
}

/** @brief Tests that relu keeps the positive elements and zeroes the others. */
TEST_F(Medium_Forward, ReluMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID r, builder.relu(x));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{-2.0, -0.5, 0.5, 2.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(r)), std::vector<double>{0.0, 0.0, 0.5, 2.0},
                    K_FLOAT_TOLERANCE);
}

/** @brief Tests that sigmoid writes 1 / (1 + e^-x) for each element. */
TEST_F(Medium_Forward, SigmoidMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID s, builder.sigmoid(x));

    OneShot run(*engine, graph);
    const std::vector<double> inputs = {-2.0, -0.5, 0.5, 2.0};
    write_tensor(run.pool.tensor(x), inputs);

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(s)),
                    mapped(inputs, [](double v) { return 1.0 / (1.0 + std::exp(-v)); }),
                    K_FLOAT_TOLERANCE);
}

/** @brief Tests that tanh writes the hyperbolic tangent of each element. */
TEST_F(Medium_Forward, TanhMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID t, builder.tanh(x));

    OneShot run(*engine, graph);
    const std::vector<double> inputs = {-2.0, -0.5, 0.5, 2.0};
    write_tensor(run.pool.tensor(x), inputs);

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(t)),
                    mapped(inputs, [](double v) { return std::tanh(v); }), K_FLOAT_TOLERANCE);
}

/**
 * @brief Tests that leaky_relu scales the negative elements by its slope and keeps the others.
 *
 * Slope 0.1 with inputs on both sides of zero, so that both branches run: -2 gives -0.2, -0.5
 * gives -0.05, 0.5 gives 0.5, 2 gives 2.
 */
TEST_F(Medium_Forward, LeakyReluMatchesItsDefinition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID lr, builder.leaky_relu(x, 0.1));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{-2.0, -0.5, 0.5, 2.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(lr)),
                    std::vector<double>{-0.2, -0.05, 0.5, 2.0}, K_FLOAT_TOLERANCE);
}

/**
 * @brief Tests that the activations write a new buffer and leave their input as it was.
 *
 * The backward rules for relu and gelu read the forward input, so an in-place activation would
 * hand them the output.
 */
TEST_F(Medium_Forward, ActivationsLeaveTheirInputAlone)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 4}, "x");
    SA_EXPECT_OK(builder.relu(x));
    SA_EXPECT_OK(builder.sigmoid(x));
    SA_EXPECT_OK(builder.tanh(x));
    SA_EXPECT_OK(builder.leaky_relu(x, 0.1));

    OneShot run(*engine, graph);
    const std::vector<double> inputs = {-2.0, -0.5, 0.5, 2.0};
    write_tensor(run.pool.tensor(x), inputs);

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(x)), inputs, K_DOUBLE_TOLERANCE);
}

/** @brief Tests that log-softmax rows match the hand values and exponentiate to a sum of one. */
TEST_F(Medium_Forward, LogSoftmaxRowsSumToOneAfterExponentiating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID l, builder.log_softmax(x));

    OneShot run(*engine, graph);
    // Row two is row one shifted by a constant; see tests/core/graph/README.md.
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 51, 52, 53});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    const std::vector<double> l_values = read_tensor(run.pool.tensor(l));

    // log-sum-exp of [1, 2, 3] is 3 + log(1 + e^-1 + e^-2).
    const double lse = 3.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
    expect_elements(l_values,
                    std::vector<double>{1.0 - lse, 2.0 - lse, 3.0 - lse,
                                        1.0 - lse, 2.0 - lse, 3.0 - lse},
                    K_FLOAT_LOOSE_TOLERANCE);

    EXPECT_NEAR(exponential_mass(l_values, 0, 3), 1.0, K_FLOAT_LOOSE_TOLERANCE) << "row 0";
    EXPECT_NEAR(exponential_mass(l_values, 3, 3), 1.0, K_FLOAT_LOOSE_TOLERANCE) << "row 1";
}

/** @brief Tests that the cross-entropy of uniform logits is ln of the class count. */
TEST_F(Medium_Forward, CrossEntropyOnUniformLogitsIsLogC)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 5}, "x");
    const ValueID targets = builder.input(Shape{4, 5}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss, builder.nll_loss(builder.log_softmax(x), targets));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>(20, 0.0));

    // Row r targets class r.
    write_tensor(run.pool.tensor(targets),
                 std::vector<double>{1, 0, 0, 0, 0,
                                     0, 1, 0, 0, 0,
                                     0, 0, 1, 0, 0,
                                     0, 0, 0, 1, 0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    EXPECT_NEAR(read_tensor(run.pool.tensor(loss))[0], std::log(5.0), K_FLOAT_TOLERANCE);
}

/** @brief Tests that the cross-entropy of one row equals its hand-computed value. */
TEST_F(Medium_Forward, CrossEntropyMatchesAHandComputedSingleRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 3}, "x");
    const ValueID targets = builder.input(Shape{1, 3}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss, builder.nll_loss(builder.log_softmax(x), targets));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1.0, 2.0, 3.0});
    write_tensor(run.pool.tensor(targets), std::vector<double>{0.0, 0.0, 1.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    const double lse = 3.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
    EXPECT_NEAR(read_tensor(run.pool.tensor(loss))[0], lse - 3.0, K_FLOAT_TOLERANCE);
}

/** @brief Tests that accumulate adds into its first operand and returns that same buffer. */
TEST_F(Medium_Forward, AccumulateAddsInPlaceAndAliasesItsResult)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4}, "a");
    const ValueID b = builder.input(Shape{4}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID total, builder.accumulate(a, b));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(a), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(b), std::vector<double>{10, 20, 30, 40});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    const std::vector<double> result = read_tensor(run.pool.tensor(total));

    EXPECT_EQ(result, read_tensor(run.pool.tensor(a))) << "the result must alias the accumulator";
    expect_elements(result, std::vector<double>{11, 22, 33, 44}, K_FLOAT_TOLERANCE);
}

/** @brief Tests that a replay writes the zero buffer's bias sum again and does not accumulate. */
TEST_F(Medium_Forward, ZeroFillsItsBufferExactlyOnce)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID zeros = builder.zeros(Shape{2, 3}, DType::Float32);
    const ValueID bias = builder.input(Shape{3}, "bias");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID broadcast, builder.bias_add(zeros, bias));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(bias), std::vector<double>{1.0, -2.0, 3.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());
    SA_EXPECT_OK(run.step->run());

    // A replay must not accumulate into the result; see tests/core/graph/README.md.
    expect_elements(read_tensor(run.pool.tensor(broadcast)),
                    std::vector<double>{1.0, -2.0, 3.0, 1.0, -2.0, 3.0}, K_FLOAT_TOLERANCE);
}

/** @brief Tests that im2col gathers one window in tap order with the channel innermost. */
TEST_F(Medium_Forward, Im2colGathersOneWindowInTapOrderChannelInnermost)
{
    // One 2x2 window covers the whole image; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 2, 2, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID cols, builder.im2col(x, g));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(cols)), std::vector<double>{0, 1, 2, 3},
                    K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that col2im gives a pixel the sum of every window that covers it. */
TEST_F(Medium_Forward, Col2imSumsEveryWindowThatOverlapsAPixel)
{
    // A pixel receives the sum of its windows; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID cols = builder.input(Shape{4, 4}, "cols");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID image, builder.col2im(cols, Shape{1, 3, 3, 1}, g));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(cols), std::vector<double>(16, 1.0));

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(image)),
                    std::vector<double>{1, 2, 1, 2, 4, 2, 1, 2, 1}, K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that an asymmetrically padded im2col uses the extents it was given. */
TEST_F(Medium_Forward, AnAsymmetricallyPaddedIm2colUsesTheExtentsItWasGivenNotDerivedOnes)
{
    // out_h is 5 and out_w is 4, not the transpose; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 4, 5, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;
    g.pad_top = 1;  g.pad_left = 1;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID cols, builder.im2col_with_extents(x, g, 5, 4));

    OneShot run(*engine, graph);
    std::vector<double> image(20);
    std::iota(image.begin(), image.end(), 0.0);
    write_tensor(run.pool.tensor(x), image);

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    const std::vector<double> expected = {
        0,  0,  0,  0,  0,  1,  0,  5,  6,
        0,  0,  0,  0,  1,  2,  5,  6,  7,
        0,  0,  0,  1,  2,  3,  6,  7,  8,
        0,  0,  0,  2,  3,  4,  7,  8,  9,
        0,  0,  1,  0,  5,  6,  0, 10, 11,
        0,  1,  2,  5,  6,  7, 10, 11, 12,
        1,  2,  3,  6,  7,  8, 11, 12, 13,
        2,  3,  4,  7,  8,  9, 12, 13, 14,
        0,  5,  6,  0, 10, 11,  0, 15, 16,
        5,  6,  7, 10, 11, 12, 15, 16, 17,
        6,  7,  8, 11, 12, 13, 16, 17, 18,
        7,  8,  9, 12, 13, 14, 17, 18, 19,
        0, 10, 11,  0, 15, 16,  0,  0,  0,
       10, 11, 12, 15, 16, 17,  0,  0,  0,
       11, 12, 13, 16, 17, 18,  0,  0,  0,
       12, 13, 14, 17, 18, 19,  0,  0,  0,
        0, 15, 16,  0,  0,  0,  0,  0,  0,
       15, 16, 17,  0,  0,  0,  0,  0,  0,
       16, 17, 18,  0,  0,  0,  0,  0,  0,
       17, 18, 19,  0,  0,  0,  0,  0,  0,
    };
    expect_elements(read_tensor(run.pool.tensor(cols)), expected, K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that max-pooling a window gives its maximum. */
TEST_F(Medium_Forward, MaxPool2dTakesEachWindowsMaximum)
{
    // One 2x2 window over 0..3 has maximum 3; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 2, 2, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const Graph::PoolResult pooled, builder.maxpool2d(x, g));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(pooled.output)), std::vector<double>{3.0},
                    K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that a layout round trip permutes the buffer on the way out and restores it. */
TEST_F(Medium_Forward, ALayoutConvertRoundTripActuallyPermutesAndComesBack)
{
    // The permuted intermediate is the real test; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // Values 1..8 are distinct, in NHWC order; see tests/core/graph/README.md.
    const ValueID x = builder.input(Shape{1, 2, 2, 2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID nchw,
                            builder.layout_convert(x, Graph::TensorLayout::NCHW));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID back,
                            builder.layout_convert(nchw, Graph::TensorLayout::NHWC));
    SA_EXPECT_OK(graph.finalise());

    SA_EXPECT_OK(Graph::insert_casts(graph, Graph::CastOptions{}));
    const Graph::FusionReport fusion = Graph::fuse(graph, Graph::FusionOptions{});
    ASSERT_TRUE(fusion.enabled());
    ASSERT_EQ(fusion.nodes_after(), graph.node_count());

    // Extents match only because H, W and C are 2; see tests/core/graph/README.md.
    EXPECT_EQ(graph.value(nchw).shape, (Shape{1, 2, 2, 2}));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    // NCHW index is ((c*2)+h)*2+w; an intermediate that reads back 1..8 was copied, not permuted.
    expect_elements(read_tensor(run.pool.tensor(nchw)),
                    std::vector<double>{1, 3, 5, 7, 2, 4, 6, 8}, K_DOUBLE_LOOSE_TOLERANCE);
    expect_elements(read_tensor(run.pool.tensor(back)),
                    std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8}, K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that max-pooling survives the pass pipeline and still routes its gradient. */
TEST_F(Medium_Forward, AMaxPoolSurvivesTheFullPassPipelineAndStillRoutesItsGradient)
{
    // Passes run in Trainer::build order; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // Only taps 6 and 7 receive a gradient; see tests/core/graph/README.md.
    const ValueID x = builder.parameter(Shape{1, 2, 2, 2}, "x");
    const ValueID target = builder.input(Shape{1, 2}, "target");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;

    SA_ASSERT_OK_AND_ASSIGN(const Graph::PoolResult pooled, builder.maxpool2d(x, g));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID logits, builder.spatial_mean(pooled.output));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            builder.nll_loss(builder.log_softmax(logits), target));
    SA_EXPECT_OK(graph.finalise());

    const std::array<ValueID, 1> wrt = {x};
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ValueID> gradients,
                            Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_EQ(graph.value(pooled.indices).dtype, DType::Int64);

    SA_EXPECT_OK(Graph::insert_casts(graph, Graph::CastOptions{}));
    const Graph::FusionReport fusion = Graph::fuse(graph, Graph::FusionOptions{});
    ASSERT_TRUE(fusion.enabled());
    ASSERT_EQ(fusion.nodes_after(), graph.node_count());

    EXPECT_EQ(graph.value(pooled.indices).dtype, DType::Int64)
        << "the pass retyped the index buffer, so the backward is routing gradients by "
           "whatever a float reinterpretation of an offset table happens to say";

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7});
    write_tensor(run.pool.tensor(target), std::vector<double>{1.0, 0.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(pooled.output)), std::vector<double>{6.0, 7.0},
                    K_DOUBLE_LOOSE_TOLERANCE);

    const std::vector<double> dx = read_tensor(run.pool.tensor(gradients[0]));
    ASSERT_EQ(dx.size(), 8u);

    // Taps 0 to 5 did not win their window; taps 6 and 7 are the maxima of the two channels.
    expect_elements(std::vector<double>(dx.begin(), dx.begin() + 6),
                    std::vector<double>(6, 0.0), K_DOUBLE_LOOSE_TOLERANCE);
    EXPECT_GT(std::abs(dx[6]), K_FLOAT_TIGHT_TOLERANCE)
        << "the c=0 maximum received no gradient at all";
    EXPECT_GT(std::abs(dx[7]), K_FLOAT_TIGHT_TOLERANCE)
        << "the c=1 maximum received no gradient at all";
}

/** @brief Tests that average-pooling divides the window sum by the full window area. */
TEST_F(Medium_Forward, AvgPool2dDividesByTheFullWindowArea)
{
    // Same window as the max-pool case above; the average of 0,1,2,3 is 1.5.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 2, 2, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID pooled, builder.avgpool2d(x, g));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(pooled)), std::vector<double>{1.5},
                    K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that upsampling copies each pixel over its whole block. */
TEST_F(Medium_Forward, Upsample2dReplicatesEachPixelOverItsBlock)
{
    // One pixel at scale 2x2 becomes a 2x2 block of the same value.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 1, 1, 1}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID up, builder.upsample2d(x, 2, 2));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{5.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(up)), std::vector<double>{5.0, 5.0, 5.0, 5.0},
                    K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that the spatial mean averages the whole plane of each channel. */
TEST_F(Medium_Forward, SpatialMeanAveragesEveryHWPlane)
{
    // N=1, H=2, W=2, C=1, values 0,1,2,3: the mean of the whole plane is 1.5.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 2, 2, 1}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID mean, builder.spatial_mean(x));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(mean)), std::vector<double>{1.5},
                    K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that concatenating channels places each operand at its own offset. */
TEST_F(Medium_Forward, ConcatChannelsPlacesEachOperandAtItsOwnOffset)
{
    // Left [10] and right [20] must land as [10, 20].
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID left = builder.input(Shape{1, 1, 1, 1}, "left");
    const ValueID right = builder.input(Shape{1, 1, 1, 1}, "right");
    const std::array<ValueID, 2> parts = {left, right};
    SA_ASSERT_OK_AND_ASSIGN(const ValueID cat,
                            builder.concat_channels(
                                SushiRuntime::span<const ValueID>(parts.data(), parts.size())));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(left), std::vector<double>{10.0});
    write_tensor(run.pool.tensor(right), std::vector<double>{20.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(cat)), std::vector<double>{10.0, 20.0},
                    K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that slicing channels returns the chosen range unchanged. */
TEST_F(Medium_Forward, SliceChannelsExtractsTheMiddleRangeUnchanged)
{
    // Slicing [1, 3) of [1, 2, 3, 4] must return [2, 3].
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 1, 1, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sliced, builder.slice_channels(x, 1, 2));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1.0, 2.0, 3.0, 4.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(sliced)), std::vector<double>{2.0, 3.0},
                    K_DOUBLE_LOOSE_TOLERANCE);
}

/** @brief Tests that channel_affine scales and shifts per channel, not per row. */
TEST_F(Medium_Forward, ChannelAffineScalesAndShiftsPerChannelNotPerRow)
{
    // Two pixels share one pair of parameters; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 1, 2, 2}, "x");
    const ValueID gamma = builder.input(Shape{2}, "gamma");
    const ValueID beta = builder.input(Shape{2}, "beta");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.channel_affine(x, gamma, beta));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(gamma), std::vector<double>{2.0, 3.0});
    write_tensor(run.pool.tensor(beta), std::vector<double>{10.0, 20.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    // [1*2+10, 2*3+20, 3*2+10, 4*3+20]
    expect_elements(read_tensor(run.pool.tensor(y)),
                    std::vector<double>{12.0, 26.0, 16.0, 32.0}, K_FLOAT_TIGHT_TOLERANCE);
}

/** @brief Tests that batchnorm2d normalises each channel over the whole batch. */
TEST_F(Medium_Forward, BatchNorm2dNormalisesEachChannelOverTheWholeBatch)
{
    // Variance is biased, divided by N*H*W; K_EPSILON is the layer's argument, not a tolerance.
    constexpr double K_EPSILON = 1e-5;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2, 1, 2}, "x");
    const ValueID gamma = builder.input(Shape{2}, "gamma");
    const ValueID beta = builder.input(Shape{2}, "beta");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::BatchNormResult bn,
                            builder.batchnorm2d(x, gamma, beta, K_EPSILON));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 10, 2, 20, 3, 30, 4, 40});
    write_tensor(run.pool.tensor(gamma), std::vector<double>{2.0, 1.0});
    write_tensor(run.pool.tensor(beta), std::vector<double>{0.0, 5.0});

    ASSERT_NO_FATAL_FAILURE(run.compile());
    SA_EXPECT_OK(run.step->run());

    expect_elements(read_tensor(run.pool.tensor(bn.saved_mean)), std::vector<double>{2.5, 25.0},
                    K_FLOAT_TIGHT_TOLERANCE);

    const double inv0 = 1.0 / std::sqrt(1.25 + K_EPSILON);
    const double inv1 = 1.0 / std::sqrt(125.0 + K_EPSILON);

    expect_elements(read_tensor(run.pool.tensor(bn.saved_inv_std)),
                    std::vector<double>{inv0, inv1}, K_FLOAT_TIGHT_TOLERANCE);

    // Elements alternate: channel 0 (gamma 2, beta 0), then channel 1 (gamma 1, beta 5).
    const std::vector<double> expected = {
        2.0 * (1.0 - 2.5) * inv0,  (10.0 - 25.0) * inv1 + 5.0,
        2.0 * (2.0 - 2.5) * inv0,  (20.0 - 25.0) * inv1 + 5.0,
        2.0 * (3.0 - 2.5) * inv0,  (30.0 - 25.0) * inv1 + 5.0,
        2.0 * (4.0 - 2.5) * inv0,  (40.0 - 25.0) * inv1 + 5.0,
    };
    expect_elements(read_tensor(run.pool.tensor(bn.output)), expected, K_FLOAT_TOLERANCE);
}

/** @brief Tests that compiling a graph with an unbound leaf stops the process with its name. */
TEST_F(Medium_ForwardDeathTest, AnUnboundLeafIsRefusedRatherThanFilledWithGarbage)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2}, "x");
    SA_EXPECT_OK(builder.relu(x));

    TensorPool pool(*engine);
    SA_EXPECT_FATAL((void)CompiledStep::compile(*engine, graph, pool),
                    "Leaf value", "'x'", "has no tensor bound");
}

/** @brief Tests that compiling a node with no lowering row stops the process naming it. */
TEST_F(Medium_ForwardDeathTest, AnUnimplementedOperationNamesItself)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2}, "x");
    const ValueID out = graph.add_value(Shape{2, 2}, DType::Float32,
                                        Graph::Lifetime::Activation, "mystery");
    graph.add_node(OpID(0xABCD'1234u), {x}, {out});

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));

    SA_EXPECT_FATAL((void)CompiledStep::compile(*engine, graph, pool),
                    "No lowering for operation");
}
