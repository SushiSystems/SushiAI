/**************************************************************************/
/* test_host_evaluator.cpp                                                */
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
 * @file test_host_evaluator.cpp
 * @brief Tests the host rules, the leaf contract and the refusals of HostEvaluator.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::HostEvaluator;
using Graph::ValueID;

namespace
{
    /**
     * @brief Holds the rounded operations on the longest path of a GELU, an exponential or a
     *        logarithm of the libm the rules call: the cube, the sum, two products, the tanh,
     *        the sum and two products, each within one unit roundoff.
     */
    constexpr std::size_t K_TRANSCENDENTAL_OPERATIONS = 8;

    /** @brief Holds one product case: the stored operands, their shapes and the flags. */
    struct ProductRow
    {
        std::string name;              /**< The row's name. */
        Shape a_shape;                 /**< The stored shape of A. */
        std::vector<double> a;         /**< A as stored. */
        Shape b_shape;                 /**< The stored shape of B. */
        std::vector<double> b;         /**< B as stored. */
        bool transpose_a = false;      /**< Whether A is read transposed. */
        bool transpose_b = false;      /**< Whether B is read transposed. */
        std::vector<double> expected;  /**< The hand-worked product. */
    };

    /**
     * @brief Lists A = [[1, 2, 3], [4, 5, 6]] times B = [[1, 0], [0, 1], [1, 1]], which is
     *        [[4, 5], [10, 11]], stored plainly, transposed, and as a batch of the two.
     */
    std::vector<ProductRow> product_rows()
    {
        const std::vector<double> a{1, 2, 3, 4, 5, 6};
        const std::vector<double> a_t{1, 4, 2, 5, 3, 6};
        const std::vector<double> b{1, 0, 0, 1, 1, 1};
        const std::vector<double> b_t{1, 0, 1, 0, 1, 1};
        return {
            {"Plain", Shape{2, 3}, a, Shape{3, 2}, b, false, false, {4, 5, 10, 11}},
            {"TransposedA", Shape{3, 2}, a_t, Shape{3, 2}, b, true, false, {4, 5, 10, 11}},
            {"TransposedB", Shape{2, 3}, a, Shape{2, 3}, b_t, false, true, {4, 5, 10, 11}},
            {"Batched", Shape{2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, Shape{2, 3, 2},
             {1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0}, false, false, {4, 5, 10, 11, 0, 0, 0, 0}},
        };
    }

    /** @brief Runs one product row. */
    class Small_HostEvaluatorProduct : public ::testing::TestWithParam<ProductRow>
    {

    };

    /** @brief Holds one binary case: the operation and its result over a fixed pair. */
    struct BinaryRow
    {
        std::string name;              /**< The row's name. */
        OpID op;                       /**< The operation. */
        std::vector<double> expected;  /**< The result over a = {1, 4, -2}, b = {3, 4, 5}. */
    };

    /** @brief Runs one binary row. */
    class Small_HostEvaluatorBinary : public ::testing::TestWithParam<BinaryRow>
    {

    };

    /** @brief Holds one unary case over the operand {1, -2}. */
    struct UnaryRow
    {
        std::string name;              /**< The row's name. */
        Graph::Traced (*emit)(const GraphBuilder&, ValueID); /**< Emits the node. */
        std::vector<double> expected;  /**< The specified result. */
    };

    /**
     * @brief Lists the unary rules with their hand-derived values over {1, -2}: exp(v) is e and
     *        0.1353352832366127, the tanh GELU is 0.8411919906082768 and -0.04540230591222494,
     *        v / (1 + exp(-v)) is 0.7310585786300049 and -0.2384058440442351, max(v, 0) is 1 and
     *        0, |v| is 1 and 2, and log(1 + exp(v)) is 1.3132616875182228 and 0.1269280110429725.
     */
    std::vector<UnaryRow> unary_rows()
    {
        return {
            {"Scale", [](const GraphBuilder& b, ValueID x) { return b.scale(x, -0.5); },
             {-0.5, 1.0}},
            {"Exp", [](const GraphBuilder& b, ValueID x) { return b.exp(x); },
             {2.718281828459045, 0.1353352832366127}},
            {"Gelu", [](const GraphBuilder& b, ValueID x) { return b.gelu(x); },
             {0.8411919906082768, -0.04540230591222494}},
            {"Silu", [](const GraphBuilder& b, ValueID x) { return b.silu(x); },
             {0.7310585786300049, -0.2384058440442351}},
            {"Relu", [](const GraphBuilder& b, ValueID x) { return b.relu(x); }, {1.0, 0.0}},
            {"Abs", [](const GraphBuilder& b, ValueID x) { return b.abs(x); }, {1.0, 2.0}},
            {"Softplus", [](const GraphBuilder& b, ValueID x) { return b.softplus(x); },
             {1.3132616875182228, 0.1269280110429725}},
        };
    }

    /** @brief Runs one unary row. */
    class Small_HostEvaluatorUnary : public ::testing::TestWithParam<UnaryRow>
    {

    };

    /** @brief Holds one negative log-likelihood case: the operands and the hand-worked loss. */
    struct NllRow
    {
        std::string name;                 /**< The row's name. */
        Shape shape;                      /**< The shape of both operands. */
        std::vector<double> log_probs;    /**< The log-probabilities. */
        std::vector<double> targets;      /**< The targets. */
        double expected;                  /**< The mean negative log-likelihood. */
    };

    /**
     * @brief Lists the loss cases: two one-hot rows pick -2 and -3, so the loss is 5 / 2; one
     *        soft row {0.25, 0.75} over {-1, -3} is 0.25 + 2.25; three one-hot rows pick -2, -4
     *        and -6, so the loss is 12 / 3.
     */
    std::vector<NllRow> nll_rows()
    {
        return {
            {"OneHotRows", Shape{2, 2}, {-1, -2, -3, -4}, {0, 1, 1, 0}, 2.5},
            {"SoftRow", Shape{1, 2}, {-1, -3}, {0.25, 0.75}, 2.5},
            {"ThreeRows", Shape{3, 2}, {-1, -2, -3, -4, -5, -6}, {0, 1, 0, 1, 0, 1}, 4.0},
        };
    }

    /** @brief Runs one negative log-likelihood row. */
    class Small_HostEvaluatorNll : public ::testing::TestWithParam<NllRow>
    {

    };

    /** @brief Holds one operation and whether the evaluator has a rule for it. */
    struct CoverageRow
    {
        std::string name; /**< The row's name. */
        OpID op;          /**< The operation. */
        bool evaluated;   /**< Whether host_evaluates() accepts it. */
    };

    /** @brief Runs one coverage row. */
    class Small_HostEvaluatorCoverage : public ::testing::TestWithParam<CoverageRow>
    {

    };

    /** @brief Expects each of @p result to sit within the rounding bound of its @p expected. */
    void expect_each_within(const std::vector<double>& result,
                            const std::vector<double>& expected)
    {
        ASSERT_EQ(result.size(), expected.size());
        for (std::size_t at = 0; at < expected.size(); ++at)
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
                result[at], expected[at], K_TRANSCENDENTAL_OPERATIONS)))
                << "element " << at;
    }

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a product reads each operand as stored, transposed or batched. */
TEST_P(Small_HostEvaluatorProduct, AProductMatchesItsHandValues)
{
    const ProductRow& row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(row.a_shape, "a");
    const ValueID b = builder.input(row.b_shape, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID c,
                            builder.matmul(a, b, row.transpose_a, row.transpose_b));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(a, row.a));
    SA_EXPECT_OK(evaluator.write(b, row.b));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> product, evaluator.read(c));
    EXPECT_EQ(product, row.expected);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_HostEvaluatorProduct, ::testing::ValuesIn(product_rows()),
                         row_name<ProductRow>);

/** @brief Tests that each binary rule applies its arithmetic element by element. */
TEST_P(Small_HostEvaluatorBinary, ABinaryRuleCombinesEachPair)
{
    const BinaryRow& row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(Shape{3}, "a");
    const ValueID b = builder.input(Shape{3}, "b");
    const ValueID y = builder.declare_like(a, Shape{3}, DType::Float32, Graph::Lifetime::Activation,
                                           "y");
    graph.add_node(row.op, {a, b}, {y});
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(a, {1, 4, -2}));
    SA_EXPECT_OK(evaluator.write(b, {3, 4, 5}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, row.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_HostEvaluatorBinary,
    ::testing::Values(BinaryRow{"Add", Ops::K_ADD.id, {4, 8, 3}},
                      BinaryRow{"Sub", Ops::K_SUB.id, {-2, 0, -7}},
                      BinaryRow{"Mul", Ops::K_MUL.id, {3, 16, -10}},
                      BinaryRow{"LessEqual", Ops::K_LESS_EQUAL.id, {1, 1, 1}}),
    row_name<BinaryRow>);

/** @brief Tests that each unary rule writes its specified value of each element. */
TEST_P(Small_HostEvaluatorUnary, AUnaryRuleMapsEachElement)
{
    const UnaryRow& row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, row.emit(builder, x));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, -2}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    ASSERT_EQ(result.size(), 2u);
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        result[0], row.expected[0], K_TRANSCENDENTAL_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        result[1], row.expected[1], K_TRANSCENDENTAL_OPERATIONS)));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_HostEvaluatorUnary, ::testing::ValuesIn(unary_rows()),
                         row_name<UnaryRow>);

/** @brief Tests that the loss is minus the mean over the rows of the picked log-probabilities. */
TEST_P(Small_HostEvaluatorNll, TheLossIsTheMeanNegativeLogLikelihood)
{
    const NllRow& row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID log_probs = builder.input(row.shape, "log_probs");
    const ValueID targets = builder.input(row.shape, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss, builder.nll_loss(log_probs, targets));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(log_probs, row.log_probs));
    SA_EXPECT_OK(evaluator.write(targets, row.targets));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(loss));
    EXPECT_EQ(result, (std::vector<double>{row.expected}));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_HostEvaluatorNll, ::testing::ValuesIn(nll_rows()),
                         row_name<NllRow>);

/** @brief Tests that host_evaluates() accepts a leaf and the ruled operations alone. */
TEST_P(Small_HostEvaluatorCoverage, HostEvaluatesNamesTheRuledOperations)
{
    const CoverageRow& row = GetParam();

    EXPECT_EQ(Graph::host_evaluates(row.op), row.evaluated);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_HostEvaluatorCoverage,
    ::testing::Values(CoverageRow{"Leaf", Ops::K_LEAF.id, true},
                      CoverageRow{"Matmul", Ops::K_MATMUL.id, true},
                      CoverageRow{"LayerNorm", Ops::K_LAYER_NORM.id, true},
                      CoverageRow{"Silu", Ops::K_SILU.id, true},
                      CoverageRow{"Im2col", Ops::K_IM2COL.id, true},
                      CoverageRow{"DepthwiseConv2d", Ops::K_DEPTHWISE_CONV2D.id, true},
                      CoverageRow{"BatchNorm2d", Ops::K_BATCHNORM2D.id, true},
                      CoverageRow{"BatchNormFold", Ops::K_BATCHNORM_FOLD.id, true},
                      CoverageRow{"Relu", Ops::K_RELU.id, true},
                      CoverageRow{"NllLoss", Ops::K_NLL_LOSS.id, true},
                      CoverageRow{"Abs", Ops::K_ABS.id, true},
                      CoverageRow{"Softplus", Ops::K_SOFTPLUS.id, true},
                      CoverageRow{"Atan", Ops::K_ATAN.id, false},
                      CoverageRow{"ScatterAddRows", Ops::K_SCATTER_ADD_ROWS.id, true},
                      CoverageRow{"CausalMask", Ops::K_CAUSAL_MASK.id, true}),
    row_name<CoverageRow>);

/** @brief Tests that a reshape keeps the elements of its operand in their order. */
TEST(Small_HostEvaluator, AReshapeKeepsTheElementsInOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.reshape(x, Shape{3, 2}));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2, 3, 4, 5, 6}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, (std::vector<double>{1, 2, 3, 4, 5, 6}));
}

/** @brief Tests that a bias is added to every row. */
TEST(Small_HostEvaluator, ABiasReachesEveryRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    const ValueID bias = builder.input(Shape{3}, "bias");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID biased, builder.bias_add(x, bias));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2, 3, 4, 5, 6}));
    SA_EXPECT_OK(evaluator.write(bias, {10, 20, 30}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> rows, evaluator.read(biased));
    EXPECT_EQ(rows, (std::vector<double>{11, 22, 33, 14, 25, 36}));
}

/** @brief Tests that the column sums add every row. */
TEST(Small_HostEvaluator, TheColumnSumsAddEveryRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sums, builder.sum_rows(x));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2, 3, 4, 5, 6}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> columns, evaluator.read(sums));
    EXPECT_EQ(columns, (std::vector<double>{5, 7, 9}));
}

/** @brief Tests that a selection takes the second operand where the mask is set. */
TEST(Small_HostEvaluator, ASelectionFollowsItsCondition)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{3}, "condition");
    const ValueID a = builder.input(Shape{3}, "a");
    const ValueID b = builder.input(Shape{3}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.select(condition, a, b));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(condition, {1, 0, 1}));
    SA_EXPECT_OK(evaluator.write(a, {1, 2, 3}));
    SA_EXPECT_OK(evaluator.write(b, {7, 8, 9}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, (std::vector<double>{1, 8, 3}));
}

/** @brief Tests that a cast to float32 rounds each element to float. */
TEST(Small_HostEvaluator, ACastToFloat32RoundsToFloat)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2}, "x", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rounded, builder.cast(x, DType::Float32));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {0.1, 0.5}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> casts, evaluator.read(rounded));
    EXPECT_EQ(casts, (std::vector<double>{static_cast<double>(0.1f), 0.5}));
}

/** @brief Tests that a constant writes its value into every element. */
TEST(Small_HostEvaluator, AConstantFillsEveryElement)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID filled, builder.constant(Shape{2}, -1.5));
    HostEvaluator evaluator(graph);

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> fills, evaluator.read(filled));
    EXPECT_EQ(fills, (std::vector<double>{-1.5, -1.5}));
}

/**
 * @brief Tests that a log-softmax of the row {0, ln 3} is {ln 1/4, ln 3/4}, that is
 *        {-1.3862943611198906, -0.2876820724517809}.
 */
TEST(Small_HostEvaluator, ALogSoftmaxNormalisesEachRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.log_softmax(x));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {0.0, 1.0986122886681098}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    ASSERT_EQ(result.size(), 2u);
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        result[0], -1.3862943611198906, K_TRANSCENDENTAL_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        result[1], -0.2876820724517809, K_TRANSCENDENTAL_OPERATIONS)));
}

/**
 * @brief Tests that a layer norm of the row {1, 3} with epsilon 0.25, gain {2, 1} and bias
 *        {0, 1} saves the mean 2 and 1 / sqrt(1.25) = 0.8944271909999159, and writes
 *        {-2 / sqrt(1.25), 1 / sqrt(1.25) + 1}.
 */
TEST(Small_HostEvaluator, ALayerNormSavesItsStatisticsAndNormalisesTheRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 2}, "x");
    const ValueID gamma = builder.input(Shape{2}, "gamma");
    const ValueID beta = builder.input(Shape{2}, "beta");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::LayerNormResult norm,
                            builder.layer_norm(x, gamma, beta, 0.25));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 3}));
    SA_EXPECT_OK(evaluator.write(gamma, {2, 1}));
    SA_EXPECT_OK(evaluator.write(beta, {0, 1}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> y, evaluator.read(norm.output));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> mean, evaluator.read(norm.saved_mean));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> inv_std,
                            evaluator.read(norm.saved_inv_std));
    EXPECT_EQ(mean, (std::vector<double>{2}));
    ASSERT_EQ(inv_std.size(), 1u);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        inv_std[0], 0.8944271909999159, K_TRANSCENDENTAL_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        y[0], -1.7888543819998317, K_TRANSCENDENTAL_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        y[1], 1.8944271909999159, K_TRANSCENDENTAL_OPERATIONS)));
}

/** @brief Tests that a gather copies the rows its indices name, in their order. */
TEST(Small_HostEvaluator, AGatherCopiesTheNamedRows)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{3, 2}, "table");
    const ValueID indices = builder.input(Shape{2}, "indices", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows, builder.gather_rows(table, indices));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(table, {1, 2, 3, 4, 5, 6}));
    SA_EXPECT_OK(evaluator.write(indices, {2, 0}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(rows));
    EXPECT_EQ(result, (std::vector<double>{5, 6, 1, 2}));
}

/** @brief Tests that a gather of a row past the table is refused by run(). */
TEST(Small_HostEvaluator, AGatherPastTheTableIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{3, 2}, "table");
    const ValueID indices = builder.input(Shape{1}, "indices", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows, builder.gather_rows(table, indices));
    static_cast<void>(rows);
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(table, {1, 2, 3, 4, 5, 6}));
    SA_EXPECT_OK(evaluator.write(indices, {3}));

    SA_EXPECT_FAILURE(evaluator.run(), StatusCode::InvalidDescription, "row 3", "3 rows");
}

/** @brief Tests that a scatter adds each row to the row its index names, in index order. */
TEST(Small_HostEvaluator, AScatterAddsEachRowToTheRowItsIndexNames)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{3, 2}, "rows");
    const ValueID indices = builder.input(Shape{3}, "indices", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sums, builder.scatter_add_rows(rows, indices, 3));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(rows, {1, 2, 3, 4, 5, 6}));
    SA_EXPECT_OK(evaluator.write(indices, {2, 0, 2}));

    SA_EXPECT_OK(evaluator.run());

    // Row 0 takes {3, 4}, row 1 stays zero, row 2 takes {1, 2} + {5, 6} = {6, 8}.
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(sums));
    EXPECT_EQ(result, (std::vector<double>{3, 4, 0, 0, 6, 8}));
}

/** @brief Tests that a scatter to a row past the result is refused by run(). */
TEST(Small_HostEvaluator, AScatterPastTheResultIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{1, 2}, "rows");
    const ValueID indices = builder.input(Shape{1}, "indices", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sums, builder.scatter_add_rows(rows, indices, 3));
    static_cast<void>(sums);
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(rows, {1, 2}));
    SA_EXPECT_OK(evaluator.write(indices, {3}));

    SA_EXPECT_FAILURE(evaluator.run(), StatusCode::InvalidDescription, "row 3", "3 rows");
}

/** @brief Tests that a causal mask replaces every entry above the diagonal of each matrix. */
TEST(Small_HostEvaluator, ACausalMaskFillsTheEntriesAboveTheDiagonal)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.input(Shape{2, 3, 3}, "scores");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID masked, builder.causal_mask(scores, -9.0));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(scores, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                          11, 12, 13, 14, 15, 16, 17, 18, 19}));

    SA_EXPECT_OK(evaluator.run());

    // Per matrix the entries (0,1), (0,2) and (1,2) lie above the diagonal; the rest stay.
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(masked));
    EXPECT_EQ(result, (std::vector<double>{1, -9, -9, 4, 5, -9, 7, 8, 9,
                                           11, -9, -9, 14, 15, -9, 17, 18, 19}));
}

/** @brief Tests that a channel slice keeps its range of the last axis. */
TEST(Small_HostEvaluator, AChannelSliceKeepsItsRange)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 1, 2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID tail, builder.slice_channels(x, 1, 2));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2, 3, 4, 5, 6}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> sliced, evaluator.read(tail));
    EXPECT_EQ(sliced, (std::vector<double>{2, 3, 5, 6}));
}

/** @brief Tests that a concat joins its operands along the last axis, in their order. */
TEST(Small_HostEvaluator, AConcatJoinsTheLastAxis)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID wide = builder.input(Shape{1, 1, 2, 2}, "wide");
    const ValueID narrow = builder.input(Shape{1, 1, 2, 1}, "narrow");
    const std::array<ValueID, 2> parts{wide, narrow};
    SA_ASSERT_OK_AND_ASSIGN(const ValueID joined,
                            builder.concat_channels(SushiRuntime::span<const ValueID>(parts)));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(wide, {2, 3, 5, 6}));
    SA_EXPECT_OK(evaluator.write(narrow, {1, 4}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(joined));
    EXPECT_EQ(result, (std::vector<double>{2, 3, 1, 5, 6, 4}));
}

/** @brief Tests that a channel affine scales and shifts each channel by its own pair. */
TEST(Small_HostEvaluator, AChannelAffineAppliesThePairOfEachChannel)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 1, 2, 2}, "x");
    const ValueID gamma = builder.input(Shape{2}, "gamma");
    const ValueID beta = builder.input(Shape{2}, "beta");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID scaled, builder.channel_affine(x, gamma, beta));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {2, 3, 5, 6}));
    SA_EXPECT_OK(evaluator.write(gamma, {10, -1}));
    SA_EXPECT_OK(evaluator.write(beta, {0, 100}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> affine, evaluator.read(scaled));
    EXPECT_EQ(affine, (std::vector<double>{20, 97, 50, 94}));
}

/** @brief Tests that a max pool writes each window's first maximum and its flat index. */
TEST(Small_HostEvaluator, AMaxPoolKeepsTheFirstMaximumOfEachWindow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 1, 4, 1}, "x");
    Graph::ConvGeometry window;
    window.kernel_w = 2;
    window.stride_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const Graph::PoolResult pooled, builder.maxpool2d(x, window));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {3, 3, 2, 7}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> maxima, evaluator.read(pooled.output));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> indices, evaluator.read(pooled.indices));
    EXPECT_EQ(maxima, (std::vector<double>{3, 7}));
    EXPECT_EQ(indices, (std::vector<double>{0, 3}));
}

/** @brief Tests that an upsample repeats each pixel by its scales. */
TEST(Small_HostEvaluator, AnUpsampleRepeatsEachPixel)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 1, 2, 1}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID repeated, builder.upsample2d(x, 1, 2));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {3, 7}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> spread, evaluator.read(repeated));
    EXPECT_EQ(spread, (std::vector<double>{3, 3, 7, 7}));
}

/**
 * @brief Tests that a 2 by 2 convolution (IM2COL, MATMUL, BIAS_ADD) reads each window's taps in
 *        the weight's order. Over x = 1..9 as 3 by 3, the filter {1, 1, 1, 1} sums each window
 *        to 12, 16, 24, 28, and {1, 0, 0, -1} gives x[r0, s0] - x[r1, s1] = -4; the biases are
 *        10 and 0.
 */
TEST(Small_HostEvaluator, AConvolutionSumsEachWindowOverItsFilter)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 3, 3, 1}, "x");
    const ValueID weight = builder.input(Shape{2, 2, 2, 1}, "weight");
    const ValueID bias = builder.input(Shape{2}, "bias");
    Graph::ConvGeometry window;
    window.kernel_h = 2;
    window.kernel_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.conv2d(x, weight, bias, window));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2, 3, 4, 5, 6, 7, 8, 9}));
    SA_EXPECT_OK(evaluator.write(weight, {1, 1, 1, 1, 1, 0, 0, -1}));
    SA_EXPECT_OK(evaluator.write(bias, {10, 0}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, (std::vector<double>{22, -4, 26, -4, 34, -4, 38, -4}));
}

/**
 * @brief Tests that a 3 by 3 window padded by one reads zero outside a 2 by 2 image. With
 *        x = {1, 2, 3, 4} and the taps 1..9, the output at (h, w) is the sum of
 *        taps[r][s] * x[h - 1 + r][w - 1 + s]: 77 at (0, 0), 67, 47 and 37 at (1, 1).
 */
TEST(Small_HostEvaluator, APaddedConvolutionReadsZeroOutsideTheImage)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 2, 2, 1}, "x");
    const ValueID weight = builder.input(Shape{1, 3, 3, 1}, "weight");
    const ValueID bias = builder.input(Shape{1}, "bias");
    Graph::ConvGeometry window;
    window.kernel_h = 3;
    window.kernel_w = 3;
    window.pad_top = 1;
    window.pad_left = 1;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.conv2d(x, weight, bias, window));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2, 3, 4}));
    SA_EXPECT_OK(evaluator.write(weight, {1, 2, 3, 4, 5, 6, 7, 8, 9}));
    SA_EXPECT_OK(evaluator.write(bias, {0}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, (std::vector<double>{77, 67, 47, 37}));
}

/**
 * @brief Tests that a depthwise convolution filters each channel by its own taps. Channel 0 of
 *        x is 1..9 as 3 by 3 and channel 1 is ten times that; the taps {1, 1, 1, 1} on channel 0
 *        give 12, 16, 24, 28 and {1, 0, 0, -1} on channel 1 give 10 * (-4) = -40 for each window.
 */
TEST(Small_HostEvaluator, ADepthwiseConvolutionFiltersEachChannelByItsOwnTaps)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 3, 3, 2}, "x");
    const ValueID taps = builder.input(Shape{2, 2, 2}, "taps");
    Graph::ConvGeometry window;
    window.kernel_h = 2;
    window.kernel_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.depthwise_conv2d(x, taps, window));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90}));
    SA_EXPECT_OK(evaluator.write(taps, {1, 1, 1, 0, 1, 0, 1, -1}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, (std::vector<double>{12, -40, 16, -40, 24, -40, 28, -40}));
}

/**
 * @brief Tests that a batch normalisation takes its statistics per channel over every pixel.
 *        Channel 0 is {1, 3, 5, 7}: mean 4, biased variance 5, and with epsilon 4 the inverse
 *        deviation is 1 / 3. Channel 1 is {2, 2, 2, 2}: mean 2, variance 0, inverse deviation
 *        1 / 2. With gamma {2, 1} and beta {0, 1} channel 0 becomes 2 (x - 4) / 3 and channel 1
 *        becomes 1.
 */
TEST(Small_HostEvaluator, ABatchNormSavesItsStatisticsAndNormalisesEachChannel)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{1, 2, 2, 2}, "x");
    const ValueID gamma = builder.input(Shape{2}, "gamma");
    const ValueID beta = builder.input(Shape{2}, "beta");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::BatchNormResult norm,
                            builder.batchnorm2d(x, gamma, beta, 4.0));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2, 3, 2, 5, 2, 7, 2}));
    SA_EXPECT_OK(evaluator.write(gamma, {2, 1}));
    SA_EXPECT_OK(evaluator.write(beta, {0, 1}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> y, evaluator.read(norm.output));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> mean, evaluator.read(norm.saved_mean));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> inv_std,
                            evaluator.read(norm.saved_inv_std));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> variance,
                            evaluator.read(norm.batch_variance));
    EXPECT_EQ(mean, (std::vector<double>{4, 2}));
    EXPECT_EQ(variance, (std::vector<double>{5, 0}));
    expect_each_within(inv_std, {1.0 / 3.0, 0.5});
    expect_each_within(y, {-2.0, 1.0, -2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0, 2.0, 1.0});
}

/**
 * @brief Tests that a fold gives scale = gamma / sqrt(variance + epsilon) and
 *        shift = beta - mean * scale. With epsilon 4, gamma {2, 1}, beta {1, 0}, mean {4, 2}
 *        and variance {5, 0}: the scale is {2 / 3, 1 / 2} and the shift {-5 / 3, -1}.
 */
TEST(Small_HostEvaluator, AFoldGivesTheScaleAndShiftOfEachChannel)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID gamma = builder.input(Shape{2}, "gamma");
    const ValueID beta = builder.input(Shape{2}, "beta");
    const ValueID mean = builder.input(Shape{2}, "mean");
    const ValueID variance = builder.input(Shape{2}, "variance");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::BatchNormFold folded,
                            builder.batchnorm_fold(gamma, beta, mean, variance, 4.0));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(gamma, {2, 1}));
    SA_EXPECT_OK(evaluator.write(beta, {1, 0}));
    SA_EXPECT_OK(evaluator.write(mean, {4, 2}));
    SA_EXPECT_OK(evaluator.write(variance, {5, 0}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> scale, evaluator.read(folded.scale));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> shift, evaluator.read(folded.shift));
    expect_each_within(scale, {2.0 / 3.0, 0.5});
    expect_each_within(shift, {-5.0 / 3.0, -1.0});
}

/** @brief Tests that run() names the first operation it has no rule for. */
TEST(Small_HostEvaluator, AnOperationWithoutARuleIsUnsupported)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.atan(x));
    static_cast<void>(y);
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, -1}));

    SA_EXPECT_FAILURE(evaluator.run(), StatusCode::Unsupported, "no rule",
                      Ops::name_of(Ops::K_ATAN.id));
}

/** @brief Tests that run() refuses a leaf that was never written. */
TEST(Small_HostEvaluator, ALeafLeftUnwrittenIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.exp(x));
    static_cast<void>(y);
    HostEvaluator evaluator(graph);

    SA_EXPECT_FAILURE(evaluator.run(), StatusCode::InvalidDescription, "'x'", "write it");
}

/** @brief Tests that read() refuses a value run() has not computed yet. */
TEST(Small_HostEvaluator, AReadBeforeRunIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.exp(x));
    HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, {1, 2}));

    SA_EXPECT_FAILURE(evaluator.read(y), StatusCode::InvalidDescription, "no elements");
}

/** @brief Tests that write() refuses elements that do not match the leaf's count. */
TEST(Small_HostEvaluator, AWriteOfAnotherCountIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2}, "x");
    HostEvaluator evaluator(graph);

    SA_EXPECT_FAILURE(evaluator.write(x, {1, 2, 3}), StatusCode::InvalidDescription,
                      "needs 2 elements", "got 3");
}

/** @brief Tests that write() refuses a value that a node other than LEAF computes. */
TEST(Small_HostEvaluator, AWriteOfAComputedValueIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.exp(x));
    HostEvaluator evaluator(graph);

    SA_EXPECT_FAILURE(evaluator.write(y, {1, 2}), StatusCode::InvalidDescription,
                      Ops::name_of(Ops::K_EXP.id), "computes it");
}
