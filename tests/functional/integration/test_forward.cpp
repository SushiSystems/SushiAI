/**************************************************************************/
/* test_forward.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Every lowered operation, against values worked out by hand. This is the
// tier that would catch a transposed gemm, a bias broadcast down the wrong
// axis, or a log-softmax whose stabilising shift did not cancel - all of
// which produce plausible numbers rather than errors.

#include <cmath>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// `Test` alone resolves to ::testing::Test inside a TEST_F body, where the
// fixture's base injects the name; the alias keeps the helpers reachable.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    class Integration_Forward : public SaTest::EngineTest {};

    /** @brief Lowers a graph, runs it once, and hands back the pool. */
    class OneShot
    {
        public:
            OneShot(SushiBLAS::Engine& engine, const Graph::Graph& graph)
                : pool(engine)
            {
                pool.materialise_leaves(graph);
                graph_ = &graph;
                engine_ = &engine;
            }

            /** @brief Compiles the graph; call after every leaf has been written. */
            void compile() { step = std::make_unique<CompiledStep>(*engine_, *graph_, pool); }

            TensorPool pool;
            std::unique_ptr<CompiledStep> step;

        private:
            const Graph::Graph* graph_;
            SushiBLAS::Engine* engine_;
    };
}

TEST_F(Integration_Forward, LinearLayerMatchesHandComputedValues)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 2}, "x");
    const ValueId w = builder.input(Shape{3, 2}, "w");
    const ValueId b = builder.input(Shape{3}, "b");

    const ValueId y = builder.bias_add(builder.matmul(x, w, false, true), b);
    const ValueId z = builder.relu(y);

    OneShot run(*engine, graph);

    // X = [[1, 2], [3, 4]], W = [[1, 0], [0, 1], [1, 1]], b = [-2, -0.5, 1]
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(w), std::vector<double>{1, 0, 0, 1, 1, 1});
    write_tensor(run.pool.tensor(b), std::vector<double>{-2.0, -0.5, 1.0});

    run.compile();
    run.step->run();

    // X * W^T = [[1, 2, 3], [3, 4, 7]], then + b.
    const std::vector<double> expected_pre = {-1.0, 1.5, 4.0, 1.0, 3.5, 8.0};
    const std::vector<double> pre = read_tensor(run.pool.tensor(y));
    for (std::size_t i = 0; i < expected_pre.size(); ++i)
        EXPECT_NEAR(pre[i], expected_pre[i], 1e-5) << "at " << i;

    const std::vector<double> expected_post = {0.0, 1.5, 4.0, 1.0, 3.5, 8.0};
    const std::vector<double> post = read_tensor(run.pool.tensor(z));
    for (std::size_t i = 0; i < expected_post.size(); ++i)
        EXPECT_NEAR(post[i], expected_post[i], 1e-5) << "at " << i;
}

TEST_F(Integration_Forward, SumRowsProducesColumnSums)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{3, 2}, "x");
    const ValueId sums = builder.sum_rows(x);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 4, 5, 6});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(sums));
    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 9.0, 1e-5);
    EXPECT_NEAR(result[1], 12.0, 1e-5);
}

TEST_F(Integration_Forward, ReshapeReinterpretsRowMajorLinearOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{3, 4}, "x");
    const ValueId y = builder.reshape(x, Shape{2, 6});

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(y));
    ASSERT_EQ(result.size(), 12u);
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], static_cast<double>(i), 1e-9) << "at " << i;
}

TEST_F(Integration_Forward, LayoutConvertPermutesChannelsToTheFront)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // N=1, H=1, W=2, C=3: small enough to hand-trace every element's move.
    const ValueId x = builder.input(Shape{1, 1, 2, 3}, "x");
    const ValueId y = builder.layout_convert(x, Graph::TensorLayout::NCHW);

    OneShot run(*engine, graph);

    // NHWC, channels innermost: (w=0,c=0..2) = 0,1,2; (w=1,c=0..2) = 3,4,5.
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3, 4, 5});

    run.compile();
    run.step->run();

    // NCHW, channels outermost of the spatial block: for each c, the two W
    // positions in order - (c=0,w=0..1)=0,3; (c=1,w=0..1)=1,4; (c=2,w=0..1)=2,5.
    const std::vector<double> expected = {0, 3, 1, 4, 2, 5};
    const std::vector<double> result = read_tensor(run.pool.tensor(y));
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-5) << "at " << i;
}

TEST_F(Integration_Forward, ElementwiseAndScaleMatchTheirDefinitions)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{2, 2}, "a");
    const ValueId b = builder.input(Shape{2, 2}, "b");

    const ValueId sum = builder.add(a, b);
    const ValueId difference = builder.sub(a, b);
    const ValueId product = builder.mul(a, b);
    const ValueId scaled = builder.scale(a, -2.5);
    const ValueId exponent = builder.exp(b);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(a), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(b), std::vector<double>{0.5, -1.0, 2.0, 0.0});

    run.compile();
    run.step->run();

    const std::vector<double> expect_sum = {1.5, 1.0, 5.0, 4.0};
    const std::vector<double> expect_difference = {0.5, 3.0, 1.0, 4.0};
    const std::vector<double> expect_product = {0.5, -2.0, 6.0, 0.0};
    const std::vector<double> expect_scaled = {-2.5, -5.0, -7.5, -10.0};

    const std::vector<double> got_sum = read_tensor(run.pool.tensor(sum));
    const std::vector<double> got_difference = read_tensor(run.pool.tensor(difference));
    const std::vector<double> got_product = read_tensor(run.pool.tensor(product));
    const std::vector<double> got_scaled = read_tensor(run.pool.tensor(scaled));
    const std::vector<double> got_exponent = read_tensor(run.pool.tensor(exponent));

    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(got_sum[i], expect_sum[i], 1e-5) << "add at " << i;
        EXPECT_NEAR(got_difference[i], expect_difference[i], 1e-5) << "sub at " << i;
        EXPECT_NEAR(got_product[i], expect_product[i], 1e-5) << "mul at " << i;
        EXPECT_NEAR(got_scaled[i], expect_scaled[i], 1e-5) << "scale at " << i;
    }

    EXPECT_NEAR(got_exponent[0], std::exp(0.5), 1e-5);
    EXPECT_NEAR(got_exponent[1], std::exp(-1.0), 1e-5);
    EXPECT_NEAR(got_exponent[2], std::exp(2.0), 1e-5);
    EXPECT_NEAR(got_exponent[3], 1.0, 1e-5);
}

TEST_F(Integration_Forward, ActivationsMatchTheirDefinitionsAndLeaveTheInputAlone)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 4}, "x");
    const ValueId r = builder.relu(x);
    const ValueId s = builder.sigmoid(x);
    const ValueId t = builder.tanh(x);

    OneShot run(*engine, graph);
    const std::vector<double> inputs = {-2.0, -0.5, 0.5, 2.0};
    write_tensor(run.pool.tensor(x), inputs);

    run.compile();
    run.step->run();

    const std::vector<double> relu = read_tensor(run.pool.tensor(r));
    const std::vector<double> sigmoid = read_tensor(run.pool.tensor(s));
    const std::vector<double> hyperbolic = read_tensor(run.pool.tensor(t));

    // Out-of-place semantics matter: the backward rules for relu and gelu read
    // the forward INPUT, so an in-place activation would silently hand them
    // the output instead.
    const std::vector<double> unchanged = read_tensor(run.pool.tensor(x));

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        EXPECT_NEAR(relu[i], std::max(0.0, inputs[i]), 1e-5) << "relu at " << i;
        EXPECT_NEAR(sigmoid[i], 1.0 / (1.0 + std::exp(-inputs[i])), 1e-5) << "sigmoid at " << i;
        EXPECT_NEAR(hyperbolic[i], std::tanh(inputs[i]), 1e-5) << "tanh at " << i;
        EXPECT_NEAR(unchanged[i], inputs[i], 1e-12) << "input mutated at " << i;
    }
}

TEST_F(Integration_Forward, LogSoftmaxRowsSumToOneAfterExponentiating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 3}, "x");
    const ValueId l = builder.log_softmax(x);

    OneShot run(*engine, graph);
    // The second row is offset by a large constant; log-softmax must be
    // invariant to it, which is what the max shift buys.
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 51, 52, 53});

    run.compile();
    run.step->run();

    const std::vector<double> l_values = read_tensor(run.pool.tensor(l));

    // log-sum-exp of [1, 2, 3] is 3 + log(1 + e^-1 + e^-2).
    const double lse = 3.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
    const std::vector<double> expected = {1.0 - lse, 2.0 - lse, 3.0 - lse,
                                          1.0 - lse, 2.0 - lse, 3.0 - lse};

    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(l_values[i], expected[i], 1e-4) << "at " << i;

    for (int row = 0; row < 2; ++row)
    {
        double mass = 0.0;
        for (int column = 0; column < 3; ++column)
            mass += std::exp(l_values[static_cast<std::size_t>(row * 3 + column)]);
        EXPECT_NEAR(mass, 1.0, 1e-4) << "row " << row << " is not normalised";
    }
}

TEST_F(Integration_Forward, CrossEntropyOnUniformLogitsIsExactlyLogC)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 5}, "x");
    const ValueId targets = builder.input(Shape{4, 5}, "targets");
    const ValueId loss = builder.nll_loss(builder.log_softmax(x), targets);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>(20, 0.0));

    std::vector<double> one_hot(20, 0.0);
    for (int row = 0; row < 4; ++row) one_hot[static_cast<std::size_t>(row * 5 + row)] = 1.0;
    write_tensor(run.pool.tensor(targets), one_hot);

    run.compile();
    run.step->run();

    EXPECT_NEAR(read_tensor(run.pool.tensor(loss))[0], std::log(5.0), 1e-5);
}

TEST_F(Integration_Forward, CrossEntropyMatchesAHandComputedSingleRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 3}, "x");
    const ValueId targets = builder.input(Shape{1, 3}, "targets");
    const ValueId loss = builder.nll_loss(builder.log_softmax(x), targets);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1.0, 2.0, 3.0});
    write_tensor(run.pool.tensor(targets), std::vector<double>{0.0, 0.0, 1.0});

    run.compile();
    run.step->run();

    const double lse = 3.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
    EXPECT_NEAR(read_tensor(run.pool.tensor(loss))[0], lse - 3.0, 1e-5);
}

TEST_F(Integration_Forward, AccumulateAddsInPlaceAndAliasesItsResult)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{4}, "a");
    const ValueId b = builder.input(Shape{4}, "b");
    const ValueId total = builder.accumulate(a, b);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(a), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(b), std::vector<double>{10, 20, 30, 40});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(total));
    const std::vector<double> accumulator = read_tensor(run.pool.tensor(a));

    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(result[i], accumulator[i], 1e-12) << "the result must alias the accumulator";
        EXPECT_NEAR(result[i], static_cast<double>(11 * (i + 1)), 1e-5) << "at " << i;
    }
}

TEST_F(Integration_Forward, ZeroFillsItsBufferExactlyOnce)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId zeros = builder.zeros(Shape{2, 3}, DType::FLOAT32);
    const ValueId bias = builder.input(Shape{3}, "bias");
    const ValueId broadcast = builder.bias_add(zeros, bias);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(bias), std::vector<double>{1.0, -2.0, 3.0});

    run.compile();
    run.step->run();
    run.step->run();

    // Replaying must not accumulate: the bias broadcast writes rather than
    // adds into its result, and the zero buffer is not re-filled per step.
    const std::vector<double> result = read_tensor(run.pool.tensor(broadcast));
    const std::vector<double> expected = {1.0, -2.0, 3.0, 1.0, -2.0, 3.0};

    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-5) << "at " << i;
}

TEST_F(Integration_Forward, AnUnboundLeafIsRefusedRatherThanFilledWithGarbage)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 2}, "x");
    (void)builder.relu(x);

    TensorPool pool(*engine);
    EXPECT_THROW(CompiledStep(*engine, graph, pool), Error);
}

TEST_F(Integration_Forward, AnUnimplementedOperationNamesItself)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 2}, "x");
    const ValueId out = graph.add_value(Shape{2, 2}, DType::FLOAT32,
                                        Graph::Lifetime::ACTIVATION, "mystery");
    graph.add_node(OpID(0xABCD'1234u), {x}, {out});

    TensorPool pool(*engine);
    pool.materialise_leaves(graph);

    EXPECT_THROW(CompiledStep(*engine, graph, pool), Error);
}
