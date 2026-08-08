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

#include <array>
#include <cmath>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
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
    const ValueId lr = builder.leaky_relu(x, 0.1);

    OneShot run(*engine, graph);
    const std::vector<double> inputs = {-2.0, -0.5, 0.5, 2.0};
    write_tensor(run.pool.tensor(x), inputs);

    run.compile();
    run.step->run();

    const std::vector<double> relu = read_tensor(run.pool.tensor(r));
    const std::vector<double> sigmoid = read_tensor(run.pool.tensor(s));
    const std::vector<double> hyperbolic = read_tensor(run.pool.tensor(t));
    const std::vector<double> leaky = read_tensor(run.pool.tensor(lr));

    // Out-of-place semantics matter: the backward rules for relu and gelu read
    // the forward INPUT, so an in-place activation would silently hand them
    // the output instead.
    const std::vector<double> unchanged = read_tensor(run.pool.tensor(x));

    // Hand-computed at slope 0.1, straddling zero so both branches of
    // leaky_relu run: -2 -> -0.2, -0.5 -> -0.05, 0.5 -> 0.5, 2 -> 2.
    const std::vector<double> expected_leaky = {-0.2, -0.05, 0.5, 2.0};

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        EXPECT_NEAR(relu[i], std::max(0.0, inputs[i]), 1e-5) << "relu at " << i;
        EXPECT_NEAR(sigmoid[i], 1.0 / (1.0 + std::exp(-inputs[i])), 1e-5) << "sigmoid at " << i;
        EXPECT_NEAR(hyperbolic[i], std::tanh(inputs[i]), 1e-5) << "tanh at " << i;
        EXPECT_NEAR(leaky[i], expected_leaky[i], 1e-5) << "leaky_relu at " << i;
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

TEST_F(Integration_Forward, Im2colGathersOneWindowInTapOrderChannelInnermost)
{
    // A gradcheck says the backward is consistent with whatever the forward
    // happens to compute; it says nothing about whether the forward computes
    // the right thing. This is the forward's own proof, worked out by hand.
    //
    // N=1, H=2, W=2, C=1 with a 2x2 kernel and no padding covers the whole
    // image in exactly one window, so the column matrix's single row is the
    // input read in tap order (r, s) with channel innermost - which, for one
    // channel, is simply row-major (h, w).
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 2, 2, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    const ValueId cols = builder.im2col(x, g);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(cols));
    const std::vector<double> expected = {0, 1, 2, 3};
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-9) << "at " << i;
}

TEST_F(Integration_Forward, Col2imSumsEveryWindowThatOverlapsAPixel)
{
    // The claim col2im makes and im2col_backward leans on: a pixel covered by
    // several windows receives the *sum* of their contributions, not the
    // last one written. N=1, H=3, W=3, C=1 with a 2x2 kernel and no padding
    // has four windows (origins (0,0), (0,1), (1,0), (1,1)); the corner
    // pixels sit under exactly one of them, the edge pixels under two, and
    // the centre pixel under all four. Feeding every column entry a 1 turns
    // the expected image into that overlap count, worked out by hand:
    //
    //   1 2 1
    //   2 4 2
    //   1 2 1
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId cols = builder.input(Shape{4, 4}, "cols");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    const ValueId image = builder.col2im(cols, Shape{1, 3, 3, 1}, g);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(cols), std::vector<double>(16, 1.0));

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(image));
    const std::vector<double> expected = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-9) << "at " << i;
}

TEST_F(Integration_Forward, AnAsymmetricallyPaddedIm2colUsesTheExtentsItWasGivenNotDerivedOnes)
{
    // The case a symmetric re-derivation gets wrong *silently*. A [1, 4, 5, 1]
    // input with a 3x3 window, stride 1, pad_top 1 and pad_left 1, padded
    // 2 below and 0 on the right, has
    //
    //     out_h = (4 + 1 + 2 - 3) + 1 = 5     out_w = (5 + 1 + 0 - 3) + 1 = 4
    //
    // Re-deriving those from the geometry alone assumes the padding is
    // symmetric and yields 4x5 instead - the transpose of the real window
    // grid. Both are 20 windows, so the row-count check agrees and both
    // window origins stay inside the padded input: nothing throws, the
    // numbers are just wrong. That is why the extents travel in the node's
    // attributes rather than being recomputed.
    //
    // The input is 0..19 in row-major (h, w) order, so a window with origin
    // (oh - 1, ow - 1) reads the pixels h*5 + w it covers and a zero
    // wherever the window hangs off the image. Row oh*4 + ow of the result
    // is that window in tap order. Worked out from that rule, the last row
    // (oh = 4, ow = 3, origin (3, 2)) covers only the input's bottom-right
    // three pixels, 17 18 19, and is padding everywhere else - whereas the
    // transposed grid would put 13 14 0 18 19 0 0 0 0 there.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 4, 5, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;
    g.pad_top = 1;  g.pad_left = 1;
    const ValueId cols = builder.im2col_with_extents(x, g, 5, 4);

    OneShot run(*engine, graph);
    std::vector<double> image(20);
    for (std::size_t i = 0; i < image.size(); ++i) image[i] = static_cast<double>(i);
    write_tensor(run.pool.tensor(x), image);

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(cols));
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
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-9)
            << "at row " << (i / 9) << " tap " << (i % 9);
}

TEST_F(Integration_Forward, MaxPool2dTakesEachWindowsMaximum)
{
    // A gradcheck says the backward is consistent with whatever the forward
    // happens to compute; this is the forward's own proof. N=1, H=2, W=2,
    // C=1, values 0,1,2,3 in row-major (h,w) order, one 2x2 window covering
    // the whole image: the maximum is 3.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 2, 2, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    const Graph::PoolResult pooled = builder.maxpool2d(x, g);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(pooled.output));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 3.0, 1e-9);
}

TEST_F(Integration_Forward, AMaxPoolSurvivesTheFullPassPipelineAndStillRoutesItsGradient)
{
    // MAXPOOL2D is the first operation here whose results are not all one
    // element type: beside the pooled tensor it writes an INT64 argmax, and
    // its backward reads that as a table of offsets. insert_casts() derived
    // one precision per node from outputs[0] and forced every operand and
    // result to it, so the argmax read as a value in the wrong type - and was
    // either retyped to float in place or handed to a cast that refuses
    // anything but real float to real float.
    //
    // Neither half showed up in a shape check, so this runs the passes in the
    // order Trainer::build runs them and then executes the result. A routing
    // table that has been through a float is not a routing table; the
    // gradient is where that becomes visible.
    //
    // Assembled by hand rather than through Trainer::build because no NN
    // layer emits a pool yet, so the trainer cannot reach this op today.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // N=1, H=2, W=2, C=2, values 0..7 in NHWC order, one 2x2 window. Per
    // channel the maximum is the last row: taps 6 (c=0, value 6) and 7 (c=1,
    // value 7). Every other tap must come back with no gradient at all.
    const ValueId x = builder.parameter(Shape{1, 2, 2, 2}, "x");
    const ValueId target = builder.input(Shape{1, 2}, "target");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;

    const Graph::PoolResult pooled = builder.maxpool2d(x, g);
    const ValueId logits = builder.spatial_mean(pooled.output);
    const ValueId loss = builder.nll_loss(builder.log_softmax(logits), target);
    graph.finalise();

    const std::array<ValueId, 1> wrt = {x};
    const std::vector<ValueId> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_EQ(graph.value(pooled.indices).dtype, DType::INT64);

    ASSERT_NO_THROW(Graph::insert_casts(graph, Graph::CastOptions{}));
    ASSERT_NO_THROW(Graph::fuse(graph, Graph::FusionOptions{}));

    EXPECT_EQ(graph.value(pooled.indices).dtype, DType::INT64)
        << "the pass retyped the index buffer, so the backward is routing gradients by "
           "whatever a float reinterpretation of an offset table happens to say";

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7});
    write_tensor(run.pool.tensor(target), std::vector<double>{1.0, 0.0});

    run.compile();
    run.step->run();

    const std::vector<double> pooled_values = read_tensor(run.pool.tensor(pooled.output));
    ASSERT_EQ(pooled_values.size(), 2u);
    EXPECT_NEAR(pooled_values[0], 6.0, 1e-9);
    EXPECT_NEAR(pooled_values[1], 7.0, 1e-9);

    const std::vector<double> dx = read_tensor(run.pool.tensor(gradients[0]));
    ASSERT_EQ(dx.size(), 8u);

    for (std::size_t i = 0; i < 6; ++i)
        EXPECT_NEAR(dx[i], 0.0, 1e-9)
            << "tap " << i << " did not win its window but was given a gradient";

    EXPECT_GT(std::abs(dx[6]), 1e-6) << "the c=0 maximum received no gradient at all";
    EXPECT_GT(std::abs(dx[7]), 1e-6) << "the c=1 maximum received no gradient at all";
}

TEST_F(Integration_Forward, AvgPool2dDividesByTheFullWindowArea)
{
    // Same window as the max-pool case above; the average of 0,1,2,3 is 1.5.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 2, 2, 1}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    const ValueId pooled = builder.avgpool2d(x, g);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(pooled));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 1.5, 1e-9);
}

TEST_F(Integration_Forward, Upsample2dReplicatesEachPixelOverItsBlock)
{
    // A single pixel, scale 2x2, replicated into a 2x2 block of the same
    // value.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 1, 1, 1}, "x");
    const ValueId up = builder.upsample2d(x, 2, 2);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{5.0});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(up));
    const std::vector<double> expected = {5.0, 5.0, 5.0, 5.0};
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-9) << "at " << i;
}

TEST_F(Integration_Forward, SpatialMeanAveragesEveryHWPlane)
{
    // N=1, H=2, W=2, C=1, values 0,1,2,3: the mean of the whole plane is 1.5.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 2, 2, 1}, "x");
    const ValueId mean = builder.spatial_mean(x);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{0, 1, 2, 3});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(mean));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 1.5, 1e-9);
}

TEST_F(Integration_Forward, ConcatChannelsPlacesEachOperandAtItsOwnOffset)
{
    // Two 1x1x1 pixels, one channel each: left [10], right [20]. Concatenated
    // in order they must land as [10, 20], not [20, 10] or summed together.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId left = builder.input(Shape{1, 1, 1, 1}, "left");
    const ValueId right = builder.input(Shape{1, 1, 1, 1}, "right");
    const std::array<ValueId, 2> parts = {left, right};
    const ValueId cat = builder.concat_channels(
        SushiRuntime::span<const ValueId>(parts.data(), parts.size()));

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(left), std::vector<double>{10.0});
    write_tensor(run.pool.tensor(right), std::vector<double>{20.0});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(cat));
    const std::vector<double> expected = {10.0, 20.0};
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-9) << "at " << i;
}

TEST_F(Integration_Forward, SliceChannelsExtractsTheMiddleRangeUnchanged)
{
    // N=1, H=1, W=1, four channels [1,2,3,4]. Slicing [1, 3) must return
    // [2, 3] exactly, not the whole input and not a shifted range.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 1, 1, 4}, "x");
    const ValueId sliced = builder.slice_channels(x, 1, 2);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1.0, 2.0, 3.0, 4.0});

    run.compile();
    run.step->run();

    const std::vector<double> result = read_tensor(run.pool.tensor(sliced));
    const std::vector<double> expected = {2.0, 3.0};
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-9) << "at " << i;
}

TEST_F(Integration_Forward, ChannelAffineScalesAndShiftsPerChannelNotPerRow)
{
    // N=1, H=1, W=2, C=2. Two pixels sharing one pair of parameters is what
    // separates a per-channel broadcast from a per-row one: BIAS_ADD's
    // broadcast would give the second pixel the second channel's numbers.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{1, 1, 2, 2}, "x");
    const ValueId gamma = builder.input(Shape{2}, "gamma");
    const ValueId beta = builder.input(Shape{2}, "beta");
    const ValueId y = builder.channel_affine(x, gamma, beta);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 2, 3, 4});
    write_tensor(run.pool.tensor(gamma), std::vector<double>{2.0, 3.0});
    write_tensor(run.pool.tensor(beta), std::vector<double>{10.0, 20.0});

    run.compile();
    run.step->run();

    // [1*2+10, 2*3+20, 3*2+10, 4*3+20]
    const std::vector<double> expected = {12.0, 26.0, 16.0, 32.0};
    const std::vector<double> result = read_tensor(run.pool.tensor(y));
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-6) << "at " << i;
}

TEST_F(Integration_Forward, BatchNorm2dNormalisesEachChannelOverTheWholeBatch)
{
    // N=2, H=2, W=1, C=2, so each channel is folded over four elements and the
    // two channels have deliberately different scales - a normalisation that
    // reduced over the wrong axis would still produce four finite numbers.
    //
    // Channel 0 sees 1, 2, 3, 4:    mean 2.5, biased variance 1.25.
    // Channel 1 sees 10, 20, 30, 40: mean 25, biased variance 125.
    //
    // Biased, dividing by N*H*W rather than by N*H*W - 1: that is what the
    // forward normalises by, and the saved statistics are what the backward
    // reads, so an unbiased divisor here would be a different function.
    constexpr double EPSILON = 1e-5;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 2, 1, 2}, "x");
    const ValueId gamma = builder.input(Shape{2}, "gamma");
    const ValueId beta = builder.input(Shape{2}, "beta");
    const Graph::BatchNormResult bn = builder.batchnorm2d(x, gamma, beta, EPSILON);

    OneShot run(*engine, graph);
    write_tensor(run.pool.tensor(x), std::vector<double>{1, 10, 2, 20, 3, 30, 4, 40});
    write_tensor(run.pool.tensor(gamma), std::vector<double>{2.0, 1.0});
    write_tensor(run.pool.tensor(beta), std::vector<double>{0.0, 5.0});

    run.compile();
    run.step->run();

    const std::vector<double> mean = read_tensor(run.pool.tensor(bn.saved_mean));
    ASSERT_EQ(mean.size(), 2u);
    EXPECT_NEAR(mean[0], 2.5, 1e-6);
    EXPECT_NEAR(mean[1], 25.0, 1e-6);

    const double inv0 = 1.0 / std::sqrt(1.25 + EPSILON);
    const double inv1 = 1.0 / std::sqrt(125.0 + EPSILON);

    const std::vector<double> inv_std = read_tensor(run.pool.tensor(bn.saved_inv_std));
    ASSERT_EQ(inv_std.size(), 2u);
    EXPECT_NEAR(inv_std[0], inv0, 1e-6);
    EXPECT_NEAR(inv_std[1], inv1, 1e-6);

    const std::array<double, 4> channel0 = {1.0, 2.0, 3.0, 4.0};
    const std::array<double, 4> channel1 = {10.0, 20.0, 30.0, 40.0};

    std::vector<double> expected(8);
    for (std::size_t i = 0; i < 4; ++i)
    {
        expected[i * 2]     = 2.0 * (channel0[i] - 2.5) * inv0 + 0.0;
        expected[i * 2 + 1] = 1.0 * (channel1[i] - 25.0) * inv1 + 5.0;
    }

    const std::vector<double> result = read_tensor(run.pool.tensor(bn.output));
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
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
