/**************************************************************************/
/* test_rematerialise.cpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free. The IM2COL rematerialisation pass is an IR-to-IR rewrite, so
// what it produces is a graph that can be inspected node by node.
//
// This tier proves the *structure*: that a clone appears, that it reads the
// same operands and carries the same attributes as the node it copies, that it
// is inserted ahead of its consumers rather than after them, and that an
// inference graph is left alone. It cannot prove the numbers are unchanged -
// that is Integration_Rematerialise, which runs the same graph twice.
//
// The brief's three cases are all here, with two deviations forced by the
// library rather than chosen:
//
//   1. Autograd::differentiate() refuses any root that is not a scalar
//      produced by NLL_LOSS (see differentiator.cpp, which explains why every
//      other root would silently return empty gradients). The brief's graphs
//      stopped at a MATMUL, so each is carried on to a loss here. The shapes,
//      geometry and expectations are the brief's.
//   2. A "backward consumer" cannot be identified by node index, because the
//      pass runs when the backward nodes are already in the graph and there is
//      no marker separating the phases. The rule the pass actually applies is
//      "every consumer after the first", which is the same set for every graph
//      differentiation produces and is well-defined for one it did not.

#include <array>
#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/rematerialise.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueId;

namespace
{
    /** @brief The nodes of @p graph reading @p value, in graph order. */
    std::vector<Graph::NodeId> readers_of(const Graph::Graph& graph, ValueId value)
    {
        std::vector<Graph::NodeId> readers;
        for (std::size_t i = 0; i < graph.node_count(); ++i)
            for (ValueId in : graph.nodes()[i].inputs)
                if (in == value) { readers.push_back(static_cast<Graph::NodeId>(i)); break; }
        return readers;
    }

    /** @brief How many IM2COL nodes @p graph holds. */
    std::size_t im2col_count(const Graph::Graph& graph)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
            if (node.op == Ops::IM2COL.id) ++count;
        return count;
    }

    /**
     * @brief A convolution as a GEMM, carried on to a scalar NLL_LOSS root.
     *
     * The loss tail is what makes the graph differentiable at all; it is not
     * part of what is being tested.
     */
    struct ConvGraph
    {
        Graph::Graph graph;
        ValueId cols = Graph::NO_VALUE;
        ValueId weight = Graph::NO_VALUE;
        ValueId loss = Graph::NO_VALUE;
    };

    /** @brief Builds and differentiates a one-convolution graph. */
    void build_and_differentiate(ConvGraph& c, Shape image, Shape weight_shape)
    {
        const GraphBuilder builder(c.graph);
        const ValueId x = builder.input(image, "x");
        c.weight = builder.parameter(weight_shape, "w");

        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        c.cols = builder.im2col(x, g);
        const ValueId out = builder.matmul(c.cols, c.weight);

        const Shape logits = c.graph.value(out).shape;
        const ValueId target = builder.input(logits, "target");
        c.loss = builder.nll_loss(builder.log_softmax(out), target);
        c.graph.finalise();

        const std::array<ValueId, 1> wrt = {c.weight};
        (void)Autograd::differentiate(c.graph, c.loss,
            SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));
    }
}

TEST(Unit_Rematerialise, TheColumnBufferStopsBeingAnActivation)
{
    // The number CV-10 is about: ResNet-18's first block at batch 128 holds
    // 128*32*32*64*9*4 bytes = ~302 MB in this one buffer, and there are twenty
    // convolutions. Cloning the gather next to its backward consumer drops the
    // lifetime to SCRATCH, so one is live at a time instead of twenty.
    ConvGraph c;
    build_and_differentiate(c, Shape{2, 5, 5, 3}, Shape{27, 8});

    EXPECT_EQ(c.graph.values()[c.cols].lifetime, Graph::Lifetime::ACTIVATION);
    ASSERT_GT(readers_of(c.graph, c.cols).size(), std::size_t{1})
        << "the backward pass did not read the column matrix, so there is "
           "nothing for the pass to do and this test proves nothing";

    Graph::rematerialise_im2col(c.graph);

    EXPECT_EQ(c.graph.values()[c.cols].lifetime, Graph::Lifetime::SCRATCH);

    // And it really did die early: exactly one node still reads the original.
    EXPECT_EQ(readers_of(c.graph, c.cols).size(), std::size_t{1});
}

TEST(Unit_Rematerialise, TheCloneComputesTheSameThingAsTheOriginal)
{
    // A clone that read a different operand or carried different attributes
    // would produce a plausible column matrix and a wrong weight gradient. Both
    // are checked because the geometry lives in the attributes and is easy to
    // drop when copying a node.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});

    const std::size_t before = c.graph.node_count();
    ASSERT_EQ(im2col_count(c.graph), std::size_t{1});

    Graph::rematerialise_im2col(c.graph);

    EXPECT_GT(c.graph.node_count(), before) << "no clone was inserted";
    EXPECT_EQ(c.graph.node_count(), before + 1) << "one clone, shared by the later consumers";

    const Graph::NodeId original_id = c.graph.value(c.cols).producer;
    const Graph::Node& original = c.graph.node(original_id);

    std::size_t clones = 0;
    for (std::size_t i = 0; i < c.graph.node_count(); ++i)
    {
        const Graph::Node& n = c.graph.nodes()[i];
        if (n.op != Ops::IM2COL.id || i == original_id) continue;
        ++clones;
        EXPECT_EQ(n.inputs, original.inputs);
        for (std::size_t slot = 0; slot < Graph::Attributes::CAPACITY; ++slot)
            EXPECT_EQ(n.attributes.get(slot), original.attributes.get(slot))
                << "the clone lost attribute slot " << slot;

        // Same description, and the same layout: the layout is a graph-wide
        // invariant, and a clone that dropped it would fail finalise() on a
        // graph that is numerically correct.
        const Graph::Value& copy = c.graph.value(n.outputs.front());
        const Graph::Value& stored = c.graph.value(c.cols);
        EXPECT_EQ(copy.shape, stored.shape);
        EXPECT_EQ(copy.dtype, stored.dtype);
        EXPECT_EQ(copy.layout, stored.layout);
        EXPECT_EQ(copy.lifetime, Graph::Lifetime::SCRATCH);
    }
    EXPECT_EQ(clones, std::size_t{1});
}

TEST(Unit_Rematerialise, TheCloneIsInsertedAheadOfEveryConsumerItServes)
{
    // The failure this catches is not a wrong number, it is a graph whose
    // insertion order stopped being a topological order - a recomputation
    // appended after the GEMM that reads it. MemoryPlan::build() rests on that
    // order and says so by name, and a lowering that got past it would record a
    // GEMM against a buffer nothing had written yet.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});
    Graph::rematerialise_im2col(c.graph);

    for (std::size_t i = 0; i < c.graph.node_count(); ++i)
        for (ValueId in : c.graph.nodes()[i].inputs)
        {
            const Graph::NodeId producer = c.graph.value(in).producer;
            if (producer == Graph::NO_NODE) continue;
            EXPECT_LT(producer, i) << "node " << i << " reads value " << in
                                   << ", which node " << producer << " writes";
        }

    // The invariant, asserted by the component that depends on it.
    EXPECT_NO_THROW((void)Graph::MemoryPlan::build(c.graph));

    // And the graph's own proofs still hold after the rewrite.
    EXPECT_NO_THROW(c.graph.finalise());
}

TEST(Unit_Rematerialise, AnIm2colWithNoBackwardConsumerIsLeftAlone)
{
    // Inference: nothing was differentiated, so there is no second consumer and
    // cloning would be a free slowdown.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{1, 4, 4, 2}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;

    const ValueId cols = builder.im2col(x, g);
    graph.finalise();

    const std::size_t before = graph.node_count();
    Graph::rematerialise_im2col(graph);

    EXPECT_EQ(graph.node_count(), before);
    EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::ACTIVATION);
}

TEST(Unit_Rematerialise, AForwardOnlyConvolutionIsLeftAlone)
{
    // The other half of the inference case, and the one that would break if the
    // pass cloned for *every* consumer rather than every consumer after the
    // first: here the column matrix has exactly one reader, the GEMM.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{1, 4, 4, 2}, "x");
    const ValueId w = builder.input(Shape{18, 4}, "w");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;
    g.pad_top = 1;  g.pad_left = 1;

    const ValueId cols = builder.im2col(x, g);
    (void)builder.matmul(cols, w);
    graph.finalise();

    const std::size_t before = graph.node_count();
    Graph::rematerialise_im2col(graph);

    EXPECT_EQ(graph.node_count(), before);
    EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::ACTIVATION);
}

TEST(Unit_Rematerialise, RunningItTwiceAddsNothingTheSecondTime)
{
    // The clone has one consumer of its own once the rewiring is done, so the
    // pass has no reason to rematerialise its own output. If it did, the pass
    // would not terminate on a graph with three consumers.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});

    Graph::rematerialise_im2col(c.graph);
    const std::size_t once = c.graph.node_count();
    Graph::rematerialise_im2col(c.graph);

    EXPECT_EQ(c.graph.node_count(), once);
}
