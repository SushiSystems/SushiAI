/**************************************************************************/
/* test_ir.cpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the IR and its builder are host-lane, which is the whole point
// of keeping shape inference out of the lowering.

#include <gtest/gtest.h>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Lifetime;
using Graph::NO_NODE;
using Graph::ValueId;

TEST(Unit_Ir, AddValueRecordsShapeDtypeAndLifetime)
{
    Graph::Graph graph;
    const ValueId id = graph.add_value(Shape{2, 3}, DType::FLOAT64,
                                       Lifetime::PARAMETER, "w");

    EXPECT_EQ(graph.value_count(), 1u);
    EXPECT_EQ(graph.value(id).shape, Shape({2, 3}));
    EXPECT_EQ(graph.value(id).dtype, DType::FLOAT64);
    EXPECT_EQ(graph.value(id).lifetime, Lifetime::PARAMETER);
    EXPECT_EQ(graph.value(id).debug_name, "w");
    EXPECT_EQ(graph.value(id).producer, NO_NODE);
}

TEST(Unit_Ir, AddNodeClaimsItsOutputs)
{
    Graph::Graph graph;
    const ValueId a = graph.add_value(Shape{2}, DType::FLOAT32, Lifetime::INPUT);
    const ValueId b = graph.add_value(Shape{2}, DType::FLOAT32, Lifetime::INPUT);
    const ValueId c = graph.add_value(Shape{2}, DType::FLOAT32, Lifetime::ACTIVATION);

    const Graph::NodeId node = graph.add_node(Ops::ADD.id, {a, b}, {c});

    EXPECT_EQ(graph.value(c).producer, node);
    EXPECT_EQ(graph.node(node).inputs.size(), 2u);
}

TEST(Unit_Ir, ASecondProducerForOneValueThrows)
{
    // Single assignment is the property that lets differentiation skip alias
    // analysis entirely, so a double write is rejected rather than tolerated.
    Graph::Graph graph;
    const ValueId a = graph.add_value(Shape{2}, DType::FLOAT32, Lifetime::INPUT);
    const ValueId b = graph.add_value(Shape{2}, DType::FLOAT32, Lifetime::INPUT);
    const ValueId c = graph.add_value(Shape{2}, DType::FLOAT32, Lifetime::ACTIVATION);

    graph.add_node(Ops::ADD.id, {a, b}, {c});
    EXPECT_THROW(graph.add_node(Ops::SUB.id, {a, b}, {c}), Error);
}

TEST(Unit_Ir, ReferencingAnUnknownValueThrows)
{
    Graph::Graph graph;
    const ValueId a = graph.add_value(Shape{2}, DType::FLOAT32, Lifetime::INPUT);

    EXPECT_THROW(graph.add_node(Ops::ADD.id, {a, 99}, {a}), Error);
    EXPECT_THROW(graph.add_node(Ops::ADD.id, {a}, {99}), Error);
}

TEST(Unit_Ir, PropagateRequiresGradFlowsForwardThroughTheWholeChain)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId w = builder.parameter(Shape{5, 3}, "w");
    const ValueId y = builder.matmul(x, w, false, true);
    const ValueId z = builder.relu(y);

    EXPECT_FALSE(graph.value(y).requires_grad);
    EXPECT_FALSE(graph.value(z).requires_grad);

    graph.finalise();

    EXPECT_FALSE(graph.value(x).requires_grad);
    EXPECT_TRUE(graph.value(w).requires_grad);
    EXPECT_TRUE(graph.value(y).requires_grad);
    EXPECT_TRUE(graph.value(z).requires_grad);
}

TEST(Unit_Ir, PropagateRequiresGradLeavesAFrozenBranchAlone)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId frozen = builder.relu(x);
    const ValueId w = builder.parameter(Shape{3, 3}, "w");
    const ValueId live = builder.matmul(x, w, false, false);

    graph.finalise();

    EXPECT_FALSE(graph.value(frozen).requires_grad);
    EXPECT_TRUE(graph.value(live).requires_grad);
}

TEST(Unit_Ir, BuilderInfersMatmulShapesForEveryTranspositionCase)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{4, 3}, "a");    // [m, k]
    const ValueId at = builder.input(Shape{3, 4}, "at");  // [k, m]
    const ValueId b = builder.input(Shape{3, 5}, "b");    // [k, n]
    const ValueId bt = builder.input(Shape{5, 3}, "bt");  // [n, k]

    EXPECT_EQ(graph.value(builder.matmul(a, b, false, false)).shape, Shape({4, 5}));
    EXPECT_EQ(graph.value(builder.matmul(at, b, true, false)).shape, Shape({4, 5}));
    EXPECT_EQ(graph.value(builder.matmul(a, bt, false, true)).shape, Shape({4, 5}));
    EXPECT_EQ(graph.value(builder.matmul(at, bt, true, true)).shape, Shape({4, 5}));
}

TEST(Unit_Ir, BuilderRejectsAContractionMismatch)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{4, 3}, "a");
    const ValueId b = builder.input(Shape{7, 5}, "b");

    EXPECT_THROW((void)builder.matmul(a, b, false, false), Error);
}

TEST(Unit_Ir, BuilderChecksBiasAndLossShapes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 5}, "x");
    const ValueId good = builder.input(Shape{5}, "good");
    const ValueId bad = builder.input(Shape{4}, "bad");

    EXPECT_EQ(graph.value(builder.bias_add(x, good)).shape, Shape({4, 5}));
    EXPECT_THROW((void)builder.bias_add(x, bad), Error);

    EXPECT_EQ(graph.value(builder.sum_rows(x)).shape, Shape({5}));

    const ValueId targets = builder.input(Shape{4, 5}, "targets");
    EXPECT_EQ(graph.value(builder.nll_loss(x, targets)).shape, Shape({1}));

    const ValueId wrong = builder.input(Shape{4, 6}, "wrong");
    EXPECT_THROW((void)builder.nll_loss(x, wrong), Error);
}

TEST(Unit_Ir, BuilderRejectsMixedElementTypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId single = builder.input(Shape{4, 4}, "f32", DType::FLOAT32);
    const ValueId doubled = builder.input(Shape{4, 4}, "f64", DType::FLOAT64);

    EXPECT_THROW((void)builder.add(single, doubled), Error);
    EXPECT_THROW((void)builder.matmul(single, doubled, false, false), Error);
}

TEST(Unit_Ir, NodesAreEmittedInTopologicalOrder)
{
    // Insertion order is relied on as a topological order by the lowering and
    // (reversed) by differentiation, so it is asserted rather than assumed.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 2}, "x");
    const ValueId y = builder.relu(x);
    const ValueId z = builder.tanh(y);
    (void)z;

    for (std::size_t i = 0; i < graph.node_count(); ++i)
        for (Graph::ValueId in : graph.nodes()[i].inputs)
        {
            const Graph::NodeId producer = graph.value(in).producer;
            if (producer != NO_NODE) EXPECT_LT(producer, i);
        }
}

TEST(Unit_IR, AttributesHoldAConvolutionsGeometryAndItsWindowGrid)
{
    // Ten, not eight. CV-5's eight are the geometry: kernel_h/w, stride_h/w,
    // dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents
    // out_h and out_w, and they have to be carried rather than re-derived:
    // deriving them from the geometry assumes symmetric padding, and an
    // asymmetric window grid can derive to its own transpose with the same
    // window count, which every downstream shape check accepts.
    //
    // Ten is not capped by anything downstream. Nothing copies an attribute
    // into a fixed-width metadata array - lowering.cpp reads slots with get()
    // and passes the values straight to SushiBLAS.
    EXPECT_EQ(Graph::Attributes::CAPACITY, std::size_t{10});

    Graph::Attributes a;
    for (std::size_t i = 0; i < Graph::Attributes::CAPACITY; ++i)
        a.values[i] = static_cast<double>(i + 1);

    EXPECT_EQ(a.get(7), 8.0);
    EXPECT_EQ(a.get(9), 10.0);
    EXPECT_EQ(a.get(10), 0.0) << "out-of-range reads must stay defined and yield zero";
}
