/**************************************************************************/
/* test_backward.cpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free. Differentiation is an IR-to-IR transform, so what it produces
// is a graph that can be inspected node by node with no hardware involved -
// which is exactly why ir.hpp insists a backward rule emits IR rather than
// device work.

#include <algorithm>
#include <span>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::NO_VALUE;
using Graph::ValueId;

namespace
{
    /** @brief The op names of nodes [from, end), for a readable expectation. */
    std::vector<std::string> op_names(const Graph::Graph& graph, std::size_t from)
    {
        std::vector<std::string> names;
        for (std::size_t i = from; i < graph.node_count(); ++i)
            names.emplace_back(Ops::name_of(graph.nodes()[i].op));
        return names;
    }

    /** @brief How many nodes of @p op the whole graph holds. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(),
                          [op](const Graph::Node& node) { return node.op == op; }));
    }
}

TEST(Unit_Backward, EveryImplementedOperationHasARule)
{
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::MATMUL.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::BIAS_ADD.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::ADD.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SUB.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::MUL.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SCALE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::EXP.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::RELU.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::GELU.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SIGMOID.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::TANH.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::SUM_ROWS.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::ACCUMULATE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::NLL_LOSS.id));

    // LEAF and ZERO produce values from nothing, so there is nothing upstream
    // to hand a gradient to and no rule to write.
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::LEAF.id));
    EXPECT_FALSE(Autograd::has_backward_rule(Ops::ZERO.id));
}

TEST(Unit_Backward, TwoLayerMlpProducesTheExpectedBackwardTopology)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{8, 6}, "x");
    const ValueId targets = builder.input(Shape{8, 3}, "targets");

    NN::Sequential model{NN::Linear{6, 4}, NN::ReLU{}, NN::Linear{4, 3}};
    const ValueId logits = model.forward(builder, x);
    const ValueId loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    const std::size_t forward_nodes = graph.node_count();

    std::vector<ValueId> wrt;
    NN::for_each_parameter(model, [&wrt](const std::string&, NN::Parameter& parameter)
    {
        wrt.push_back(parameter.id());
    });
    ASSERT_EQ(wrt.size(), 4u);

    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, std::span<const ValueId>(wrt.data(), wrt.size()));

    // The cross-entropy pair is differentiated as one fused rule: exp of the
    // log-probabilities, minus the one-hot targets, scaled by 1/N. Then the
    // chain walks back through the second linear layer, the activation and the
    // first linear layer.
    EXPECT_EQ(op_names(graph, forward_nodes), (std::vector<std::string>{
        "sushiai.exp",
        "sushiai.sub",
        "sushiai.scale",
        "sushiai.sum_rows",         // d(bias) of the output layer
        "sushiai.matmul",           // d(hidden activation)
        "sushiai.matmul",           // d(weight) of the output layer
        "sushiai.relu_backward",
        "sushiai.sum_rows",         // d(bias) of the hidden layer
        "sushiai.matmul",           // d(weight) of the hidden layer
    }));

    // No LOG_SOFTMAX gradient: the fused rule absorbed that node.
    EXPECT_EQ(count_op(graph, Ops::LOG_SOFTMAX.id), 1u);

    // Every value has exactly one contributor in a plain feed-forward stack,
    // so the tiered accumulation emits nothing at all.
    EXPECT_EQ(count_op(graph, Ops::ACCUMULATE.id), 0u);

    ASSERT_EQ(grads.size(), 4u);
    for (std::size_t i = 0; i < grads.size(); ++i)
    {
        ASSERT_NE(grads[i], NO_VALUE) << "parameter " << i << " received no gradient";
        EXPECT_EQ(graph.value(grads[i]).shape, graph.value(wrt[i]).shape);
        EXPECT_EQ(graph.value(grads[i]).lifetime, Graph::Lifetime::GRADIENT);
    }
}

TEST(Unit_Backward, TheInputIsNotDifferentiatedWhenItDoesNotRequireOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{8, 6}, "x");
    const ValueId targets = builder.input(Shape{8, 3}, "targets");

    NN::Linear layer{6, 3};
    const ValueId logits = layer.forward(builder, x);
    const ValueId loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    const std::vector<ValueId> wrt = {layer.weight.id()};
    (void)Autograd::differentiate(graph, loss, std::span<const ValueId>(wrt.data(), wrt.size()));

    // Only one MATMUL in the backward pass: the weight gradient. The input
    // gradient would have been a second one, and it is skipped rather than
    // computed and discarded.
    EXPECT_EQ(count_op(graph, Ops::MATMUL.id), 2u);
}

TEST(Unit_Backward, FanOutEmitsAnAccumulateChain)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    // x feeds two branches, so its gradient has two contributors and the
    // tiered accumulation drops to its ordered ACCUMULATE form.
    const ValueId left = builder.relu(x);
    const ValueId right = builder.tanh(x);
    const ValueId sum = builder.add(left, right);
    const ValueId loss = builder.nll_loss(sum, targets);

    const std::vector<ValueId> wrt = {x};
    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, std::span<const ValueId>(wrt.data(), wrt.size()));

    ASSERT_EQ(grads.size(), 1u);
    ASSERT_NE(grads[0], NO_VALUE);
    EXPECT_EQ(count_op(graph, Ops::ACCUMULATE.id), 1u);
    EXPECT_EQ(graph.node(graph.value(grads[0]).producer).op, Ops::ACCUMULATE.id);
}

TEST(Unit_Backward, ThreeContributorsChainTwoAccumulates)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    const ValueId a = builder.relu(x);
    const ValueId b = builder.tanh(x);
    const ValueId c = builder.sigmoid(x);
    const ValueId loss = builder.nll_loss(builder.add(builder.add(a, b), c), targets);

    const std::vector<ValueId> wrt = {x};
    (void)Autograd::differentiate(graph, loss, std::span<const ValueId>(wrt.data(), wrt.size()));

    EXPECT_EQ(count_op(graph, Ops::ACCUMULATE.id), 2u);
}

TEST(Unit_Backward, ADisconnectedParameterGetsASymbolicZeroNotABuffer)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");
    const ValueId orphan = builder.parameter(Shape{9, 9}, "orphan");
    const ValueId loss = builder.nll_loss(builder.relu(x), targets);

    const std::vector<ValueId> wrt = {orphan, x};
    const std::size_t before = graph.node_count();

    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, std::span<const ValueId>(wrt.data(), wrt.size()));

    ASSERT_EQ(grads.size(), 2u);
    EXPECT_EQ(grads[0], NO_VALUE) << "a disconnected parameter has no gradient value at all";
    EXPECT_NE(grads[1], NO_VALUE);

    // A zero gradient must cost nothing: no ZERO node, no allocation, no
    // wasted kernel. Only the live path emitted anything, and that is exactly
    // the loss gradient and the activation gradient.
    EXPECT_EQ(count_op(graph, Ops::ZERO.id), 0u);
    EXPECT_EQ(graph.node_count(), before + 2);
}

TEST(Unit_Backward, ADeadLossPathEmitsNothingAtAll)
{
    // Nothing in this graph requires a gradient, so the reverse walk visits
    // every node and writes none of them. A frozen model costs zero nodes,
    // not a set of nodes computing gradients nobody reads.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");
    const ValueId orphan = builder.parameter(Shape{9, 9}, "orphan");
    const ValueId loss = builder.nll_loss(builder.relu(x), targets);

    const std::vector<ValueId> wrt = {orphan};
    const std::size_t before = graph.node_count();

    const std::vector<ValueId> grads =
        Autograd::differentiate(graph, loss, std::span<const ValueId>(wrt.data(), wrt.size()));

    EXPECT_EQ(grads.at(0), NO_VALUE);
    EXPECT_EQ(graph.node_count(), before);
}

TEST(Unit_Backward, AnOperationWithNoRuleThrowsAndNamesItself)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    // A synthetic identity that no rule covers, standing in for a future op
    // added to the lowering but not to the rule table.
    const ValueId mystery = graph.add_value(Shape{4, 3}, DType::FLOAT32,
                                            Graph::Lifetime::ACTIVATION, "mystery");
    graph.add_node(OpID(0x1234'5678u), {x}, {mystery});

    const ValueId loss = builder.nll_loss(mystery, targets);
    const std::vector<ValueId> wrt = {x};

    EXPECT_THROW((void)Autograd::differentiate(
                     graph, loss, std::span<const ValueId>(wrt.data(), wrt.size())),
                 Error);
}

TEST(Unit_Backward, ANonScalarRootIsRejected)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId y = builder.relu(x);
    const std::vector<ValueId> wrt = {x};

    EXPECT_THROW((void)Autograd::differentiate(
                     graph, y, std::span<const ValueId>(wrt.data(), wrt.size())),
                 Error);
}

TEST(Unit_Backward, LogSoftmaxOutsideTheFusedPairSaysSoClearly)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    // The log-softmax result is consumed by a RELU, so the cross-entropy
    // fusion does not apply and the standalone rule has to refuse.
    const ValueId detoured = builder.relu(builder.log_softmax(x));
    const ValueId loss = builder.nll_loss(detoured, targets);

    const std::vector<ValueId> wrt = {x};
    EXPECT_THROW((void)Autograd::differentiate(
                     graph, loss, std::span<const ValueId>(wrt.data(), wrt.size())),
                 Error);
}
