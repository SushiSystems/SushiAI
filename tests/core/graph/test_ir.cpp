/**************************************************************************/
/* test_ir.cpp                                                            */
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
 * @file test_ir.cpp
 * @brief Tests the IR graph: its values, nodes and producers, and the errors it names.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Lifetime;
using Graph::K_NO_STATE;
using Graph::K_NO_NODE;
using Graph::K_NO_VALUE;
using Graph::StateID;
using Graph::ValueID;

TEST(Small_Ir, AddValueRecordsShapeDTypeAndLifetime)
{
    Graph::Graph graph;
    const ValueID id = graph.add_value(Shape{2, 3}, DType::Float64,
                                       Lifetime::Parameter, "w");

    EXPECT_EQ(graph.value_count(), 1u);
    EXPECT_EQ(graph.value(id).shape, Shape({2, 3}));
    EXPECT_EQ(graph.value(id).dtype, DType::Float64);
    EXPECT_EQ(graph.value(id).lifetime, Lifetime::Parameter);
    EXPECT_EQ(graph.value(id).debug_name, "w");
    EXPECT_EQ(graph.value(id).producer, K_NO_NODE);
}

TEST(Small_Ir, AddNodeClaimsItsOutputs)
{
    Graph::Graph graph;
    const ValueID a = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID b = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID c = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation);

    const Graph::NodeID node = graph.add_node(Ops::K_ADD.id, {a, b}, {c});

    EXPECT_EQ(graph.value(c).producer, node);
    EXPECT_EQ(graph.node(node).inputs.size(), 2u);
}

TEST(Small_Ir, ASecondProducerForOneValueIsFatal)
{
    // A value has exactly one producer; see tests/core/graph/README.md.
    Graph::Graph graph;
    const ValueID a = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID b = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID c = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation);

    graph.add_node(Ops::K_ADD.id, {a, b}, {c});
    SA_EXPECT_FATAL((void)graph.add_node(Ops::K_SUB.id, {a, b}, {c}), "already written");
}

TEST(Medium_IrFatal, ReferencingAnUnknownValueIsFatal)
{
    Graph::Graph graph;
    const ValueID a = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);

    SA_EXPECT_FATAL((void)graph.add_node(Ops::K_ADD.id, {a, 99}, {a}), "reads value 99");
    SA_EXPECT_FATAL((void)graph.add_node(Ops::K_ADD.id, {a}, {99}), "writes value 99");
}

TEST(Small_Ir, PropagateRequiresGradFlowsForwardThroughTheWholeChain)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID w = builder.parameter(Shape{5, 3}, "w");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.matmul(x, w, false, true));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID z, builder.relu(y));

    EXPECT_FALSE(graph.value(y).requires_grad);
    EXPECT_FALSE(graph.value(z).requires_grad);

    SA_EXPECT_OK(graph.finalise());

    EXPECT_FALSE(graph.value(x).requires_grad);
    EXPECT_TRUE(graph.value(w).requires_grad);
    EXPECT_TRUE(graph.value(y).requires_grad);
    EXPECT_TRUE(graph.value(z).requires_grad);
}

TEST(Small_Ir, PropagateRequiresGradLeavesAFrozenBranchAlone)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID frozen, builder.relu(x));
    const ValueID w = builder.parameter(Shape{3, 3}, "w");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID live, builder.matmul(x, w, false, false));

    SA_EXPECT_OK(graph.finalise());

    EXPECT_FALSE(graph.value(frozen).requires_grad);
    EXPECT_TRUE(graph.value(live).requires_grad);
}

namespace
{
    /** @brief Names the operand shapes and the transposition flags of one matmul row. */
    struct MatmulRow
    {
        const char* name; /**< The row's name in the test report. */
        Shape a;          /**< The left operand shape. */
        Shape b;          /**< The right operand shape. */
        bool transpose_a; /**< Whether the left operand is read transposed. */
        bool transpose_b; /**< Whether the right operand is read transposed. */
    };

    /** @brief Runs once per transposition case of matmul shape inference. */
    class Small_IrMatmulRow : public ::testing::TestWithParam<MatmulRow>
    {
    };

    const MatmulRow K_MATMUL_ROWS[] = {
        {"NoTranspose", Shape{4, 3}, Shape{3, 5}, false, false},
        {"TransposedLeft", Shape{3, 4}, Shape{3, 5}, true, false},
        {"TransposedRight", Shape{4, 3}, Shape{5, 3}, false, true},
        {"TransposedBoth", Shape{3, 4}, Shape{5, 3}, true, true},
    };
} // namespace

/** @brief Tests that matmul infers a [4, 5] product under each transposition case. */
TEST_P(Small_IrMatmulRow, BuilderInfersMatmulShapesForEveryTranspositionCase)
{
    const MatmulRow row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(row.a, "a");
    const ValueID b = builder.input(row.b, "b");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID product,
                            builder.matmul(a, b, row.transpose_a, row.transpose_b));

    EXPECT_EQ(graph.value(product).shape, Shape({4, 5}));
}

INSTANTIATE_TEST_SUITE_P(, Small_IrMatmulRow, ::testing::ValuesIn(K_MATMUL_ROWS),
                         [](const ::testing::TestParamInfo<MatmulRow>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });

TEST(Small_Ir, BuilderRejectsAContractionMismatch)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{7, 5}, "b");

    SA_EXPECT_FAILURE(builder.matmul(a, b, false, false), StatusCode::InvalidDescription);
}

/** @brief Tests that bias_add keeps the shape of its input. */
TEST(Small_Ir, BiasAddKeepsTheInputShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 5}, "x");
    const ValueID bias = builder.input(Shape{5}, "bias");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID biased, builder.bias_add(x, bias));

    EXPECT_EQ(graph.value(biased).shape, Shape({4, 5}));
}

/** @brief Tests that bias_add refuses a bias whose extent is not the column count. */
TEST(Small_Ir, BiasAddRefusesABiasOfTheWrongExtent)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 5}, "x");
    const ValueID bad = builder.input(Shape{4}, "bad");

    SA_EXPECT_FAILURE(builder.bias_add(x, bad), StatusCode::InvalidDescription);
}

/** @brief Tests that sum_rows yields one element per column. */
TEST(Small_Ir, SumRowsYieldsOneElementPerColumn)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 5}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID column_sums, builder.sum_rows(x));

    EXPECT_EQ(graph.value(column_sums).shape, Shape({5}));
}

/** @brief Tests that nll_loss yields a single element for logits and targets of one shape. */
TEST(Small_Ir, NllLossYieldsOneElement)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 5}, "x");
    const ValueID targets = builder.input(Shape{4, 5}, "targets");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss, builder.nll_loss(x, targets));

    EXPECT_EQ(graph.value(loss).shape, Shape({1}));
}

/** @brief Tests that nll_loss refuses targets of another shape than the logits. */
TEST(Small_Ir, NllLossRefusesTargetsOfAnotherShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 5}, "x");
    const ValueID wrong = builder.input(Shape{4, 6}, "wrong");

    SA_EXPECT_FAILURE(builder.nll_loss(x, wrong), StatusCode::InvalidDescription);
}

TEST(Small_Ir, BuilderRejectsMixedElementTypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID single = builder.input(Shape{4, 4}, "f32", DType::Float32);
    const ValueID doubled = builder.input(Shape{4, 4}, "f64", DType::Float64);

    SA_EXPECT_FAILURE(builder.add(single, doubled), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE(builder.matmul(single, doubled, false, false),
                      StatusCode::InvalidDescription);
}

/** @brief Tests that a node is emitted after the node that produces its operand. */
TEST(Small_Ir, NodesAreEmittedInTopologicalOrder)
{
    // Insertion order is a topological order; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.relu(x));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID z, builder.tanh(y));

    EXPECT_LT(graph.value(y).producer, graph.value(z).producer);}

/** @brief Tests that insert_node is fatal for a producer placed after the node that reads it. */
TEST(Small_Ir, InsertNodeIsFatalForAResultThatIsAlreadyReadEarlier)
{
    // insert_node refuses a producer placed after its reader; see tests/core/graph/README.md.
    Graph::Graph graph;
    const ValueID x = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");
    const ValueID orphan = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "orphan");
    const ValueID y = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "y");

    // Node 0 reads `orphan`, which nothing writes yet.
    (void)graph.add_node(Ops::K_ADD.id, {x, orphan}, {y});

    SA_EXPECT_FATAL((void)graph.insert_node(1, Graph::Node{Ops::K_RELU.id, {x}, {orphan}, {}}),
                    "sushiai.relu", "inserted at 1", "value 1", "node 0 reads", "topological");
}

/** @brief Tests that insert_node accepts the same producer ahead of the node that reads it. */
TEST(Small_Ir, InsertNodeAcceptsAResultAheadOfItsReader)
{
    Graph::Graph graph;
    const ValueID x = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");
    const ValueID orphan = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "orphan");
    const ValueID y = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "y");
    (void)graph.add_node(Ops::K_ADD.id, {x, orphan}, {y});

    (void)graph.insert_node(0, Graph::Node{Ops::K_RELU.id, {x}, {orphan}, {}});

    EXPECT_EQ(graph.node_count(), 2u);
    EXPECT_EQ(graph.value(orphan).producer, Graph::NodeID{0});
    EXPECT_EQ(graph.value(y).producer, Graph::NodeID{1});
}

TEST(Small_Ir, InsertNodeIsFatalForAnOperandProducedNoEarlierThanItself)
{
    // The mirror of the case above, and the one that was guarded first.
    Graph::Graph graph;
    const ValueID x = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");
    const ValueID y = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "y");
    const ValueID z = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "z");

    (void)graph.add_node(Ops::K_RELU.id, {x}, {y});

    SA_EXPECT_FATAL((void)graph.insert_node(0, Graph::Node{Ops::K_TANH.id, {y}, {z}, {}}),
                    "sushiai.tanh", "inserted at 0", "value 1", "node 0 writes", "topological");
}

/** @brief Tests that a read past the last attribute slot yields zero. */
TEST(Small_Ir, AnAttributeReadPastTheLastSlotYieldsZero)
{
    Graph::Attributes a;
    std::fill(std::begin(a.values), std::end(a.values), 1.0);

    EXPECT_EQ(a.get(Graph::Attributes::K_CAPACITY), 0.0);
}

/** @brief Tests that a buffer leaf is persistent and is produced by LEAF. */
TEST(Small_Ir, ABufferLeafIsPersistent)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID running = builder.buffer(Shape{3}, "running_mean", DType::Float64);

    SA_EXPECT_OK(graph.finalise());

    EXPECT_EQ(graph.value(running).shape, Shape({3}));
    EXPECT_EQ(graph.value(running).dtype, DType::Float64);
    EXPECT_EQ(graph.value(running).lifetime, Lifetime::Buffer);
    EXPECT_EQ(graph.value(running).debug_name, "running_mean");
    ASSERT_NE(graph.value(running).producer, K_NO_NODE);
    EXPECT_EQ(graph.node(graph.value(running).producer).op, Ops::K_LEAF.id);
    EXPECT_TRUE(Graph::is_persistent(Lifetime::Buffer));
    EXPECT_EQ(std::string_view(Graph::lifetime_name(Lifetime::Buffer)), "buffer");
}

/** @brief Tests that a buffer leaf needs no gradient and passes none to what reads it. */
TEST(Small_Ir, ABufferLeafNeedsNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID running = builder.buffer(Shape{3}, "running_mean", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y,
                            builder.bias_add(x, builder.buffer(Shape{3}, "shift")));

    SA_EXPECT_OK(graph.finalise());

    EXPECT_FALSE(graph.value(running).requires_grad);
    EXPECT_FALSE(graph.value(y).requires_grad);
}

namespace
{
    /** @brief Pairs a lifetime with whether it outlives a step. */
    struct LifetimeRow
    {
        Lifetime lifetime; /**< The lifetime under test. */
        bool persistent;   /**< Whether values of it outlive a step. */
    };

    /** @brief Runs once per lifetime. */
    class Small_IrLifetimeRow : public ::testing::TestWithParam<LifetimeRow>
    {
    };

    const LifetimeRow K_LIFETIME_ROWS[] = {
        {Lifetime::Parameter, true},      {Lifetime::Gradient, true},
        {Lifetime::OptimizerState, true}, {Lifetime::Input, true},
        {Lifetime::Buffer, true},         {Lifetime::Activation, false},
        {Lifetime::Scratch, false},
    };
} // namespace

/** @brief Tests that the five lifetimes that outlive a step persist and the other two do not. */
TEST_P(Small_IrLifetimeRow, ActivationAndScratchAreTheOnlyLifetimesThatDoNotPersist)
{
    const LifetimeRow row = GetParam();

    EXPECT_EQ(Graph::is_persistent(row.lifetime), row.persistent)
        << Graph::lifetime_name(row.lifetime);
}

INSTANTIATE_TEST_SUITE_P(, Small_IrLifetimeRow, ::testing::ValuesIn(K_LIFETIME_ROWS),
                         [](const ::testing::TestParamInfo<LifetimeRow>& info)
                         {
                             return ::SushiAI::Test::parameter_name(Graph::lifetime_name(info.param.lifetime));
                         });

/** @brief Tests that state_leaf returns the leaf bound to a state. */
TEST(Small_Ir, StateLeafReturnsTheBoundLeaf)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{3}, "weight");
    const ValueID running = builder.buffer(Shape{3}, "running_mean");

    graph.bind_state_leaf(StateID{7}, running);
    graph.bind_state_leaf(StateID{5}, weight);

    EXPECT_EQ(graph.state_leaf(StateID{7}), running);
    EXPECT_EQ(graph.state_leaf(StateID{5}), weight);
}

/** @brief Tests that state_leaf returns NO_VALUE for a state no leaf is bound to. */
TEST(Small_Ir, StateLeafReturnsNoValueForAnUnboundState)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    graph.bind_state_leaf(StateID{7}, builder.buffer(Shape{3}, "running_mean"));

    EXPECT_EQ(graph.state_leaf(StateID{6}), K_NO_VALUE);
    EXPECT_EQ(graph.state_leaf(K_NO_STATE), K_NO_VALUE);
}

/** @brief Tests that state_leaves lists the bindings in the order they were made. */
TEST(Small_Ir, StateLeavesListTheBindingsInBindingOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{3}, "weight");
    const ValueID running = builder.buffer(Shape{3}, "running_mean");

    EXPECT_TRUE(graph.state_leaves().empty());

    graph.bind_state_leaf(StateID{7}, running);
    graph.bind_state_leaf(StateID{5}, weight);

    ASSERT_EQ(graph.state_leaves().size(), 2u);
    EXPECT_EQ(graph.state_leaves()[0], std::make_pair(StateID{7}, running));
    EXPECT_EQ(graph.state_leaves()[1], std::make_pair(StateID{5}, weight));
}

/** @brief Tests that a state binds one leaf and a leaf stands for one state, in one graph. */
TEST(Medium_IrFatal, BindStateLeafIsFatalForASecondBindingOfAStateOrOfALeaf)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID first = builder.buffer(Shape{3}, "first");
    const ValueID second = builder.buffer(Shape{3}, "second");

    graph.bind_state_leaf(StateID{7}, first);

    SA_EXPECT_FATAL(graph.bind_state_leaf(StateID{7}, second), "State 7", "already has leaf");
    SA_EXPECT_FATAL(graph.bind_state_leaf(StateID{8}, first), "already stands for state 7");
}

/** @brief Tests that bind_state_leaf is fatal without an identity and for an unknown value. */
TEST(Medium_IrFatal, BindStateLeafIsFatalForNoIdentityOrAnUnknownValue)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.buffer(Shape{3}, "leaf");

    SA_EXPECT_FATAL(graph.bind_state_leaf(K_NO_STATE, leaf), "no identity");
    SA_EXPECT_FATAL(graph.bind_state_leaf(StateID{9}, ValueID{99}), "value 99", "does not exist");
}

namespace
{
    /** @brief Holds a finalised graph with two layouts, both gradient flags and three types. */
    struct StrippedFixture
    {
        Graph::Graph graph;    /**< The source graph. */
        ValueID x_nchw = 0;    /**< A converted input in the NCHW layout. */
        ValueID weight = 0;    /**< A parameter that requires a gradient. */
        ValueID running = 0;   /**< A half-precision buffer. */
        Graph::Graph stripped; /**< The copy without nodes. */

        /** @brief Builds the graph, binds two state leaves and strips a copy. */
        StrippedFixture()
        {
            const GraphBuilder builder(graph);
            const ValueID x = builder.input(Shape{1, 3, 4, 4}, "x", DType::Float64);
            x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW).value();
            weight = builder.parameter(Shape{2, 3}, "weight");
            running = builder.buffer(Shape{3}, "running", DType::Half);
            graph.bind_state_leaf(StateID{11}, weight);
            graph.bind_state_leaf(StateID{12}, running);
            (void)graph.finalise();
            stripped = graph.without_nodes();
        }
    };

    /** @brief Expects every field of a value in @p after to equal the one in @p before. */
    void expect_value_kept(const Graph::Value& before, const Graph::Value& after)
    {
        EXPECT_EQ(after.shape, before.shape) << before.debug_name;
        EXPECT_EQ(after.dtype, before.dtype) << before.debug_name;
        EXPECT_EQ(after.layout, before.layout) << before.debug_name;
        EXPECT_EQ(after.lifetime, before.lifetime) << before.debug_name;
        EXPECT_EQ(after.requires_grad, before.requires_grad) << before.debug_name;
        EXPECT_EQ(after.debug_name, before.debug_name);
    }
} // namespace

/** @brief Tests that without_nodes holds no node and leaves no value with a producer. */
TEST(Small_Ir, WithoutNodesHoldsNoNode)
{
    const StrippedFixture fixture;

    EXPECT_EQ(fixture.stripped.node_count(), 0u);
    ASSERT_EQ(fixture.stripped.value_count(), fixture.graph.value_count());
    EXPECT_EQ(fixture.stripped.value(fixture.x_nchw).producer, K_NO_NODE);
    EXPECT_EQ(fixture.stripped.value(fixture.weight).producer, K_NO_NODE);
}

/** @brief Tests that without_nodes keeps every field of the values it copies. */
TEST(Small_Ir, WithoutNodesKeepsEveryValueField)
{
    const StrippedFixture fixture;

    expect_value_kept(fixture.graph.value(fixture.x_nchw), fixture.stripped.value(fixture.x_nchw));
    expect_value_kept(fixture.graph.value(fixture.weight), fixture.stripped.value(fixture.weight));
    expect_value_kept(fixture.graph.value(fixture.running),
                      fixture.stripped.value(fixture.running));
    EXPECT_EQ(fixture.stripped.value(fixture.x_nchw).layout, Graph::TensorLayout::NCHW);
    EXPECT_TRUE(fixture.stripped.value(fixture.weight).requires_grad);
    EXPECT_EQ(fixture.stripped.value(fixture.running).dtype, DType::Half);
}

/** @brief Tests that without_nodes keeps the state leaves. */
TEST(Small_Ir, WithoutNodesKeepsTheStateLeaves)
{
    const StrippedFixture fixture;

    EXPECT_EQ(fixture.stripped.state_leaves(), fixture.graph.state_leaves());
    EXPECT_EQ(fixture.stripped.state_leaf(StateID{12}), fixture.running);
}

/** @brief Tests that without_nodes leaves the source graph unchanged. */
TEST(Small_Ir, WithoutNodesLeavesTheSourceGraphUnchanged)
{
    const StrippedFixture fixture;

    EXPECT_NE(fixture.graph.value(fixture.x_nchw).producer, K_NO_NODE);
    EXPECT_GT(fixture.graph.node_count(), 0u);
}

/** @brief Tests that a graph reports a refusal only once one was marked. */
TEST(Small_Ir, ARefusalIsReportedAfterItIsMarked)
{
    Graph::Graph graph;
    (void)graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");

    EXPECT_FALSE(graph.has_refusal());

    graph.mark_refused();

    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that a copy without nodes keeps a marked refusal and adds none. */
TEST(Small_Ir, AMarkedRefusalSurvivesWithoutNodes)
{
    Graph::Graph graph;
    (void)graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");

    EXPECT_FALSE(graph.without_nodes().has_refusal());

    graph.mark_refused();

    EXPECT_TRUE(graph.without_nodes().has_refusal());
}

TEST(Small_Ir, ASecondProducerNamesBothOperationsAndTheRule)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID out = graph.add_value(Shape{4, 3}, DType::Float32,
                                        Graph::Lifetime::Activation, "out");
    graph.add_node(Ops::K_RELU.id, {x}, {out});

    // The message names both colliding operations; see tests/core/graph/README.md.
    SA_EXPECT_FATAL((void)graph.add_node(Ops::K_TANH.id, {x}, {out}),
                    "sushiai.tanh", "sushiai.relu", "single-assignment");
}

TEST(Small_Ir, AnUnknownOperandNamesTheOperationThatAskedForIt)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID out = graph.add_value(Shape{4, 3}, DType::Float32,
                                        Graph::Lifetime::Activation, "out");

    SA_EXPECT_FATAL((void)graph.add_node(Ops::K_RELU.id, {9999u}, {out}),
                    "sushiai.relu", "reads value 9999", "does not exist");
    SA_EXPECT_FATAL((void)graph.add_node(Ops::K_RELU.id, {x}, {9999u}),
                    "sushiai.relu", "writes value 9999", "does not exist");
}
