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
 * @file test_ir.cpp
 * @brief Tests the IR's single-assignment and ordering rules and the builder's shape inference.
 * @author Mustafa Garip
 */

#include <string_view>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Lifetime;
using Graph::K_NO_STATE;
using Graph::K_NO_NODE;
using Graph::K_NO_VALUE;
using Graph::StateID;
using Graph::StateIdentity;
using Graph::ValueID;

TEST(Unit_Ir, AddValueRecordsShapeDTypeAndLifetime)
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

TEST(Unit_Ir, AddNodeClaimsItsOutputs)
{
    Graph::Graph graph;
    const ValueID a = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID b = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID c = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation);

    const Graph::NodeID node = graph.add_node(Ops::K_ADD.id, {a, b}, {c});

    EXPECT_EQ(graph.value(c).producer, node);
    EXPECT_EQ(graph.node(node).inputs.size(), 2u);
}

TEST(Unit_Ir, ASecondProducerForOneValueThrows)
{
    // A value has exactly one producer; see tests/unit/README.md.
    Graph::Graph graph;
    const ValueID a = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID b = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
    const ValueID c = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation);

    graph.add_node(Ops::K_ADD.id, {a, b}, {c});
    EXPECT_THROW(graph.add_node(Ops::K_SUB.id, {a, b}, {c}), Error);
}

TEST(Unit_Ir, ReferencingAnUnknownValueThrows)
{
    Graph::Graph graph;
    const ValueID a = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);

    EXPECT_THROW(graph.add_node(Ops::K_ADD.id, {a, 99}, {a}), Error);
    EXPECT_THROW(graph.add_node(Ops::K_ADD.id, {a}, {99}), Error);
}

TEST(Unit_Ir, PropagateRequiresGradFlowsForwardThroughTheWholeChain)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID w = builder.parameter(Shape{5, 3}, "w");
    const ValueID y = builder.matmul(x, w, false, true);
    const ValueID z = builder.relu(y);

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

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID frozen = builder.relu(x);
    const ValueID w = builder.parameter(Shape{3, 3}, "w");
    const ValueID live = builder.matmul(x, w, false, false);

    graph.finalise();

    EXPECT_FALSE(graph.value(frozen).requires_grad);
    EXPECT_TRUE(graph.value(live).requires_grad);
}

TEST(Unit_Ir, BuilderInfersMatmulShapesForEveryTranspositionCase)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 3}, "a");    // [m, k]
    const ValueID at = builder.input(Shape{3, 4}, "at");  // [k, m]
    const ValueID b = builder.input(Shape{3, 5}, "b");    // [k, n]
    const ValueID bt = builder.input(Shape{5, 3}, "bt");  // [n, k]

    EXPECT_EQ(graph.value(builder.matmul(a, b, false, false)).shape, Shape({4, 5}));
    EXPECT_EQ(graph.value(builder.matmul(at, b, true, false)).shape, Shape({4, 5}));
    EXPECT_EQ(graph.value(builder.matmul(a, bt, false, true)).shape, Shape({4, 5}));
    EXPECT_EQ(graph.value(builder.matmul(at, bt, true, true)).shape, Shape({4, 5}));
}

TEST(Unit_Ir, BuilderRejectsAContractionMismatch)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{7, 5}, "b");

    EXPECT_THROW((void)builder.matmul(a, b, false, false), Error);
}

TEST(Unit_Ir, BuilderChecksBiasAndLossShapes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 5}, "x");
    const ValueID good = builder.input(Shape{5}, "good");
    const ValueID bad = builder.input(Shape{4}, "bad");

    EXPECT_EQ(graph.value(builder.bias_add(x, good)).shape, Shape({4, 5}));
    EXPECT_THROW((void)builder.bias_add(x, bad), Error);

    EXPECT_EQ(graph.value(builder.sum_rows(x)).shape, Shape({5}));

    const ValueID targets = builder.input(Shape{4, 5}, "targets");
    EXPECT_EQ(graph.value(builder.nll_loss(x, targets)).shape, Shape({1}));

    const ValueID wrong = builder.input(Shape{4, 6}, "wrong");
    EXPECT_THROW((void)builder.nll_loss(x, wrong), Error);
}

TEST(Unit_Ir, BuilderRejectsMixedElementTypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID single = builder.input(Shape{4, 4}, "f32", DType::Float32);
    const ValueID doubled = builder.input(Shape{4, 4}, "f64", DType::Float64);

    EXPECT_THROW((void)builder.add(single, doubled), Error);
    EXPECT_THROW((void)builder.matmul(single, doubled, false, false), Error);
}

TEST(Unit_Ir, NodesAreEmittedInTopologicalOrder)
{
    // Insertion order is a topological order; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 2}, "x");
    const ValueID y = builder.relu(x);
    const ValueID z = builder.tanh(y);
    (void)z;

    for (std::size_t i = 0; i < graph.node_count(); ++i)
        for (Graph::ValueID in : graph.nodes()[i].inputs)
        {
            const Graph::NodeID producer = graph.value(in).producer;
            if (producer != K_NO_NODE) EXPECT_LT(producer, i);
        }
}

TEST(Unit_Ir, InsertNodeRejectsAResultThatIsAlreadyReadEarlier)
{
    // insert_node refuses a producer placed after its reader; see tests/unit/README.md.
    Graph::Graph graph;
    const ValueID x = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");
    const ValueID orphan = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "orphan");
    const ValueID y = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "y");

    // Node 0 reads `orphan`, which nothing writes yet.
    (void)graph.add_node(Ops::K_ADD.id, {x, orphan}, {y});

    SA_EXPECT_ERROR((void)graph.insert_node(1, Graph::Node{Ops::K_RELU.id, {x}, {orphan}, {}}),
                    "sushiai.relu", "inserted at 1", "value 1", "node 0 reads", "topological");

    // A rejected insert leaves the graph unchanged; see tests/unit/README.md.
    EXPECT_EQ(graph.node_count(), 1u);
    EXPECT_EQ(graph.value(orphan).producer, K_NO_NODE);

    // The same node ahead of the reader is accepted; see tests/unit/README.md.
    EXPECT_NO_THROW((void)graph.insert_node(0, Graph::Node{Ops::K_RELU.id, {x}, {orphan}, {}}));
    EXPECT_EQ(graph.node_count(), 2u);
    EXPECT_EQ(graph.value(orphan).producer, Graph::NodeID{0});
    EXPECT_EQ(graph.value(y).producer, Graph::NodeID{1});
}

TEST(Unit_Ir, InsertNodeRejectsAnOperandProducedNoEarlierThanItself)
{
    // The mirror of the case above, and the one that was guarded first.
    Graph::Graph graph;
    const ValueID x = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");
    const ValueID y = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "y");
    const ValueID z = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation, "z");

    (void)graph.add_node(Ops::K_RELU.id, {x}, {y});

    SA_EXPECT_ERROR((void)graph.insert_node(0, Graph::Node{Ops::K_TANH.id, {y}, {z}, {}}),
                    "sushiai.tanh", "inserted at 0", "value 1", "node 0 writes", "topological");

    EXPECT_EQ(graph.node_count(), 1u);
    EXPECT_EQ(graph.value(z).producer, K_NO_NODE);
}

TEST(Unit_IR, AttributesHoldAConvolutionsGeometryAndItsWindowGrid)
{
    // Slots 8 and 9 carry the output extents out_h and out_w; see tests/unit/README.md.
    EXPECT_EQ(Graph::Attributes::K_CAPACITY, std::size_t{10});

    Graph::Attributes a;
    for (std::size_t i = 0; i < Graph::Attributes::K_CAPACITY; ++i)
        a.values[i] = static_cast<double>(i + 1);

    EXPECT_EQ(a.get(7), 8.0);
    EXPECT_EQ(a.get(9), 10.0);
    EXPECT_EQ(a.get(10), 0.0) << "out-of-range reads must stay defined and yield zero";
}

/** @brief Tests that a buffer leaf is persistent, produced by LEAF and never needs a gradient. */
TEST(Unit_Ir, ABufferLeafIsPersistentAndNeedsNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID running = builder.buffer(Shape{3}, "running_mean", DType::Float64);
    const ValueID y = builder.bias_add(x, builder.buffer(Shape{3}, "shift"));

    graph.finalise();

    EXPECT_EQ(graph.value(running).shape, Shape({3}));
    EXPECT_EQ(graph.value(running).dtype, DType::Float64);
    EXPECT_EQ(graph.value(running).lifetime, Lifetime::Buffer);
    EXPECT_EQ(graph.value(running).debug_name, "running_mean");
    EXPECT_FALSE(graph.value(running).requires_grad);
    EXPECT_FALSE(graph.value(y).requires_grad);

    ASSERT_NE(graph.value(running).producer, K_NO_NODE);
    EXPECT_EQ(graph.node(graph.value(running).producer).op, Ops::K_LEAF.id);

    EXPECT_TRUE(Graph::is_persistent(Lifetime::Buffer));
    EXPECT_EQ(std::string_view(Graph::lifetime_name(Lifetime::Buffer)), "buffer");
}

/** @brief Tests that the persistent lifetimes are the five that outlive a step. */
TEST(Unit_Ir, ActivationAndScratchAreTheOnlyLifetimesThatDoNotPersist)
{
    for (Lifetime lifetime : {Lifetime::Parameter, Lifetime::Gradient, Lifetime::OptimizerState,
                              Lifetime::Input, Lifetime::Buffer})
        EXPECT_TRUE(Graph::is_persistent(lifetime)) << Graph::lifetime_name(lifetime);

    EXPECT_FALSE(Graph::is_persistent(Lifetime::Activation));
    EXPECT_FALSE(Graph::is_persistent(Lifetime::Scratch));
}

/** @brief Tests that state_leaf returns the leaf bound to a state and NO_VALUE for any other. */
TEST(Unit_Ir, StateLeafReturnsTheBoundLeafAndNoValueForAnUnboundState)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{3}, "weight");
    const ValueID running = builder.buffer(Shape{3}, "running_mean");

    EXPECT_EQ(graph.state_leaf(StateID{7}), K_NO_VALUE);
    EXPECT_TRUE(graph.state_leaves().empty());

    graph.bind_state_leaf(StateID{7}, running);
    graph.bind_state_leaf(StateID{5}, weight);

    EXPECT_EQ(graph.state_leaf(StateID{7}), running);
    EXPECT_EQ(graph.state_leaf(StateID{5}), weight);
    EXPECT_EQ(graph.state_leaf(StateID{6}), K_NO_VALUE);
    EXPECT_EQ(graph.state_leaf(K_NO_STATE), K_NO_VALUE);

    ASSERT_EQ(graph.state_leaves().size(), 2u);
    EXPECT_EQ(graph.state_leaves()[0], std::make_pair(StateID{7}, running));
    EXPECT_EQ(graph.state_leaves()[1], std::make_pair(StateID{5}, weight));
}

/** @brief Tests that a state binds one leaf and a leaf stands for one state, in one graph. */
TEST(Unit_Ir, BindStateLeafRefusesASecondBindingOfAStateOrOfALeaf)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID first = builder.buffer(Shape{3}, "first");
    const ValueID second = builder.buffer(Shape{3}, "second");

    graph.bind_state_leaf(StateID{7}, first);

    SA_EXPECT_ERROR(graph.bind_state_leaf(StateID{7}, second), "State 7", "already has leaf");
    SA_EXPECT_ERROR(graph.bind_state_leaf(StateID{8}, first), "already stands for state 7");
    SA_EXPECT_ERROR(graph.bind_state_leaf(K_NO_STATE, second), "no identity");
    SA_EXPECT_ERROR(graph.bind_state_leaf(StateID{9}, ValueID{99}), "value 99", "does not exist");

    // A refused binding records nothing, so the table still holds the first pair alone.
    ASSERT_EQ(graph.state_leaves().size(), 1u);
    EXPECT_EQ(graph.state_leaf(StateID{7}), first);
    EXPECT_EQ(graph.state_leaf(StateID{8}), K_NO_VALUE);
}

/** @brief Tests that without_nodes keeps every value field and state leaf and holds no node. */
TEST(Unit_Ir, WithoutNodesKeepsEveryValueAndStateLeafAndHoldsNoNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{1, 3, 4, 4}, "x", DType::Float64);
    const ValueID x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
    const ValueID weight = builder.parameter(Shape{2, 3}, "weight");
    const ValueID running = builder.buffer(Shape{3}, "running", DType::Half);
    graph.bind_state_leaf(StateID{11}, weight);
    graph.bind_state_leaf(StateID{12}, running);
    graph.finalise();

    const Graph::Graph stripped = graph.without_nodes();

    EXPECT_EQ(stripped.node_count(), 0u);
    ASSERT_EQ(stripped.value_count(), graph.value_count());

    for (ValueID id = 0; id < static_cast<ValueID>(graph.value_count()); ++id)
    {
        const Graph::Value& before = graph.value(id);
        const Graph::Value& after = stripped.value(id);

        EXPECT_EQ(after.shape, before.shape) << "value " << id;
        EXPECT_EQ(after.dtype, before.dtype) << "value " << id;
        EXPECT_EQ(after.layout, before.layout) << "value " << id;
        EXPECT_EQ(after.lifetime, before.lifetime) << "value " << id;
        EXPECT_EQ(after.requires_grad, before.requires_grad) << "value " << id;
        EXPECT_EQ(after.debug_name, before.debug_name) << "value " << id;
        EXPECT_EQ(after.producer, K_NO_NODE) << "value " << id;
    }

    // The fixture holds both layouts, both gradient flags and three element types.
    EXPECT_EQ(stripped.value(x_nchw).layout, Graph::TensorLayout::NCHW);
    EXPECT_TRUE(stripped.value(weight).requires_grad);
    EXPECT_EQ(stripped.value(running).dtype, DType::Half);

    EXPECT_EQ(stripped.state_leaves(), graph.state_leaves());
    EXPECT_EQ(stripped.state_leaf(StateID{12}), running);

    // The source graph is not changed by the copy.
    EXPECT_NE(graph.value(x_nchw).producer, K_NO_NODE);
    EXPECT_GT(graph.node_count(), 0u);
}

/** @brief Tests that a graph reports a refusal only once one was marked, and a copy keeps it. */
TEST(Unit_Ir, ARefusalIsReportedAfterItIsMarkedAndSurvivesWithoutNodes)
{
    Graph::Graph graph;
    (void)graph.add_value(Shape{2}, DType::Float32, Lifetime::Input, "x");

    EXPECT_FALSE(graph.has_refusal());
    EXPECT_FALSE(graph.without_nodes().has_refusal());

    graph.mark_refused();

    EXPECT_TRUE(graph.has_refusal());
    EXPECT_TRUE(graph.without_nodes().has_refusal());
}

/** @brief Tests that get() draws one identity per holder and that two holders never share one. */
TEST(Unit_Ir, StateIdentityDrawsOneIdentityPerHolder)
{
    StateIdentity first;
    StateIdentity second;

    EXPECT_EQ(first.peek(), K_NO_STATE);

    const StateID drawn = first.get();
    EXPECT_NE(drawn, K_NO_STATE);
    EXPECT_EQ(first.get(), drawn);
    EXPECT_EQ(first.peek(), drawn);

    EXPECT_NE(second.get(), K_NO_STATE);
    EXPECT_NE(second.get(), drawn);
}

/** @brief Tests that a copied holder is new state and never takes the source's identity. */
TEST(Unit_Ir, ACopiedStateIdentityDoesNotShareTheSourcesIdentity)
{
    StateIdentity source;
    const StateID drawn = source.get();

    StateIdentity copied(source);
    EXPECT_EQ(copied.peek(), K_NO_STATE);
    EXPECT_NE(copied.get(), drawn);

    StateIdentity assigned;
    const StateID replaced = assigned.get();
    assigned = source;
    EXPECT_EQ(assigned.peek(), K_NO_STATE);
    EXPECT_NE(assigned.get(), drawn);
    EXPECT_NE(assigned.get(), replaced);

    EXPECT_EQ(source.peek(), drawn);
}

/** @brief Tests that a move carries the identity to the target and leaves the source without. */
TEST(Unit_Ir, AMovedStateIdentityCarriesItsIdentityAway)
{
    StateIdentity source;
    const StateID drawn = source.get();

    StateIdentity moved(std::move(source));
    EXPECT_EQ(moved.peek(), drawn);
    EXPECT_EQ(source.peek(), K_NO_STATE);

    StateIdentity assigned;
    (void)assigned.get();
    assigned = std::move(moved);
    EXPECT_EQ(assigned.peek(), drawn);
    EXPECT_EQ(moved.peek(), K_NO_STATE);
}
