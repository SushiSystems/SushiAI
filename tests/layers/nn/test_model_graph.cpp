/**************************************************************************/
/* test_model_graph.cpp                                                   */
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
 * @file test_model_graph.cpp
 * @brief Tests ModelGraph: what add and output refuse, what a trace emits, and the state names.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/upsample2d.hpp>

using namespace SushiAI;
using NN::Conv2d;
using NN::ModelGraph;
using NN::NodeRef;
using NN::Trace;

static_assert(NN::TraceModule<ModelGraph>,
              "ModelGraph must be a TraceModule so it nests in a Sequential");
static_assert(!NN::Module<ModelGraph>,
              "ModelGraph must not trace from a builder alone, which states no mode");
static_assert(NN::Detail::HasStateWalk<ModelGraph>::value,
              "ModelGraph must be walked through its own walk_state");

namespace
{
    /** @brief Lists the operation of every node of a graph, in order. */
    using OpList = std::vector<decltype(Graph::Node::op)>;

    /** @brief Emits a ReLU and a GELU of its one input, as the ports "low" and "high". */
    class TwoPortNode final : public NN::Node
    {
        public:
            /** @brief Declares both ports over the per-sample shape @p sample. */
            explicit TwoPortNode(const Shape& sample)
                : TwoPortNode(sample, "low", "high")
            {

            }

            /** @brief Declares the ports @p first and @p second over @p sample. */
            TwoPortNode(const Shape& sample, std::string first, std::string second)
                : ports_{NN::Port{std::move(first), sample}, NN::Port{std::move(second), sample}}
            {

            }

            /** @brief Appends the ReLU, then the GELU, of the one input. */
            [[nodiscard]] Status forward(const Trace& trace, NN::ValueSpan inputs,
                                         std::vector<Graph::ValueID>& outputs) override
            {
                SA_TRY(NN::Detail::require_input_count(inputs, 1));

                outputs.push_back(trace.builder().relu(inputs[0]));
                outputs.push_back(trace.builder().gelu(inputs[0]));
                return Status();
            }

            /** @brief Returns one, the number of values the node reads. */
            [[nodiscard]] std::size_t input_count() const noexcept override
            {
                return 1;
            }

            /** @brief Visits nothing. */
            void walk_state(std::string_view, const NN::StateVisitor&) override
            {

            }

            /** @brief Lists "low" and "high". */
            [[nodiscard]] SushiRuntime::span<const NN::Port> ports() const noexcept override
            {
                return SushiRuntime::span<const NN::Port>(ports_.data(), ports_.size());
            }

        private:
            /** @brief Holds the two ports forward() appends to. */
            std::vector<NN::Port> ports_;
    };

    /** @brief Holds the branch-and-merge model and the refs its tests read. */
    struct BranchingModel
    {
        ModelGraph model; /**< stem, down, up, merge(up, stem), mix. */
        NodeRef stem;     /**< The [4, 4, 4] branch that skips ahead. */
        NodeRef up;       /**< The [4, 4, 8] branch that went down and back. */
        NodeRef mix;      /**< The [4, 4, 4] result after the merge. */
    };

    /** @brief Builds the branch and merge over an [8, 8, 3] sample, with no output published. */
    Result<BranchingModel> build_branching_model()
    {
        ModelGraph model(Shape{8, 8, 3});

        SA_TRY_ASSIGN(const NodeRef stem,
                      model.add("stem", Conv2d::same_padded(3, 4, 3, 2), model.input()));
        SA_TRY_ASSIGN(const NodeRef down,
                      model.add("down", Conv2d::same_padded(4, 8, 3, 2), stem));
        SA_TRY_ASSIGN(const NodeRef up, model.add("up", NN::Upsample2d{2}, down));

        const NodeRef both[] = {up, stem};
        const Shape shapes[] = {model.shape_of(up), model.shape_of(stem)};
        SA_TRY_ASSIGN(std::unique_ptr<NN::Concat> concat,
                      NN::Concat::create(NN::ShapeSpan(shapes, 2)));
        SA_TRY_ASSIGN(const NodeRef merged, model.add("merge", std::move(concat), both));

        SA_TRY_ASSIGN(const NodeRef mix,
                      model.add("mix", Conv2d::same_padded(12, 4, 1, 1), merged));

        return BranchingModel{std::move(model), stem, up, mix};
    }

    /** @brief Returns the operation of every node of @p graph, in order. */
    OpList ops_of(const Graph::Graph& graph)
    {
        OpList ops;
        for (const Graph::Node& node : graph.nodes())
            ops.push_back(node.op);

        return ops;
    }

    /** @brief Returns the declared parameters of @p model under @p prefix as "name:extents". */
    template<typename M>
    std::vector<std::string> parameter_names(M& model, std::string_view prefix)
    {
        std::vector<std::string> names;
        NN::for_each_parameter(model, prefix,
                               [&names](const std::string& name, NN::Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });

        return names;
    }

    /** @brief Appends a one-in one-out node that declares @p declared and traces a ReLU. */
    Result<NodeRef> add_relu_declared_as(ModelGraph& model, std::string name,
                                         const Shape& declared, NodeRef from)
    {
        std::unique_ptr<NN::Node> node = std::make_unique<NN::LayerNode>(
            std::make_unique<NN::LayerAdapter<NN::ReLU>>(NN::ReLU{}, declared));

        return model.add(std::move(name), std::move(node),
                         SushiRuntime::span<const NodeRef>(&from, 1));
    }
} // namespace

/** @brief Tests that the input ref is the input and a node's ref carries its index and port 0. */
TEST(Unit_ModelGraph, RefsNameTheInputAndTheNodesInTheOrderAdded)
{
    SA_ASSERT_OK_AND_ASSIGN(const BranchingModel built, build_branching_model());

    EXPECT_TRUE(built.model.input().is_input());
    EXPECT_EQ(built.model.input().port(), 0u);
    EXPECT_FALSE(built.stem.is_input());
    EXPECT_EQ(built.stem.node(), 0u);
    EXPECT_EQ(built.up.node(), 2u);
    EXPECT_EQ(built.mix.node(), 4u);
    EXPECT_EQ(built.mix.port(), 0u);
    EXPECT_NE(built.stem, built.up);
    EXPECT_EQ(built.model.size(), 5u);
}

/** @brief Tests that shape_of returns the input sample and each node's declared port shape. */
TEST(Unit_ModelGraph, ShapeOfReturnsTheDeclaredPerSampleShapes)
{
    SA_ASSERT_OK_AND_ASSIGN(const BranchingModel built, build_branching_model());

    EXPECT_EQ(built.model.shape_of(built.model.input()), (Shape{8, 8, 3}));
    EXPECT_EQ(built.model.shape_of(built.stem), (Shape{4, 4, 4}));
    EXPECT_EQ(built.model.shape_of(built.up), (Shape{4, 4, 8}));
    EXPECT_EQ(built.model.shape_of(built.mix), (Shape{4, 4, 4}));
    EXPECT_EQ(built.model.node(3).ports()[0].shape, (Shape{4, 4, 12}));
}

/**
 * @brief Tests that the branch and merge traces the op sequence of the same layers traced by
 *        hand, with one UPSAMPLE2D, one CONCAT_CHANNELS and a [2, 4, 4, 4] result.
 */
TEST(Unit_ModelGraph, BranchAndMergeTracesTheNodeSequenceOfTheHandWrittenTrace)
{
    SA_ASSERT_OK_AND_ASSIGN(BranchingModel built, build_branching_model());
    SA_EXPECT_OK(built.model.output("features", built.mix));

    Graph::Graph traced;
    const Graph::GraphBuilder traced_builder(traced);
    const Graph::ValueID traced_x = traced_builder.input(Shape{2, 8, 8, 3}, "x");
    const Graph::ValueID features =
        built.model.forward(Trace::evaluation(traced_builder), traced_x);

    Graph::Graph expected;
    const Graph::GraphBuilder builder(expected);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    Conv2d stem = Conv2d::same_padded(3, 4, 3, 2);
    Conv2d down = Conv2d::same_padded(4, 8, 3, 2);
    Conv2d mix = Conv2d::same_padded(12, 4, 1, 1);
    const Graph::ValueID p3 = stem.forward(builder, x);
    const Graph::ValueID up = NN::Upsample2d{2}.forward(builder, down.forward(builder, p3));
    const Graph::ValueID parts[] = {up, p3};
    const Graph::ValueID merged = builder.concat_channels(NN::ValueSpan(parts, 2));
    const Graph::ValueID expected_features = mix.forward(builder, merged);

    const OpList ops = ops_of(traced);
    EXPECT_EQ(ops, ops_of(expected));
    EXPECT_EQ(traced.value(features).shape, expected.value(expected_features).shape);

    EXPECT_EQ(traced.value(features).shape, (Shape{2, 4, 4, 4}));
    EXPECT_EQ(std::count(ops.begin(), ops.end(), Ops::K_UPSAMPLE2D.id), 1);
    EXPECT_EQ(std::count(ops.begin(), ops.end(), Ops::K_CONCAT_CHANNELS.id), 1);
}

/** @brief Tests that the merge is one CONCAT_CHANNELS reading the two branches in from order. */
TEST(Unit_ModelGraph, MergeEmitsOneConcatChannelsReadingBothBranches)
{
    SA_ASSERT_OK_AND_ASSIGN(BranchingModel built, build_branching_model());
    SA_EXPECT_OK(built.model.output("stem", built.stem));
    SA_EXPECT_OK(built.model.output("up", built.up));
    SA_EXPECT_OK(built.model.output("features", built.mix));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            built.model.trace(Trace::evaluation(builder), x));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID stem, outputs.at("stem"));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID up, outputs.at("up"));

    std::vector<const Graph::Node*> concats;
    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op == Ops::K_CONCAT_CHANNELS.id)
            concats.push_back(&node);
    }

    ASSERT_EQ(concats.size(), 1u);
    EXPECT_EQ(concats.front()->inputs, (std::vector<Graph::ValueID>{up, stem}));
    ASSERT_EQ(concats.front()->outputs.size(), 1u);
    EXPECT_EQ(graph.value(concats.front()->outputs.front()).shape, (Shape{2, 4, 4, 12}));
}

/** @brief Tests that trace returns every published output by name, in the order published. */
TEST(Unit_ModelGraph, TraceReturnsEveryPublishedOutputByName)
{
    SA_ASSERT_OK_AND_ASSIGN(BranchingModel built, build_branching_model());
    SA_EXPECT_OK(built.model.output("features", built.mix));
    SA_EXPECT_OK(built.model.output("skip", built.stem));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            built.model.trace(Trace::evaluation(builder), x));

    ASSERT_EQ(outputs.all().size(), 2u);
    EXPECT_EQ(outputs.all()[0].first, "features");
    EXPECT_EQ(outputs.all()[1].first, "skip");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID features, outputs.at("features"));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID skip, outputs.at("skip"));
    EXPECT_NE(features, skip);
    EXPECT_EQ(graph.value(features).shape, (Shape{2, 4, 4, 4}));
    EXPECT_EQ(graph.value(skip).shape, (Shape{2, 4, 4, 4}));
}

/** @brief Tests that forward returns the value trace publishes when one output is published. */
TEST(Unit_ModelGraph, ForwardReturnsTheSinglePublishedOutput)
{
    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef relu, model.add("act", NN::ReLU{}, model.input()));
    SA_EXPECT_OK(model.output("activated", relu));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    const std::size_t leaves = graph.node_count();
    const Graph::ValueID y = model.forward(Trace::evaluation(builder), x);

    ASSERT_EQ(graph.node_count(), leaves + 1u);
    EXPECT_EQ(graph.nodes()[leaves].op, Ops::K_RELU.id);
    EXPECT_EQ(graph.nodes()[leaves].outputs, (std::vector<Graph::ValueID>{y}));
    EXPECT_EQ(graph.nodes()[leaves].inputs, (std::vector<Graph::ValueID>{x}));
}

/** @brief Tests that forward refuses a model that publishes two outputs and names both. */
TEST(Unit_ModelGraph, ForwardFailsWhenTwoOutputsArePublished)
{
    SA_ASSERT_OK_AND_ASSIGN(BranchingModel built, build_branching_model());
    SA_EXPECT_OK(built.model.output("features", built.mix));
    SA_EXPECT_OK(built.model.output("skip", built.stem));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");

    const Trace trace = Trace::evaluation(builder);

    EXPECT_EQ(built.model.forward(trace, x), x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription,
                      "exactly one model output", "features", "skip");
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that output_shape is the single output's shape, and empty without exactly one. */
TEST(Unit_ModelGraph, OutputShapeIsTheShapeOfTheSinglePublishedOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(BranchingModel built, build_branching_model());
    EXPECT_EQ(built.model.output_shape(Shape{8, 8, 3}), Shape{});

    SA_EXPECT_OK(built.model.output("features", built.up));
    EXPECT_EQ(built.model.output_shape(Shape{8, 8, 3}), (Shape{4, 4, 8}));

    SA_EXPECT_OK(built.model.output("skip", built.stem));
    EXPECT_EQ(built.model.output_shape(Shape{8, 8, 3}), Shape{});
}

/** @brief Tests that a graph added to another is one node, traced and named under its name. */
TEST(Unit_ModelGraph, AGraphIsANodeOfAnotherGraph)
{
    ModelGraph inner(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef dense,
                            inner.add("dense", NN::Linear{6, 8}, inner.input()));
    SA_EXPECT_OK(inner.output("output", dense));

    ModelGraph outer(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef nested,
                            outer.add("inner", std::move(inner), outer.input()));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef activated, outer.add("act", NN::ReLU{}, nested));
    SA_EXPECT_OK(outer.output("output", activated));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            outer.trace(Trace::evaluation(builder), x));

    EXPECT_EQ(outer.shape_of(nested), (Shape{8}));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, outputs.only());
    EXPECT_EQ(graph.value(y).shape, (Shape{4, 8}));
    EXPECT_EQ(parameter_names(outer, ""), (std::vector<std::string>{
        "inner.dense.weight:[8, 6]", "inner.dense.bias:[8]"}));
}

/** @brief Tests that a nested graph with two outputs fails the outer trace under its name. */
TEST(Unit_ModelGraph, ANestedGraphThatPublishesTwoOutputsFailsTheOuterTrace)
{
    ModelGraph inner(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef relu, inner.add("act", NN::ReLU{}, inner.input()));
    SA_EXPECT_OK(inner.output("first", relu));
    SA_EXPECT_OK(inner.output("second", relu));

    ModelGraph outer(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef nested,
                            outer.add("inner", std::move(inner), outer.input()));
    SA_EXPECT_OK(outer.output("output", nested));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");

    SA_EXPECT_FAILURE(outer.trace(Trace::evaluation(builder), x),
                      StatusCode::InvalidDescription, "node 'inner'",
                      "exactly one model output", "first", "second");
}

/** @brief Tests that a Sequential and a LayerStack are each one node of a graph. */
TEST(Unit_ModelGraph, ASequentialAndALayerStackAreNodesOfAGraph)
{
    NN::LayerStack stack;
    stack.push(std::make_unique<NN::LayerAdapter<NN::Linear>>(NN::Linear{8, 4}, 4));
    stack.push(std::make_unique<NN::LayerAdapter<NN::GELU>>(NN::GELU{}, 4));

    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(
        const NodeRef head,
        model.add("head", NN::Sequential{NN::Linear{6, 8}, NN::ReLU{}}, model.input()));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef tail, model.add("tail", std::move(stack), head));
    SA_EXPECT_OK(model.output("output", tail));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            model.trace(Trace::evaluation(builder), x));

    EXPECT_EQ(model.shape_of(head), (Shape{8}));
    EXPECT_EQ(model.shape_of(tail), (Shape{4}));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, outputs.only());
    EXPECT_EQ(graph.value(y).shape, (Shape{4, 4}));
    EXPECT_EQ(parameter_names(model, ""), (std::vector<std::string>{
        "head.0.weight:[8, 6]", "head.0.bias:[8]", "tail.0.weight:[4, 8]", "tail.0.bias:[4]"}));
}

/** @brief Tests that the input can be published and traces to the input value itself. */
TEST(Unit_ModelGraph, TheInputCanBePublishedAsAnOutput)
{
    ModelGraph model(Shape{6});
    SA_EXPECT_OK(model.output("echo", model.input()));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    const std::size_t leaves = graph.node_count();

    EXPECT_EQ(model.forward(Trace::evaluation(builder), x), x);
    EXPECT_EQ(graph.node_count(), leaves);
}

/** @brief Tests that the walk names state "<prefix>.<node name>.<member>" in the order added. */
TEST(Unit_ModelGraph, WalkNamesStateByNodeNameInTheOrderAdded)
{
    SA_ASSERT_OK_AND_ASSIGN(BranchingModel built, build_branching_model());
    SA_EXPECT_OK(built.model.output("features", built.mix));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    SA_EXPECT_OK(built.model.trace(Trace::evaluation(builder), x));

    EXPECT_EQ(parameter_names(built.model, "model"), (std::vector<std::string>{
        "model.stem.weight:[4, 3, 3, 3]",
        "model.down.weight:[8, 3, 3, 4]",
        "model.mix.weight:[4, 1, 1, 12]"}));
    EXPECT_EQ(NN::buffer_count(built.model), 0u);
}

/**
 * @brief Tests that a training trace reaches a modal node with its sink, and that the walk
 *        names the node's parameters and buffers under the node's name.
 */
TEST(Unit_ModelGraph, ATrainingTraceReachesAModalNodeAndTheWalkNamesItsBuffers)
{
    ModelGraph model(Shape{4, 4, 3});
    SA_ASSERT_OK_AND_ASSIGN(
        const NodeRef norm, model.add("bn", NN::BatchNorm2d(3, NN::NormOptions{}), model.input()));
    SA_EXPECT_OK(model.output("normalised", norm));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 3}, "x");
    NN::BufferUpdateList list;
    SA_EXPECT_OK(model.trace(Trace::training(builder, list), x));

    const OpList ops = ops_of(graph);
    EXPECT_EQ(std::count(ops.begin(), ops.end(), Ops::K_BATCHNORM2D.id), 1);
    EXPECT_EQ(list.updates().size(), 2u);
    EXPECT_EQ(parameter_names(model, "model"), (std::vector<std::string>{
        "model.bn.weight:[3]", "model.bn.bias:[3]"}));

    std::vector<std::string> buffers;
    NN::for_each_buffer(model, "model", [&buffers](const std::string& name, NN::Buffer&)
    {
        buffers.push_back(name);
    });
    EXPECT_EQ(buffers, (std::vector<std::string>{
        "model.bn.running_mean", "model.bn.running_var"}));
}

/**
 * @brief Tests that a chain named by row index yields the names a LayerStack yields, which are
 *        the weight and bias of rows 0 and 2.
 */
TEST(Unit_ModelGraph, AChainNamedByIndexYieldsTheNamesOfALayerStack)
{
    const NN::Linear first{6, 8};
    const NN::Linear last{8, 4};

    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef row0, model.add("0", first, model.input()));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef row1, model.add("1", NN::GELU{}, row0));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef row2, model.add("2", last, row1));
    SA_EXPECT_OK(model.output("output", row2));

    NN::LayerStack stack;
    stack.push(std::make_unique<NN::LayerAdapter<NN::Linear>>(first, 8));
    stack.push(std::make_unique<NN::LayerAdapter<NN::GELU>>(NN::GELU{}, 8));
    stack.push(std::make_unique<NN::LayerAdapter<NN::Linear>>(last, 4));

    Graph::Graph graph_of_model;
    const Graph::GraphBuilder model_builder(graph_of_model);
    static_cast<void>(model.forward(Trace::evaluation(model_builder),
                                    model_builder.input(Shape{4, 6}, "x")));

    Graph::Graph graph_of_stack;
    const Graph::GraphBuilder stack_builder(graph_of_stack);
    static_cast<void>(stack.forward(stack_builder, stack_builder.input(Shape{4, 6}, "x")));

    const std::vector<std::string> names = parameter_names(model, "");
    EXPECT_EQ(names, parameter_names(stack, ""));
    EXPECT_EQ(ops_of(graph_of_model), ops_of(graph_of_stack));

    EXPECT_EQ(names, (std::vector<std::string>{
        "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]", "2.bias:[4]"}));
}

/** @brief Tests that a ModelGraph inside a Sequential is traced and named under its index. */
TEST(Unit_ModelGraph, NestsInASequentialUnderItsIndex)
{
    const NN::Linear linear{6, 8};
    ModelGraph inner(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef dense, inner.add("dense", linear, inner.input()));
    SA_EXPECT_OK(inner.output("output", dense));

    NN::Sequential<ModelGraph, NN::ReLU> model{std::move(inner), NN::ReLU{}};

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    const Graph::ValueID y = model.forward(Trace::evaluation(builder), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 8}));
    EXPECT_EQ(graph.nodes().back().op, Ops::K_RELU.id);
    EXPECT_EQ(parameter_names(model, ""), (std::vector<std::string>{
        "0.dense.weight:[8, 6]", "0.dense.bias:[8]"}));
}

/** @brief Tests that a second node under a taken name is refused and the graph keeps its size. */
TEST(Unit_ModelGraph, AddRejectsADuplicateName)
{
    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef first, model.add("act", NN::ReLU{}, model.input()));

    SA_EXPECT_FAILURE(model.add("act", NN::GELU{}, first), StatusCode::InvalidDescription,
                      "already has a node named 'act'", "index 0");
    EXPECT_EQ(model.size(), 1u);
}

/** @brief Tests that an empty node name is refused. */
TEST(Unit_ModelGraph, AddRejectsAnEmptyName)
{
    ModelGraph model(Shape{6});

    SA_EXPECT_FAILURE(model.add("", NN::ReLU{}, model.input()),
                      StatusCode::InvalidDescription, "needs a name");
    EXPECT_EQ(model.size(), 0u);
}

/** @brief Tests that a node name containing '.' is refused by both adds. */
TEST(Unit_ModelGraph, AddRejectsANameWithADot)
{
    ModelGraph model(Shape{6});

    SA_EXPECT_FAILURE(model.add("a.conv", NN::ReLU{}, model.input()),
                      StatusCode::InvalidDescription, "'a.conv'", "contains '.'");
    SA_EXPECT_FAILURE(add_relu_declared_as(model, "b.act", Shape{6}, model.input()),
                      StatusCode::InvalidDescription, "'b.act'", "contains '.'");
    EXPECT_EQ(model.size(), 0u);
}

/** @brief Tests that a refused name is reported before the layer is asked for its shape. */
TEST(Unit_ModelGraph, AddReportsARefusedNameBeforeTheLayerSeesItsSource)
{
    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef first, model.add("act", NN::ReLU{}, model.input()));

    SA_EXPECT_FAILURE(model.add("act", Conv2d::same_padded(3, 4, 3, 2), first),
                      StatusCode::InvalidDescription, "already has a node named 'act'");
    SA_EXPECT_FAILURE(model.add("", Conv2d::same_padded(3, 4, 3, 2), first),
                      StatusCode::InvalidDescription, "needs a name");
    EXPECT_EQ(model.size(), 1u);
}

/** @brief Tests that a layer that refuses its source's shape fails add under the node's name. */
TEST(Unit_ModelGraph, AddReturnsTheFailureOfALayerThatRefusesItsSource)
{
    ModelGraph model(Shape{6});

    SA_EXPECT_FAILURE(model.add("stem", Conv2d::same_padded(3, 4, 3, 2), model.input()),
                      StatusCode::InvalidDescription, "node 'stem'", "Conv2d(3, 4)", "[6]");
    EXPECT_EQ(model.size(), 0u);
}

/** @brief Tests that a source naming a node the graph does not hold is refused by both adds. */
TEST(Unit_ModelGraph, AddRejectsAnUnknownSource)
{
    ModelGraph other(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef row0, other.add("0", NN::ReLU{}, other.input()));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef row1, other.add("1", NN::ReLU{}, row0));

    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef only, model.add("only", NN::ReLU{}, model.input()));
    static_cast<void>(only);

    SA_EXPECT_FAILURE(model.add("late", NN::ReLU{}, row1), StatusCode::InvalidDescription,
                      "node 'late'", "source node 1", "holds 1 nodes");
    SA_EXPECT_FAILURE(add_relu_declared_as(model, "later", Shape{6}, row1),
                      StatusCode::InvalidDescription, "node 'later'", "source node 1");
    EXPECT_EQ(model.size(), 1u);
}

/** @brief Tests that a source naming a port its node does not have is refused. */
TEST(Unit_ModelGraph, AddRejectsASourcePortTheNodeDoesNotHave)
{
    ModelGraph other(Shape{6});
    const NodeRef other_input = other.input();
    SA_ASSERT_OK_AND_ASSIGN(
        const NodeRef pair,
        other.add("pair", std::make_unique<TwoPortNode>(Shape{6}),
                  SushiRuntime::span<const NodeRef>(&other_input, 1)));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef high, other.port(pair, "high"));

    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef only, model.add("only", NN::ReLU{}, model.input()));
    static_cast<void>(only);

    SA_EXPECT_FAILURE(model.add("late", NN::ReLU{}, high), StatusCode::InvalidDescription,
                      "node 'late'", "'only' has 1 ports", "port 1");
}

/** @brief Tests that port selects a node's port by name and a later node can read it. */
TEST(Unit_ModelGraph, PortSelectsANamedPortThatALaterNodeReads)
{
    ModelGraph model(Shape{6});
    const NodeRef input = model.input();
    SA_ASSERT_OK_AND_ASSIGN(
        const NodeRef pair,
        model.add("pair", std::make_unique<TwoPortNode>(Shape{6}),
                  SushiRuntime::span<const NodeRef>(&input, 1)));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef high, model.port(pair, "high"));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef low, model.port(high, "low"));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef after, model.add("after", NN::ReLU{}, high));
    SA_EXPECT_OK(model.output("output", after));

    EXPECT_EQ(high.node(), 0u);
    EXPECT_EQ(high.port(), 1u);
    EXPECT_EQ(low, pair);

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    const std::size_t leaves = graph.node_count();
    static_cast<void>(model.forward(Trace::evaluation(builder), x));

    ASSERT_EQ(graph.node_count(), leaves + 3u);
    EXPECT_EQ(graph.nodes()[leaves + 1u].op, Ops::K_GELU.id);
    EXPECT_EQ(graph.nodes()[leaves + 2u].inputs, graph.nodes()[leaves + 1u].outputs);
}

/** @brief Tests that port lists the node's ports for an unknown name and refuses the input. */
TEST(Unit_ModelGraph, PortRejectsAnUnknownNameAndTheInput)
{
    ModelGraph model(Shape{6});
    const NodeRef input = model.input();
    SA_ASSERT_OK_AND_ASSIGN(
        const NodeRef pair,
        model.add("pair", std::make_unique<TwoPortNode>(Shape{6}),
                  SushiRuntime::span<const NodeRef>(&input, 1)));

    SA_EXPECT_FAILURE(model.port(pair, "middle"), StatusCode::InvalidDescription,
                      "node 'pair'", "'middle'", "low, high");
    SA_EXPECT_FAILURE(model.port(input, "low"), StatusCode::InvalidDescription,
                      "the model input has no port named 'low'");
}

/** @brief Tests that output_ports publishes every port of a node under the port's own name. */
TEST(Unit_ModelGraph, OutputPortsPublishesEveryPortUnderItsOwnName)
{
    ModelGraph model(Shape{6});
    const NodeRef input = model.input();
    SA_ASSERT_OK_AND_ASSIGN(
        const NodeRef pair,
        model.add("pair", std::make_unique<TwoPortNode>(Shape{6}),
                  SushiRuntime::span<const NodeRef>(&input, 1)));
    SA_EXPECT_OK(model.output_ports(pair));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    const std::size_t leaves = graph.node_count();
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            model.trace(Trace::evaluation(builder), x));

    ASSERT_EQ(outputs.all().size(), 2u);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID low, outputs.at("low"));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID high, outputs.at("high"));
    EXPECT_EQ(graph.nodes()[leaves].outputs, (std::vector<Graph::ValueID>{low}));
    EXPECT_EQ(graph.nodes()[leaves + 1u].outputs, (std::vector<Graph::ValueID>{high}));
}

/** @brief Tests that output_ports publishes nothing when one of the port names is taken. */
TEST(Unit_ModelGraph, OutputPortsPublishesNothingWhenAPortNameIsTaken)
{
    ModelGraph model(Shape{6});
    const NodeRef input = model.input();
    SA_ASSERT_OK_AND_ASSIGN(
        const NodeRef pair,
        model.add("pair", std::make_unique<TwoPortNode>(Shape{6}),
                  SushiRuntime::span<const NodeRef>(&input, 1)));
    SA_EXPECT_OK(model.output("high", input));

    SA_EXPECT_FAILURE(model.output_ports(pair), StatusCode::InvalidDescription,
                      "port 'high' of node 'pair'");
    SA_EXPECT_FAILURE(model.output_ports(input), StatusCode::InvalidDescription,
                      "the model input has no ports to publish");

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            model.trace(Trace::evaluation(builder), x));
    EXPECT_EQ(outputs.all().size(), 1u);
}

/** @brief Tests that output refuses a taken name, an empty name and an unknown source. */
TEST(Unit_ModelGraph, OutputRejectsATakenNameAnEmptyNameAndAnUnknownSource)
{
    ModelGraph other(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef foreign, other.add("0", NN::ReLU{}, other.input()));

    ModelGraph model(Shape{6});
    SA_EXPECT_OK(model.output("echo", model.input()));

    SA_EXPECT_FAILURE(model.output("echo", model.input()), StatusCode::InvalidDescription,
                      "already publishes an output named 'echo'");
    SA_EXPECT_FAILURE(model.output("", model.input()), StatusCode::InvalidDescription,
                      "needs a name");
    SA_EXPECT_FAILURE(model.output("far", foreign), StatusCode::InvalidDescription,
                      "output 'far'", "source node 0", "holds 0 nodes");
}

/** @brief Tests that a node whose traced shape differs from its declared port fails the trace. */
TEST(Unit_ModelGraph, TraceRejectsADeclaredShapeThatDisagreesWithTheTracedOne)
{
    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef wrong,
                            add_relu_declared_as(model, "narrow", Shape{5}, model.input()));
    SA_EXPECT_OK(model.output("output", wrong));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");

    SA_EXPECT_FAILURE(model.trace(Trace::evaluation(builder), x),
                      StatusCode::InvalidDescription, "node 'narrow'", "port 'output'", "[5]",
                      "[4, 6]");
}

/** @brief Tests that an input whose sample shape is not the declared one fails before tracing. */
TEST(Unit_ModelGraph, TraceRejectsAnInputOfAnotherSampleShape)
{
    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef relu, model.add("act", NN::ReLU{}, model.input()));
    SA_EXPECT_OK(model.output("output", relu));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{4, 7}, "wide");
    const Graph::ValueID flat = builder.input(Shape{6}, "flat");
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_FAILURE(model.trace(Trace::evaluation(builder), wide),
                      StatusCode::InvalidDescription, "the model input", "[6]", "[4, 7]");
    SA_EXPECT_FAILURE(model.trace(Trace::evaluation(builder), flat),
                      StatusCode::InvalidDescription, "the model input", "[6]");
    EXPECT_EQ(graph.node_count(), leaves);
}

/** @brief Tests that a model with no published output fails the trace and emits no node. */
TEST(Unit_ModelGraph, TraceRejectsAModelThatPublishesNoOutput)
{
    ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef relu, model.add("act", NN::ReLU{}, model.input()));
    static_cast<void>(relu);

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "x");
    const std::size_t leaves = graph.node_count();

    SA_EXPECT_FAILURE(model.trace(Trace::evaluation(builder), x),
                      StatusCode::InvalidDescription, "publishes no output");
    EXPECT_EQ(graph.node_count(), leaves);
}

/** @brief Tests that add refuses a count of sources other than the node's input count. */
TEST(Unit_ModelGraph, AddRejectsASourceCountTheNodeDoesNotRead)
{
    ModelGraph model(Shape{8, 8, 3});
    const NodeRef input = model.input();
    const NodeRef twice[] = {input, input};
    const Shape shapes[] = {Shape{8, 8, 3}, Shape{8, 8, 3}};
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<NN::Concat> concat,
                            NN::Concat::create(NN::ShapeSpan(shapes, 2)));

    SA_EXPECT_FAILURE(
        model.add("merge", std::move(concat), SushiRuntime::span<const NodeRef>(&input, 1)),
        StatusCode::InvalidDescription, "node 'merge'", "reads 2 values", "given is 1");
    SA_EXPECT_FAILURE(model.add("split", std::make_unique<TwoPortNode>(Shape{8, 8, 3}), twice),
                      StatusCode::InvalidDescription, "node 'split'", "reads 1 values",
                      "given is 2");
    EXPECT_EQ(model.size(), 0u);
}

/**
 * @brief Tests that a moved graph resolves the refs made before the move, and that the graph
 *        moved from is empty and traces a node added to it afterwards.
 */
TEST(Unit_ModelGraph, AMovedGraphKeepsItsRefsAndLeavesAnEmptyUsableGraph)
{
    SA_ASSERT_OK_AND_ASSIGN(BranchingModel built, build_branching_model());
    ModelGraph moved = std::move(built.model);

    EXPECT_EQ(moved.size(), 5u);
    EXPECT_EQ(moved.shape_of(built.up), (Shape{4, 4, 8}));
    SA_EXPECT_OK(moved.output("features", built.mix));

    ASSERT_EQ(built.model.size(), 0u);
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef relu,
                            built.model.add("act", NN::ReLU{}, built.model.input()));
    SA_EXPECT_OK(built.model.output("activated", relu));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    const std::size_t leaves = graph.node_count();
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            built.model.trace(Trace::evaluation(builder), x));

    ASSERT_EQ(graph.node_count(), leaves + 1u);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID activated, outputs.at("activated"));
    EXPECT_EQ(graph.nodes()[leaves].outputs, (std::vector<Graph::ValueID>{activated}));
}

/** @brief Tests that shape_of and node stop the process on a ref or an index out of range. */
TEST(Unit_ModelGraphDeathTest, AccessorsAreFatalOutOfRange)
{
    ModelGraph other(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef foreign, other.add("0", NN::ReLU{}, other.input()));
    const ModelGraph model(Shape{6});

    SA_EXPECT_FATAL(static_cast<void>(model.shape_of(foreign)), "shape_of", "source node 0");
    SA_EXPECT_FATAL(static_cast<void>(model.node(0)), "ModelGraph::node", "node 0");
}

/** @brief Tests that adding a null node stops the process and names the node. */
TEST(Unit_ModelGraphDeathTest, AddIsFatalOnANullNode)
{
    ModelGraph model(Shape{6});
    const NodeRef input = model.input();

    SA_EXPECT_FATAL(
        static_cast<void>(model.add("ghost", std::unique_ptr<NN::Node>(),
                                    SushiRuntime::span<const NodeRef>(&input, 1))),
        "null node", "'ghost'");
}

/** @brief Tests that adding a node with two ports of one name stops the process and names it. */
TEST(Unit_ModelGraphDeathTest, AddIsFatalOnANodeWithTwoPortsOfOneName)
{
    ModelGraph model(Shape{6});
    const NodeRef input = model.input();

    SA_EXPECT_FATAL(
        static_cast<void>(model.add("pair", std::make_unique<TwoPortNode>(Shape{6}, "low", "low"),
                                    SushiRuntime::span<const NodeRef>(&input, 1))),
        "'pair'", "two ports named 'low'");
}

namespace
{
    /** @brief Traces a ReLU, or a GELU in its incremental form, and notes its sample. */
    struct SliceProbe
    {
        /** @brief Points at where forward_incremental() writes its sample; not owned. */
        Shape* seen = nullptr;

        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Traces a ReLU of @p x. */
        Graph::ValueID forward(const Trace& trace, Graph::ValueID x)
        {
            return trace.builder().relu(x);
        }

        /** @brief Traces a GELU of @p x and writes @p sample to seen. */
        Graph::ValueID forward_incremental(const Trace& trace, Graph::ValueID x,
                                           const Shape& sample)
        {
            *seen = sample;
            return trace.builder().gelu(x);
        }

        /** @brief Returns @p input. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    static_assert(NN::IncrementalModule<SliceProbe>, "SliceProbe must have an incremental form");
    static_assert(!NN::IncrementalModule<NN::ReLU>, "ReLU must have no incremental form");
    static_assert(NN::IncrementalModule<ModelGraph>,
                  "ModelGraph must nest as a node that has an incremental form");
} // namespace

/** @brief Tests that the incremental walk calls the incremental form with the node's sample. */
TEST(Unit_ModelGraph, TraceIncrementalHandsEachLayerThePerSampleShapeItReads)
{
    Shape seen{};
    ModelGraph model(Shape{5, 6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef probe,
                            model.add("probe", SliceProbe{&seen}, model.input()));
    SA_EXPECT_OK(model.output("y", probe));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{1, 1, 6}, "slice");
    const std::size_t leaves = graph.node_count();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    SA_ASSERT_OK_AND_ASSIGN(
        const NN::ModelOutputs outputs,
        model.trace_incremental(Trace::carrying(builder, fed, carried), slice));

    EXPECT_EQ(seen, (Shape{5, 6}));
    ASSERT_EQ(graph.node_count(), leaves + 1u);
    EXPECT_EQ(graph.nodes()[leaves].op, Ops::K_GELU.id);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, outputs.at("y"));
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, 6}));
}

/** @brief Tests that the ordinary walk still refuses the slice the incremental walk takes. */
TEST(Unit_ModelGraph, TraceStillChecksThePerSampleShapeTheIncrementalWalkSkips)
{
    Shape seen{};
    ModelGraph model(Shape{5, 6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef probe,
                            model.add("probe", SliceProbe{&seen}, model.input()));
    SA_EXPECT_OK(model.output("y", probe));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{1, 1, 6}, "slice");

    SA_EXPECT_FAILURE(model.trace(Trace::evaluation(builder), slice),
                      StatusCode::InvalidDescription, "[5, 6]", "[1, 1, 6]");
}

/** @brief Tests that a node without an incremental form is refused by name. */
TEST(Unit_ModelGraph, TraceIncrementalNamesTheNodeThatHasNoIncrementalForm)
{
    Shape seen{};
    ModelGraph model(Shape{5, 6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef probe,
                            model.add("probe", SliceProbe{&seen}, model.input()));
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef act, model.add("act", NN::ReLU{}, probe));
    SA_EXPECT_OK(model.output("y", act));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{1, 1, 6}, "slice");
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    SA_EXPECT_FAILURE(model.trace_incremental(Trace::carrying(builder, fed, carried), slice),
                      StatusCode::Unsupported, "node 'act'", "no incremental form");
}

/** @brief Tests that the incremental walk is refused in a trace that carries nothing. */
TEST(Unit_ModelGraph, TraceIncrementalNeedsACarryingTrace)
{
    Shape seen{};
    ModelGraph model(Shape{5, 6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef probe,
                            model.add("probe", SliceProbe{&seen}, model.input()));
    SA_EXPECT_OK(model.output("y", probe));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{1, 1, 6}, "slice");
    const std::size_t nodes = graph.node_count();

    SA_EXPECT_FAILURE(model.trace_incremental(Trace::evaluation(builder), slice),
                      StatusCode::InvalidDescription, "keeps nothing between runs");
    EXPECT_EQ(graph.node_count(), nodes);
}

/** @brief Tests that a nested graph traces its own nodes in their incremental form. */
TEST(Unit_ModelGraph, ANestedGraphTracesItsIncrementalForm)
{
    Shape seen{};
    ModelGraph inner(Shape{5, 6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef probe,
                            inner.add("probe", SliceProbe{&seen}, inner.input()));
    SA_EXPECT_OK(inner.output("y", probe));

    ModelGraph outer(Shape{5, 6});
    SA_ASSERT_OK_AND_ASSIGN(const NodeRef nested,
                            outer.add("inner", std::move(inner), outer.input()));
    SA_EXPECT_OK(outer.output("y", nested));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{1, 1, 6}, "slice");
    const std::size_t leaves = graph.node_count();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    SA_ASSERT_OK_AND_ASSIGN(
        const NN::ModelOutputs outputs,
        outer.trace_incremental(Trace::carrying(builder, fed, carried), slice));
    static_cast<void>(outputs);

    EXPECT_EQ(seen, (Shape{5, 6}));
    ASSERT_EQ(graph.node_count(), leaves + 1u);
    EXPECT_EQ(graph.nodes()[leaves].op, Ops::K_GELU.id);
}
