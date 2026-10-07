/**************************************************************************/
/* test_node.cpp                                                          */
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
 * @file test_node.cpp
 * @brief Tests ModuleNode, LayerNode and the buffer check LayerAdapter makes at compile time.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "../../common/test_status.hpp"

using namespace SushiAI;
using NN::LayerAdapter;
using NN::LayerNode;
using NN::Linear;
using NN::ModuleNode;
using NN::ReLU;
using NN::Sequential;
using NN::Trace;

namespace
{
    /** @brief Lists one buffer and defines it when traced; stands in for a layer with state. */
    struct BufferedStub
    {
        /** @brief Holds the buffer the stub lists. */
        NN::Buffer running{};

        /** @brief Lists the buffer. */
        static constexpr auto fields()
        {
            return std::tuple{NN::buffer("running", &BufferedStub::running)};
        }

        /** @brief Defines the buffer as [4] zeros and returns its input. */
        Graph::ValueId forward(const Graph::GraphBuilder&, Graph::ValueId x)
        {
            const Status defined = running.define(Shape{4}, 0.0);
            static_cast<void>(defined);
            return x;
        }

        /** @brief Returns @p input, which the stub keeps. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    /** @brief Lists a child that lists a buffer; stands in for a block around such a layer. */
    struct BufferedParentStub
    {
        /** @brief Holds the child whose fields() lists the buffer. */
        BufferedStub inner{};

        /** @brief Lists the child. */
        static constexpr auto fields()
        {
            return std::tuple{NN::child("inner", &BufferedParentStub::inner)};
        }

        /** @brief Traces the child. */
        Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x)
        {
            return inner.forward(builder, x);
        }
    };

    /** @brief Counts which of its two forms a trace asked for; stands in for a modal layer. */
    struct ModalStub
    {
        /** @brief Counts the calls of trace_training. */
        int training_calls = 0;

        /** @brief Counts the calls of trace_evaluation. */
        int evaluation_calls = 0;

        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Returns its input and counts the call. */
        Graph::ValueId trace_training(const Trace&, Graph::ValueId x)
        {
            ++training_calls;
            return x;
        }

        /** @brief Returns its input and counts the call. */
        Graph::ValueId trace_evaluation(const Trace&, Graph::ValueId x)
        {
            ++evaluation_calls;
            return x;
        }

        /** @brief Returns @p input, which the stub keeps. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    /** @brief Records the mode of every trace it is given; stands in for a TraceModule. */
    struct TraceStub
    {
        /** @brief Holds the mode of each forward call, in call order. */
        std::vector<NN::TraceMode> modes{};

        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Returns its input and records the mode of @p trace. */
        Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
        {
            modes.push_back(trace.mode());
            return x;
        }

        /** @brief Returns @p input, which the stub keeps. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    /** @brief Has fields() and a forward and no output_shape; ModuleNode must refuse it. */
    struct ShapelessStub
    {
        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Returns its input. */
        Graph::ValueId forward(const Graph::GraphBuilder&, Graph::ValueId x)
        {
            return x;
        }
    };

    /** @brief Refuses every input: records one failure on the trace and returns its input. */
    struct RefusingStub
    {
        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Records an InvalidDescription that names the stub and emits nothing. */
        Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
        {
            trace.fail(Status::from_message(StatusCode::InvalidDescription, []()
            {
                return std::string("RefusingStub refuses its input");
            }));
            return x;
        }

        /** @brief Returns @p input, which the stub keeps. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    /** @brief Records an OutOfMemory on the trace, then throws an Error about something else. */
    struct RefusingThenThrowingStub
    {
        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Records the cause, then throws as a later emitter would on a fallback value. */
        Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
        {
            trace.fail(Status::from_message(StatusCode::OutOfMemory, []()
            {
                return std::string("the stub's recorded cause");
            }));
            SA_THROW_IF(x != Graph::NO_VALUE, "the stub's later throw over value {}", x);
            return x;
        }

        /** @brief Returns @p input, which the stub keeps. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    /** @brief Builds the node of @p layer over @p input and expects create() to succeed. */
    template<typename L>
    std::unique_ptr<ModuleNode<L>> node_of(L layer, const Shape& input)
    {
        Result<std::unique_ptr<ModuleNode<L>>> made =
            ModuleNode<L>::create(std::move(layer), input);
        EXPECT_TRUE(Test::status_is_ok(made.status()));
        return std::move(made).value();
    }

    /** @brief Views @p ids as the inputs of one forward call. */
    NN::ValueSpan span_of(const std::vector<Graph::ValueId>& ids)
    {
        return NN::ValueSpan(ids.data(), ids.size());
    }

    /** @brief Collects the name of every parameter and buffer @p node walks under @p prefix. */
    std::vector<std::string> state_names(NN::Node& node, std::string_view prefix)
    {
        std::vector<std::string> names;
        const auto on_parameter = [&names](const std::string& name, NN::Parameter&)
        {
            names.push_back(name);
        };
        const auto on_buffer = [&names](const std::string& name, NN::Buffer&)
        {
            names.push_back(name);
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        node.walk_state(prefix, visitor);
        return names;
    }

    /** @brief Builds the conv2d layer of a model-file row with 5 filters and a 3x3 kernel. */
    std::unique_ptr<NN::ILayer> make_test_conv2d(const Shape& input)
    {
        const Config::Value attributes =
            Config::parse_json(R"({"filters": 5, "kernel": 3})", "<test>");
        return NN::make_conv2d(input, attributes, "layer 0 'conv2d'");
    }
} // namespace

// LayerAdapter asserts on this predicate, so a true case here is a LayerAdapter compile error.
static_assert(NN::Detail::HoldsBufferField<BufferedStub>::value,
              "a type whose fields() lists a buffer must be refused by LayerAdapter");
static_assert(NN::Detail::HoldsBufferField<BufferedParentStub>::value,
              "a type whose child lists a buffer must be refused by LayerAdapter");
static_assert(NN::Detail::HoldsBufferField<Sequential<Linear, BufferedStub>>::value,
              "a Sequential holding a buffered layer must be refused by LayerAdapter");
static_assert(NN::Detail::HoldsBufferField<NN::ModuleList<BufferedStub>>::value,
              "a ModuleList of buffered layers must be refused by LayerAdapter");
static_assert(!NN::Detail::HoldsBufferField<Linear>::value,
              "a type that lists parameters only must be accepted by LayerAdapter");
static_assert(!NN::Detail::HoldsBufferField<Sequential<Linear, ReLU>>::value,
              "a Sequential of parameter-only layers must be accepted by LayerAdapter");

// ModuleNode asserts on this predicate, so a false case here is a ModuleNode compile error.
static_assert(NN::Detail::HasOutputShapeOfInput<Linear>::value,
              "a layer with output_shape(const Shape&) must be accepted by ModuleNode");
static_assert(NN::Detail::HasOutputShapeOfInput<TraceStub>::value,
              "a TraceModule with output_shape(const Shape&) must be accepted by ModuleNode");
static_assert(!NN::Detail::HasOutputShapeOfInput<ShapelessStub>::value,
              "a layer without output_shape(const Shape&) must be refused by ModuleNode");

/** @brief Tests that a ModuleNode declares one port named "output" with the layer's shape. */
TEST(Unit_Node, ModuleNodeDeclaresOnePortWithTheLayersOutputShape)
{
    const std::unique_ptr<ModuleNode<Linear>> node = node_of(Linear{6, 8}, Shape{6});

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, "output");
    EXPECT_EQ(node->ports()[0].shape, (Shape{8}));
}

/** @brief Tests that a ModuleNode and a LayerNode each state that they read one value. */
TEST(Unit_Node, ALayerNodeOfEitherKindReadsOneValue)
{
    const std::unique_ptr<ModuleNode<Linear>> module = node_of(Linear{6, 8}, Shape{6});
    const LayerNode layer(std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8));

    EXPECT_EQ(module->input_count(), 1u);
    EXPECT_EQ(layer.input_count(), 1u);
}

/** @brief Tests that create() returns the failure of a shape the layer's output_shape refuses. */
TEST(Unit_Node, ModuleNodeCreateReturnsTheFailureOfAShapeTheLayerRefuses)
{
    SA_EXPECT_FAILURE(ModuleNode<Linear>::create(Linear{6, 8}, Shape{5}),
                      StatusCode::InvalidDescription, "Linear(6, 8)", "[5]", "[6]");
}

/** @brief Tests that a layer's thrown refusal of a traced value comes back as a failure. */
TEST(Unit_Node, ModuleNodeForwardReturnsTheFailureOfALayerThatThrows)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId narrow = builder.input(Shape{2, 5}, "narrow");

    const std::unique_ptr<ModuleNode<Linear>> node = node_of(Linear{6, 8}, Shape{6});
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_FAILURE(node->forward(Trace::evaluation(builder), span_of({narrow}), outputs),
                      StatusCode::InvalidDescription, "Linear(6, 8)", "[2, 5]");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that a failure the layer recorded on the trace comes back and appends nothing. */
TEST(Unit_Node, ModuleNodeForwardReturnsTheFailureALayerRecorded)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");

    const std::unique_ptr<ModuleNode<RefusingStub>> node = node_of(RefusingStub{}, Shape{6});
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_FAILURE(node->forward(Trace::evaluation(builder), span_of({x}), outputs),
                      StatusCode::InvalidDescription, "RefusingStub refuses");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that a failure a layer recorded outranks an Error the layer threw after it. */
TEST(Unit_Node, ModuleNodeForwardReturnsTheRecordedFailureBeforeALaterThrow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");

    const std::unique_ptr<ModuleNode<RefusingThenThrowingStub>> node =
        node_of(RefusingThenThrowingStub{}, Shape{6});
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_FAILURE(node->forward(Trace::evaluation(builder), span_of({x}), outputs),
                      StatusCode::OutOfMemory, "the stub's recorded cause");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that ModuleNode::forward appends the layer's result after what is there. */
TEST(Unit_Node, ModuleNodeForwardAppendsTheLayersResult)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");
    const Trace trace = Trace::evaluation(builder);

    const std::unique_ptr<ModuleNode<Linear>> node = node_of(Linear{6, 8}, Shape{6});
    std::vector<Graph::ValueId> outputs{x};
    SA_EXPECT_OK(node->forward(trace, span_of({x}), outputs));

    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0], x);
    EXPECT_EQ(graph.value(outputs[1]).shape, (Shape{2, 8}));
}

/** @brief Tests that a ModuleNode traces a modal layer in the form the trace's mode names. */
TEST(Unit_Node, ModuleNodeTracesAModalLayerInTheModeOfTheTrace)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");

    const std::unique_ptr<ModuleNode<ModalStub>> node = node_of(ModalStub{}, Shape{6});
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_OK(node->forward(Trace::training(builder, NN::discard_updates()), span_of({x}),
                              outputs));
    EXPECT_EQ(node->layer().training_calls, 1);
    EXPECT_EQ(node->layer().evaluation_calls, 0);

    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({x}), outputs));
    EXPECT_EQ(node->layer().training_calls, 1);
    EXPECT_EQ(node->layer().evaluation_calls, 1);
}

/** @brief Tests that a ModuleNode gives a TraceModule the trace itself, mode included. */
TEST(Unit_Node, ModuleNodeHandsTheTraceToATraceModule)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");

    const std::unique_ptr<ModuleNode<TraceStub>> node = node_of(TraceStub{}, Shape{6});
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_OK(node->forward(Trace::training(builder, NN::discard_updates()), span_of({x}),
                              outputs));
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({x}), outputs));

    EXPECT_EQ(node->layer().modes, (std::vector<NN::TraceMode>{NN::TraceMode::Training,
                                                              NN::TraceMode::Evaluation}));
    EXPECT_EQ(outputs, (std::vector<Graph::ValueId>{x, x}));
}

/** @brief Tests that a ModuleNode walks the layer's parameters under the given prefix. */
TEST(Unit_Node, ModuleNodeWalksTheLayersParametersUnderThePrefix)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");

    const std::unique_ptr<ModuleNode<Linear>> node = node_of(Linear{6, 8}, Shape{6});
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({x}), outputs));

    EXPECT_EQ(state_names(*node, "stem"), (std::vector<std::string>{"stem.weight", "stem.bias"}));
}

/** @brief Tests that a ModuleNode walk reaches a buffer, which a LayerAdapter walk cannot. */
TEST(Unit_Node, ModuleNodeWalksABufferOfTheLayer)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");

    const std::unique_ptr<ModuleNode<BufferedStub>> node = node_of(BufferedStub{}, Shape{4});
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({x}), outputs));

    EXPECT_EQ(state_names(*node, "bn"), (std::vector<std::string>{"bn.running"}));
}

/** @brief Tests that a ModuleNode given two inputs fails and neither traces nor appends. */
TEST(Unit_Node, ModuleNodeRejectsTwoInputs)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");
    const std::size_t nodes_before = graph.node_count();

    const std::unique_ptr<ModuleNode<Linear>> node = node_of(Linear{6, 8}, Shape{6});
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_FAILURE(node->forward(Trace::evaluation(builder), span_of({x, x}), outputs),
                      StatusCode::InvalidDescription, "input count is 1", "given is 2");
    EXPECT_TRUE(outputs.empty());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that a LayerNode over make_conv2d reports the layer's per-sample shape. */
TEST(Unit_Node, LayerNodeReportsTheShapeOfTheLayerItWraps)
{
    std::unique_ptr<NN::ILayer> layer = make_test_conv2d(Shape{8, 8, 2});
    const Shape layer_shape = layer->output_shape();

    const LayerNode node(std::move(layer));

    ASSERT_EQ(node.ports().size(), 1u);
    EXPECT_EQ(node.ports()[0].name, "output");
    EXPECT_EQ(node.ports()[0].shape, layer_shape);
    EXPECT_EQ(node.ports()[0].shape, (Shape{6, 6, 5}));
}

/** @brief Tests that LayerNode::forward appends a value shaped like its port with a batch axis. */
TEST(Unit_Node, LayerNodeForwardAppendsAValueShapedLikeItsPort)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 2}, "x");

    LayerNode node(make_test_conv2d(Shape{8, 8, 2}));
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node.forward(Trace::evaluation(builder), span_of({x}), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 6, 6, 5}));
}

/** @brief Tests that a run-time layer's thrown refusal of a value comes back as a failure. */
TEST(Unit_Node, LayerNodeForwardReturnsTheFailureOfALayerThatThrows)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId narrow = builder.input(Shape{2, 5}, "narrow");

    LayerNode node(std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8));
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_FAILURE(node.forward(Trace::evaluation(builder), span_of({narrow}), outputs),
                      StatusCode::InvalidDescription, "Linear(6, 8)", "[2, 5]");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that a LayerNode walks the layer's declared parameters under the prefix. */
TEST(Unit_Node, LayerNodeWalksTheLayersParametersUnderThePrefix)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");

    LayerNode node(std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8));
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node.forward(Trace::evaluation(builder), span_of({x}), outputs));

    EXPECT_EQ(state_names(node, "0"), (std::vector<std::string>{"0.weight", "0.bias"}));
}

/** @brief Tests that a LayerNode given two inputs fails and neither traces nor appends. */
TEST(Unit_Node, LayerNodeRejectsTwoInputs)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6}, "x");
    const std::size_t nodes_before = graph.node_count();

    LayerNode node(std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8));
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_FAILURE(node.forward(Trace::evaluation(builder), span_of({x, x}), outputs),
                      StatusCode::InvalidDescription, "input count is 1", "given is 2");
    EXPECT_TRUE(outputs.empty());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that a LayerNode given no input fails with the count it was given. */
TEST(Unit_Node, LayerNodeRejectsNoInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    LayerNode node(std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8));
    std::vector<Graph::ValueId> outputs;

    SA_EXPECT_FAILURE(node.forward(Trace::evaluation(builder), span_of({}), outputs),
                      StatusCode::InvalidDescription, "input count is 1", "given is 0");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that building a LayerNode over a null layer stops the process. */
TEST(Unit_NodeDeathTest, LayerNodeOverANullLayerIsFatal)
{
    SA_EXPECT_FATAL((void)LayerNode(std::unique_ptr<NN::ILayer>()), "LayerNode", "null layer");
}
