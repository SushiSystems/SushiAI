/**************************************************************************/
/* test_c2f.cpp                                                           */
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
 * @file test_c2f.cpp
 * @brief Tests C2f's trace, its state names and shapes, and its factory.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c2f.hpp>

using namespace SushiAI;
using NN::BufferUpdateList;
using Vision::Blocks::Bottleneck;
using Vision::Blocks::C2f;
using NN::NormOptions;
using NN::Trace;

namespace
{
    /** @brief Holds the epsilon and the momentum every case hands to its norms. */
    constexpr NormOptions K_TEST_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 4 'c2f'";

    /** @brief Names which of the two kinds of state a record is. */
    enum class StateKind : std::uint8_t
    {
        Parameter, /**< A learnable tensor. */
        Buffer,    /**< A stored tensor no gradient reaches. */
    };

    /** @brief Writes one piece of state as "name [extents] kind", for a readable mismatch. */
    std::string record(const std::string& name, const Shape& shape, StateKind kind)
    {
        return name + " " + shape.to_string() +
               (kind == StateKind::Buffer ? " buffer" : " parameter");
    }

    /** @brief Lists every piece of state under @p module, in the order the walk yields. */
    template<typename M>
    std::vector<std::string> walk(M& module, std::string_view prefix)
    {
        std::vector<std::string> records;
        const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
        {
            records.push_back(record(name, state.shape(), StateKind::Parameter));
        };
        const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
        {
            records.push_back(record(name, state.shape(), StateKind::Buffer));
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(module, prefix, visitor);

        return records;
    }

    /** @brief Counts the nodes of @p graph that carry the operation @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }

    /** @brief Returns the index of the first node of @p graph that carries @p op. */
    std::size_t first_node(const Graph::Graph& graph, OpID op)
    {
        std::size_t index = 0;
        while (index < graph.node_count() && graph.nodes()[index].op != op)
            ++index;

        return index;
    }

    /** @brief Views one shape as the inputs of a one-input row. */
    NN::ShapeSpan one_input(const Shape& shape)
    {
        return NN::ShapeSpan(&shape, static_cast<std::size_t>(1));
    }

    /** @brief Parses @p text as the attributes of one row. */
    Config::Value attributes_of(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }

    /** @brief Expects @p options on every BATCHNORM2D node of @p graph and every update. */
    void expect_norm_options(const Graph::Graph& graph, const BufferUpdateList& list,
                             const NormOptions& options)
    {
        // Both constants are stored as given, so they compare exactly.
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_BATCHNORM2D.id)
                EXPECT_EQ(node.attributes.values[0], options.epsilon);
        }

        for (const auto& update : list.updates())
            EXPECT_EQ(update.momentum, options.momentum);
    }

    /** @brief Lists the five records of one Conv: its weight and the norm's four tensors. */
    std::vector<std::string> conv_records(const std::string& prefix, const Shape& weight)
    {
        const Shape channels{weight[0]};

        return {
            record(prefix + ".conv.weight", weight, StateKind::Parameter),
            record(prefix + ".bn.weight", channels, StateKind::Parameter),
            record(prefix + ".bn.bias", channels, StateKind::Parameter),
            record(prefix + ".bn.running_mean", channels, StateKind::Buffer),
            record(prefix + ".bn.running_var", channels, StateKind::Buffer),
        };
    }

    /** @brief Appends @p more to @p all. */
    void append(std::vector<std::string>& all, const std::vector<std::string>& more)
    {
        all.insert(all.end(), more.begin(), more.end());
    }

} // namespace

namespace
{
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Builds @p repeats bottlenecks of width 4 that keep their width. */
    NN::ModuleList<Bottleneck> make_chain(int64_t repeats, bool shortcut)
    {
        NN::ModuleList<Bottleneck> chain;
        for (int64_t index = 0; index < repeats; ++index)
            chain.push_back(Bottleneck(4, 4, 4, shortcut, K_TEST_NORM));

        return chain;
    }

} // namespace

/** @brief Tests that C2f keeps H and W and yields the output width. */
TEST(Small_BlockC2f, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const C2f block(8, 12, 4, make_chain(2, true), K_TEST_NORM);

    EXPECT_EQ(block.output_shape(Shape{5, 7, 8}), (Shape{5, 7, 12}));
}

/**
 * @brief Tests that C2f(8, 12, 4) over two bottlenecks traces six Conv blocks, two slices, two
 *        ADDs and one CONCAT_CHANNELS.
 */
TEST(Small_BlockC2f, ATrainingTraceEmitsSixConvBlocksTwoSlicesTwoAddsAndOneConcat)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 8}, "x");
    C2f block(8, 12, 4, make_chain(2, true), K_TEST_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    EXPECT_EQ(list.updates().size(), 12u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 12}));
    expect_norm_options(graph, list, K_TEST_NORM);
}

/**
 * @brief Tests the wiring on a tiny [1, 2, 2, 8] input: the first slice starts at channel 0, the
 *        second at channel 4, the first bottleneck reads the second slice, the second reads the
 *        first bottleneck, and the join takes the four parts of 4 channels in that order.
 */
TEST(Small_BlockC2f, TheJoinReadsBothHalvesThenEachBottleneckOffTheOneBefore)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 2, 2, 8}, "x");
    C2f block(8, 12, 4, make_chain(2, true), K_TEST_NORM);
    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    ASSERT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    const Graph::Node& join = graph.nodes()[first_node(graph, Ops::K_CONCAT_CHANNELS.id)];
    ASSERT_EQ(join.inputs.size(), 4u);

    const Graph::Node& first_half = graph.node(graph.value(join.inputs[0]).producer);
    const Graph::Node& second_half = graph.node(graph.value(join.inputs[1]).producer);
    const Graph::Node& first_result = graph.node(graph.value(join.inputs[2]).producer);
    const Graph::Node& second_result = graph.node(graph.value(join.inputs[3]).producer);
    ASSERT_EQ(first_half.op, Ops::K_SLICE_CHANNELS.id);
    ASSERT_EQ(second_half.op, Ops::K_SLICE_CHANNELS.id);
    ASSERT_EQ(first_result.op, Ops::K_ADD.id);
    ASSERT_EQ(second_result.op, Ops::K_ADD.id);

    // A slice stores its offset as a whole number in slot 0, so it compares exactly.
    EXPECT_NEAR(first_half.attributes.values[0], 0.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(second_half.attributes.values[0], 4.0, K_EXACT_TOLERANCE);
    EXPECT_EQ(first_half.inputs[0], second_half.inputs[0]);

    // A bottleneck adds its own input, which is the value it read.
    EXPECT_EQ(first_result.inputs[0], join.inputs[1]);
    EXPECT_EQ(second_result.inputs[0], join.inputs[2]);
    for (std::size_t part = 0; part < join.inputs.size(); ++part)
        EXPECT_EQ(graph.value(join.inputs[part]).shape, (Shape{1, 2, 2, 4}));
    EXPECT_EQ(graph.value(join.outputs[0]).shape, (Shape{1, 2, 2, 16}));
}

/** @brief Tests that a chain built without a shortcut traces no ADD. */
TEST(Small_BlockC2f, NoShortcutTracesNoAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 8}, "x");
    C2f block(8, 12, 4, make_chain(2, false), K_TEST_NORM);

    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 6u);
}

/**
 * @brief Tests that the state is cv1, cv2 and m.<i>.cv1, m.<i>.cv2 in walk order: cv1 at twice
 *        the hidden width, cv2 over (2 + 2) * 4 channels, and 3x3 bottleneck windows at the full
 *        hidden width.
 */
TEST(Small_BlockC2f, StateNamesAndShapesFollowTheSplitLayout)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 8}, "x");
    C2f block(8, 12, 4, make_chain(2, true), K_TEST_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    std::vector<std::string> expected;
    append(expected, conv_records("c.cv1", Shape{8, 1, 1, 8}));
    append(expected, conv_records("c.cv2", Shape{12, 1, 1, 16}));
    append(expected, conv_records("c.m.0.cv1", Shape{4, 3, 3, 4}));
    append(expected, conv_records("c.m.0.cv2", Shape{4, 3, 3, 4}));
    append(expected, conv_records("c.m.1.cv1", Shape{4, 3, 3, 4}));
    append(expected, conv_records("c.m.1.cv2", Shape{4, 3, 3, 4}));

    EXPECT_EQ(walk(block, "c"), expected);
}

/** @brief Tests that make_c2f_node defaults to one bottleneck with a shortcut. */
TEST(Small_BlockC2f, TheFactoryDefaultsToOneBottleneckWithAShortcut)
{
    const Shape input{4, 4, 8};
    const Config::Value attributes = attributes_of(R"({"filters": 12, "hidden": 4})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2f_node(one_input(input), attributes, K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{4, 4, 12}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 8}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 4, 4, 12}));
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
}

/** @brief Tests that the factory reads "repeats" and "shortcut" and builds the same state. */
TEST(Small_BlockC2f, TheFactoryReadsRepeatsAndShortcut)
{
    const Shape input{4, 4, 8};
    const Config::Value attributes = attributes_of(
        R"({"filters": 12, "hidden": 4, "repeats": 2, "shortcut": false})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2f_node(one_input(input), attributes, K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 8}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);

    const std::vector<std::string> state = walk(*node, "c");
    ASSERT_EQ(state.size(), 30u);
    EXPECT_EQ(state[25], record("c.m.1.cv2.conv.weight", Shape{4, 3, 3, 4},
                                StateKind::Parameter));
}

/** @brief Tests that the factory refuses an unknown key, a missing key and zero repeats. */
TEST(Small_BlockC2f, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{4, 4, 8};
    const Config::Value unknown = attributes_of(R"({"filters": 12, "hidden": 4, "inner": "x"})");
    const Config::Value no_hidden = attributes_of(R"({"filters": 12})");
    const Config::Value no_filters = attributes_of(R"({"hidden": 4})");
    const Config::Value zero_repeats =
        attributes_of(R"({"filters": 12, "hidden": 4, "repeats": 0})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2f_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 4 'c2f'", "'inner'");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c2f_node(one_input(input), no_hidden, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "absent");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c2f_node(one_input(input), no_filters, K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "absent");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c2f_node(one_input(input), zero_repeats, K_CONTEXT),
                      StatusCode::InvalidDescription, "'repeats'", "positive");
}

/** @brief Tests that the factory refuses two inputs. */
TEST(Small_BlockC2f, TheFactoryRefusesTwoInputs)
{
    const Shape two[] = {Shape{4, 4, 8}, Shape{4, 4, 8}};
    const Config::Value attributes = attributes_of(R"({"filters": 12, "hidden": 4})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2f_node(NN::ShapeSpan(two), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "one input", "2");
}

/** @brief Tests that the factory hands "bn_epsilon" and "bn_momentum" to all six norms. */
TEST(Small_BlockC2f, TheFactoryHandsTheNormOptionsToEveryNorm)
{
    const Shape input{4, 4, 8};
    const Config::Value attributes = attributes_of(
        R"({"filters": 12, "hidden": 4, "repeats": 2, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2f_node(one_input(input), attributes, K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 8}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 6u);
    ASSERT_EQ(list.updates().size(), 12u);
    expect_norm_options(graph, list, K_TEST_NORM);
}
