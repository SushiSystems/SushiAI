/**************************************************************************/
/* test_sppf.cpp                                                          */
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
 * @file test_sppf.cpp
 * @brief Tests SPPF's trace, its state names against YOLO26n row 9, and its factory.
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
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/sppf.hpp>
#include "host_state_support.hpp"

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using Vision::Blocks::SPPF;
using NN::Trace;

namespace SaTest = SushiAI::Test;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 9 'sppf'";

    /** @brief Holds one block whose closing add is not traced, and why. */
    struct NoAddRow
    {
        std::string name;  /**< The row's name. */
        int64_t out;       /**< The block's output width. */
        bool shortcut;     /**< Whether the block asks for the add. */
    };

    /** @brief Runs one block that traces no closing add. */
    class Small_BlockSppfNoAdd : public ::testing::TestWithParam<NoAddRow>
    {

    };

    /** @brief Names a row by its name field. */
    std::string row_name(const ::testing::TestParamInfo<NoAddRow>& info)
    {
        return info.param.name;
    }

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

    /** @brief Counts the norms a walk lists, one running mean each. */
    std::size_t count_norms(const std::vector<std::string>& records)
    {
        std::size_t norms = 0;
        for (const std::string& line : records)
        {
            if (line.find(".bn.running_mean ") != std::string::npos)
                ++norms;
        }

        return norms;
    }

    /** @brief Lists the shape of each input of @p node, in order. */
    std::vector<Shape> input_shapes(const Graph::Graph& graph, const Graph::Node& node)
    {
        std::vector<Shape> shapes;
        for (const Graph::ValueID input : node.inputs)
            shapes.push_back(graph.value(input).shape);

        return shapes;
    }

    /** @brief Counts the ADD nodes of @p graph that read @p value as either operand. */
    std::size_t count_adds_reading(const Graph::Graph& graph, Graph::ValueID value)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_ADD.id
                && (node.inputs[0] == value || node.inputs[1] == value))
                ++count;
        }

        return count;
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
        return require_value(Config::parse_json(text, "<test>"));
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

} // namespace

/** @brief Tests that the block keeps H and W and yields the output width. */
TEST(Small_BlockSppf, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const SPPF row9(256, 256, 5, 3, true, K_YOLO_NORM);
    const SPPF widening(64, 96, 5, 3, true, K_YOLO_NORM);

    EXPECT_EQ(row9.output_shape(Shape{20, 20, 256}).value(), (Shape{20, 20, 256}));
    EXPECT_EQ(widening.output_shape(Shape{5, 7, 64}).value(), (Shape{5, 7, 96}));
}

/** @brief Tests that the block reports the kernel it was built with: 5 for row 9, 3 for a narrow pool. */
TEST(Small_BlockSppf, TheKernelSizeIsTheOneTheConstructorTook)
{
    const SPPF row9(256, 256, 5, 3, true, K_YOLO_NORM);
    const SPPF narrow(64, 64, 3, 1, false, K_YOLO_NORM);

    EXPECT_EQ(row9.kernel_size(), 5);
    EXPECT_EQ(narrow.kernel_size(), 3);
}

/** @brief Tests that the pool keeps the extents of its input. */
TEST(Small_BlockSppf, ThePoolKeepsTheExtents)
{
    const SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);

    EXPECT_EQ(block.pool.output_shape(Shape{20, 20, 128}).value(), (Shape{20, 20, 128}));
}

/**
 * @brief Tests that a training trace of row 9 keeps the input's shape and requests a mean and a
 *        variance update for each norm of its state.
 */
TEST(Small_BlockSppf, Row9KeepsTheShapeAndUpdatesEveryNorm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 256}));
    EXPECT_EQ(list.updates().size(), 2 * count_norms(walk(block, "model.9")));
}

/** @brief Tests that the join takes the narrowed value and three pooled ones, 128 channels each. */
TEST(Small_BlockSppf, TheJoinTakesTheNarrowedValueAndEachPool)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    ASSERT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    const Graph::Node& join = graph.nodes()[first_node(graph, Ops::K_CONCAT_CHANNELS.id)];

    EXPECT_EQ(input_shapes(graph, join), std::vector<Shape>(4, Shape{2, 4, 4, 128}));
    EXPECT_EQ(graph.value(join.outputs[0]).shape, (Shape{2, 4, 4, 512}));
}

/** @brief Tests that a block with a shortcut over equal widths adds its input once. */
TEST(Small_BlockSppf, AShortcutOverEqualWidthsAddsTheInputOnce)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_TRUE(block.adds_input);
    EXPECT_EQ(count_adds_reading(graph, x), 1u);
    EXPECT_EQ(graph.node(graph.value(y).producer).op, Ops::K_ADD.id);
}

/** @brief Tests that no ADD is traced without a shortcut, or with one over unequal widths. */
TEST_P(Small_BlockSppfNoAdd, NoAddIsTracedWithoutAShortcutOrOverUnequalWidths)
{
    const NoAddRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    SPPF block(64, row.out, 5, 3, row.shortcut, K_YOLO_NORM);

    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, 4, 4, 64}, "x")));

    EXPECT_FALSE(block.adds_input);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockSppfNoAdd,
                         ::testing::Values(NoAddRow{"WithoutAShortcut", 64, false},
                                           NoAddRow{"OverUnequalWidths", 96, true}),
                         row_name);

/** @brief Tests that the pool count sets the width cv2 reads, as seen in the state. */
TEST(Small_BlockSppf, ThePoolCountSetsTheJoinedWidthCv2Reads)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 64}, "x");
    SPPF block(64, 64, 3, 1, false, K_YOLO_NORM);
    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    // One pool joins two parts of 32 channels, so cv2 reads 64.
    std::vector<std::string> expected = conv_records("b.cv1", Shape{32, 1, 1, 64});
    const std::vector<std::string> cv2 = conv_records("b.cv2", Shape{64, 1, 1, 64});
    expected.insert(expected.end(), cv2.begin(), cv2.end());

    EXPECT_EQ(walk(block, "b"), expected);
}

/**
 * @brief Tests that a traced SPPF(256, 256, 5, 3) yields the ten records MANIFEST.json lists
 *        under model.9, names, NHWC shapes and order.
 */
TEST(Small_BlockSppf, StateMatchesManifestRow9)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // MANIFEST.json of the registration fixtures, weights.sushi tensors, from line 2680.
    const std::vector<std::string> expected = {
        record("model.9.cv1.conv.weight", Shape{128, 1, 1, 256}, StateKind::Parameter),
        record("model.9.cv1.bn.weight", Shape{128}, StateKind::Parameter),
        record("model.9.cv1.bn.bias", Shape{128}, StateKind::Parameter),
        record("model.9.cv1.bn.running_mean", Shape{128}, StateKind::Buffer),
        record("model.9.cv1.bn.running_var", Shape{128}, StateKind::Buffer),
        record("model.9.cv2.conv.weight", Shape{256, 1, 1, 512}, StateKind::Parameter),
        record("model.9.cv2.bn.weight", Shape{256}, StateKind::Parameter),
        record("model.9.cv2.bn.bias", Shape{256}, StateKind::Parameter),
        record("model.9.cv2.bn.running_mean", Shape{256}, StateKind::Buffer),
        record("model.9.cv2.bn.running_var", Shape{256}, StateKind::Buffer),
    };

    EXPECT_EQ(walk(block, "model.9"), expected);
}

/** @brief Tests that make_sppf_node builds row 9 from its file row: the port's shape. */
TEST(Small_BlockSppf, TheFactoryBuildsRow9)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "kernel": 5, "pools": 3, "shortcut": true})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_sppf_node(one_input(input), attributes,
                                                           K_CONTEXT));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{20, 20, 256}));
}

/**
 * @brief Tests that a row stating only "filters" gets a 5x5 pool applied three times and no
 *        closing add.
 */
TEST(Small_BlockSppf, TheFactoryDefaultsToThreePoolsOfFiveAndNoShortcut)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(R"({"filters": 64})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_sppf_node(one_input(input), attributes,
                                                           K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 64}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);

    // Three pools join four parts of 32 channels, so cv2 reads 128.
    const std::vector<std::string> state = walk(*node, "");
    ASSERT_EQ(state.size(), 10u);
    EXPECT_EQ(state[5], record("cv2.conv.weight", Shape{64, 1, 1, 128}, StateKind::Parameter));
}

/** @brief Tests that the factory refuses an even kernel, which would change H and W. */
TEST(Small_BlockSppf, TheFactoryRefusesAnEvenKernel)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(R"({"filters": 64, "kernel": 4})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_sppf_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 9 'sppf'", "'kernel'", "odd", "4");
}

/** @brief Tests that the factory refuses a one-channel input, which leaves no hidden channel. */
TEST(Small_BlockSppf, TheFactoryRefusesAnInputOfOneChannel)
{
    const Shape input{4, 4, 1};
    const Config::Value attributes = attributes_of(R"({"filters": 64})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_sppf_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "[4, 4, 1]", "at least 2 channels");
}

/** @brief Tests that the factory refuses a key the row does not take. */
TEST(Small_BlockSppf, TheFactoryRefusesAnUnknownKey)
{
    const Shape input{4, 4, 64};
    const Config::Value unknown = attributes_of(R"({"filters": 64, "hidden": 32})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_sppf_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 9 'sppf'", "'hidden'");
}

/** @brief Tests that the factory refuses a row without its output width. */
TEST(Small_BlockSppf, TheFactoryRefusesAnAbsentFilterCount)
{
    const Shape input{4, 4, 64};
    const Config::Value no_filters = attributes_of(R"({"kernel": 5})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_sppf_node(one_input(input), no_filters, K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "absent");
}

/** @brief Tests that the factory refuses a pool count of zero. */
TEST(Small_BlockSppf, TheFactoryRefusesZeroPools)
{
    const Shape input{4, 4, 64};
    const Config::Value zero_pools = attributes_of(R"({"filters": 64, "pools": 0})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_sppf_node(one_input(input), zero_pools, K_CONTEXT),
                      StatusCode::InvalidDescription, "'pools'", "positive");
}

/** @brief Tests that the factory hands "bn_epsilon" and "bn_momentum" to both norms. */
TEST(Small_BlockSppf, TheFactoryHandsTheNormOptionsToBothNorms)
{
    const Shape input{4, 4, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "shortcut": true, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_sppf_node(one_input(input), attributes,
                                                           K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 2u);
    ASSERT_EQ(list.updates().size(), 4u);
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/** @brief Tests that the factory reads "pools": one pool joins two parts, so cv2 reads 64. */
TEST(Small_BlockSppf, TheFactoryReadsThePoolCount)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes =
        attributes_of(R"({"filters": 64, "kernel": 3, "pools": 1})");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_sppf_node(one_input(input), attributes,
                                                           K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 64}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    const std::vector<std::string> state = walk(*node, "");
    ASSERT_EQ(state.size(), 10u);
    EXPECT_EQ(state[5], record("cv2.conv.weight", Shape{64, 1, 1, 64}, StateKind::Parameter));
}

/**
 * @brief Tests that the closing add returns the input once: every weight and buffer is zero, so
 *        the branch is zero and a shortcut over equal widths gives x = 1..16 back.
 */
TEST(Small_BlockSppf, AShortcutReturnsTheInputWhenTheBranchIsZero)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 2, 2, 4}, "x");
    SPPF block(4, 4, 3, 1, true, K_YOLO_NORM);
    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);
    ASSERT_TRUE(trace.status().is_ok()) << trace.status().message();
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, SaTest::ramp(16)));
    SA_EXPECT_OK(SaTest::write_state_by(
        evaluator, graph, block, "",
        [](const std::string&, const Shape& shape) { return SaTest::filled(shape, 0.0); }));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, SaTest::ramp(16));
}

/** @brief Tests that without a shortcut the same zero branch gives zeros, not the input. */
TEST(Small_BlockSppf, WithoutAShortcutTheZeroBranchGivesZeros)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 2, 2, 4}, "x");
    SPPF block(4, 4, 3, 1, false, K_YOLO_NORM);
    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);
    ASSERT_TRUE(trace.status().is_ok()) << trace.status().message();
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, SaTest::ramp(16)));
    SA_EXPECT_OK(SaTest::write_state_by(
        evaluator, graph, block, "",
        [](const std::string&, const Shape& shape) { return SaTest::filled(shape, 0.0); }));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(y));
    EXPECT_EQ(result, std::vector<double>(16, 0.0));
}
