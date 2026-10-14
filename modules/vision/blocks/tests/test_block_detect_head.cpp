/**************************************************************************/
/* test_block_detect_head.cpp                                             */
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
 * @file test_block_detect_head.cpp
 * @brief Tests DetectHead's ports, state names and gradient barriers against YOLO26n row 23.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::DetectHead;

namespace
{
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 23 'detect'";

    /** @brief Holds the class count of the fixture. */
    constexpr int64_t K_CLASSES = 80;

    /** @brief Holds the number of levels row 23 reads. */
    constexpr std::size_t K_LEVELS = 3;

    /** @brief Holds the widths of rows 16, 19 and 22, which row 23 reads. */
    constexpr std::array<int64_t, K_LEVELS> K_WIDTHS = {64, 128, 256};

    /** @brief Holds the H and the W of each level for a 640 by 640 image. */
    constexpr std::array<int64_t, K_LEVELS> K_SIDES = {80, 40, 20};

    /** @brief Holds the H and the W of each level of a traced graph. */
    constexpr std::array<int64_t, K_LEVELS> K_TRACED_SIDES = {8, 4, 2};

    /** @brief Holds the batch extent of a traced graph. */
    constexpr int64_t K_BATCH = 2;

    /** @brief Holds the hidden width of the fixture's box branches. */
    constexpr int64_t K_BOX_WIDTH = 16;

    /** @brief Holds the hidden width of the fixture's class branches. */
    constexpr int64_t K_CLASS_WIDTH = 80;

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

    /** @brief Appends the five records of a convolution and its norm under @p prefix. */
    void append_conv_norm(std::vector<std::string>& records, const std::string& prefix,
                          const Shape& weight, int64_t width)
    {
        records.push_back(record(prefix + ".conv.weight", weight, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.weight", Shape{width}, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.bias", Shape{width}, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.running_mean", Shape{width}, StateKind::Buffer));
        records.push_back(record(prefix + ".bn.running_var", Shape{width}, StateKind::Buffer));
    }

    /** @brief Appends the two records of a biased 1x1 convolution under @p prefix. */
    void append_biased_pointwise(std::vector<std::string>& records, const std::string& prefix,
                                 int64_t out_channels, int64_t in_channels)
    {
        records.push_back(record(prefix + ".weight", Shape{out_channels, 1, 1, in_channels},
                                 StateKind::Parameter));
        records.push_back(record(prefix + ".bias", Shape{out_channels}, StateKind::Parameter));
    }

    /** @brief Appends the twelve records of each level's box branch under @p prefix. */
    void append_box_branches(std::vector<std::string>& records, const std::string& prefix)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            const std::string branch = prefix + "." + std::to_string(level);
            append_conv_norm(records, branch + ".0", Shape{K_BOX_WIDTH, 3, 3, K_WIDTHS[level]},
                             K_BOX_WIDTH);
            append_conv_norm(records, branch + ".1", Shape{K_BOX_WIDTH, 3, 3, K_BOX_WIDTH},
                             K_BOX_WIDTH);
            append_biased_pointwise(records, branch + ".2", 4, K_BOX_WIDTH);
        }
    }

    /** @brief Appends the twenty-two records of each level's class branch under @p prefix. */
    void append_class_branches(std::vector<std::string>& records, const std::string& prefix)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            const std::string branch = prefix + "." + std::to_string(level);
            const int64_t width = K_WIDTHS[level];
            append_conv_norm(records, branch + ".0.0", Shape{3, 3, width}, width);
            append_conv_norm(records, branch + ".0.1", Shape{K_CLASS_WIDTH, 1, 1, width},
                             K_CLASS_WIDTH);
            append_conv_norm(records, branch + ".1.0", Shape{3, 3, K_CLASS_WIDTH},
                             K_CLASS_WIDTH);
            append_conv_norm(records, branch + ".1.1",
                             Shape{K_CLASS_WIDTH, 1, 1, K_CLASS_WIDTH}, K_CLASS_WIDTH);
            append_biased_pointwise(records, branch + ".2", K_CLASSES, K_CLASS_WIDTH);
        }
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

    /** @brief Parses @p text as the attributes of one row. */
    Config::Value attributes_of(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }

    /** @brief Lists the per-sample [side, side, width] shape of each level. */
    std::vector<Shape> level_shapes(const std::array<int64_t, K_LEVELS>& sides,
                                    const std::array<int64_t, K_LEVELS>& widths)
    {
        std::vector<Shape> shapes;
        for (std::size_t level = 0; level < K_LEVELS; ++level)
            shapes.push_back(Shape{sides[level], sides[level], widths[level]});

        return shapes;
    }

    /** @brief Views @p shapes as the inputs of a row. */
    NN::ShapeSpan span_of(const std::vector<Shape>& shapes)
    {
        return NN::ShapeSpan(shapes.data(), shapes.size());
    }

    /** @brief Holds a graph with the three features of a batch of K_BATCH declared in it. */
    struct FeatureGraph
    {
        /** @brief Holds the graph a head is traced into. */
        Graph::Graph graph;

        /** @brief Holds the builder over the graph. */
        Graph::GraphBuilder builder{graph};

        /** @brief Holds the id of each level's [K_BATCH, side, side, width] input. */
        std::vector<Graph::ValueID> features;

        /** @brief Declares one input per level of K_TRACED_SIDES and K_WIDTHS. */
        FeatureGraph()
        {
            for (std::size_t level = 0; level < K_LEVELS; ++level)
            {
                const int64_t side = K_TRACED_SIDES[level];
                features.push_back(
                    builder.input(Shape{K_BATCH, side, side, K_WIDTHS[level]}, "feature"));
            }
        }

        /** @brief Views the three inputs as the values a head reads. */
        NN::ValueSpan inputs() const
        {
            return NN::ValueSpan(features.data(), features.size());
        }
    };

    /** @brief Builds the head of row 23 over K_SIDES and K_WIDTHS, or fails the test. */
    std::unique_ptr<DetectHead> make_row23()
    {
        const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
        Result<std::unique_ptr<DetectHead>> made =
            DetectHead::create(span_of(levels), K_CLASSES, K_YOLO_NORM);
        EXPECT_TRUE(Test::status_is_ok(made.status()));
        if (!made.status().is_ok())
            return nullptr;

        return std::move(made).value();
    }

} // namespace

/**
 * @brief Tests that row 23 publishes twelve ports under the names forward.sacp records: per
 *        copy the boxes of each level, then the scores, one-to-many first.
 */
TEST(Unit_BlockDetectHead, ThePortsCarryTheFixtureNamesAndShapes)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);

    EXPECT_EQ(head->input_count(), K_LEVELS);
    ASSERT_EQ(head->ports().size(), 4 * K_LEVELS);

    // tests/fixtures/yolo26/MANIFEST.json, forward.sacp tensors, from line 9856.
    const std::array<std::string_view, 2> copies = {"one2many", "one2one"};
    const std::array<std::string_view, 2> roles = {"boxes", "scores"};
    const std::array<int64_t, 2> channels = {4, K_CLASSES};

    std::size_t port = 0;
    for (std::string_view copy : copies)
    {
        for (std::size_t role = 0; role < roles.size(); ++role)
        {
            for (std::size_t level = 0; level < K_LEVELS; ++level, ++port)
            {
                const std::string name = std::string(copy) + ".level" + std::to_string(level) +
                                         "." + std::string(roles[role]);
                EXPECT_EQ(head->ports()[port].name, name) << "port " << port;
                EXPECT_EQ(head->ports()[port].shape,
                          (Shape{K_SIDES[level], K_SIDES[level], channels[role]}))
                    << "port " << port;
            }
        }
    }
}

/**
 * @brief Tests that a traced head yields the 204 records MANIFEST.json lists under model.23,
 *        records 390 to 593: names, NHWC shapes and order.
 */
TEST(Unit_BlockDetectHead, StateMatchesManifestRecords390To593)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    // tests/fixtures/yolo26/MANIFEST.json, weights.sushi tensors, from line 6485.
    std::vector<std::string> expected;
    append_box_branches(expected, "model.23.cv2");
    append_class_branches(expected, "model.23.cv3");
    append_box_branches(expected, "model.23.one2one_cv2");
    append_class_branches(expected, "model.23.one2one_cv3");
    ASSERT_EQ(expected.size(), 204u);

    EXPECT_EQ(walk(*head, "model.23"), expected);
}

/**
 * @brief Tests that a training trace emits three STOP_GRADIENT nodes, one over each feature
 *        in level order, and appends twelve values of the ports' shapes.
 */
TEST(Unit_BlockDetectHead, ATrainingTraceDetachesEachFeatureOnce)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    std::vector<Graph::ValueID> detached;
    for (const Graph::Node& node : traced.graph.nodes())
    {
        if (node.op == Ops::K_STOP_GRADIENT.id)
            detached.push_back(node.inputs[0]);
    }
    EXPECT_EQ(detached, traced.features);

    ASSERT_EQ(outputs.size(), 4 * K_LEVELS);
    for (std::size_t port = 0; port < outputs.size(); ++port)
    {
        const std::size_t level = port % K_LEVELS;
        const Shape& sample = head->ports()[port].shape;
        EXPECT_EQ(traced.graph.value(outputs[port]).shape,
                  (Shape{K_BATCH, K_TRACED_SIDES[level], K_TRACED_SIDES[level], sample[2]}))
            << "port " << port;
    }
}

/**
 * @brief Tests that an evaluation trace emits no STOP_GRADIENT, folds the 36 norms and keeps
 *        the 12 depthwise convolutions of the two copies.
 */
TEST(Unit_BlockDetectHead, AnEvaluationTraceEmitsNoStopGradient)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::evaluation(traced.builder), traced.inputs(), outputs));

    EXPECT_EQ(count_nodes(traced.graph, Ops::K_STOP_GRADIENT.id), 0u);
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_BATCHNORM2D.id), 0u);
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_BATCHNORM_FOLD.id), 36u);
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_DEPTHWISE_CONV2D.id), 12u);
    EXPECT_EQ(outputs.size(), 4 * K_LEVELS);
}

/**
 * @brief Tests the widths over a first level of 128 channels and 3 classes: a box branch of
 *        128 / 4 = 32 and a class branch of max(128, 3) = 128.
 */
TEST(Unit_BlockDetectHead, TheFirstLevelSetsTheBranchWidths)
{
    const std::vector<Shape> levels = {Shape{4, 4, 128}};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectHead> head,
                            DetectHead::create(span_of(levels), 3, K_YOLO_NORM));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, 4, 4, 128}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    const std::vector<std::string> state = walk(*head, "");
    ASSERT_EQ(state.size(), 68u);
    EXPECT_EQ(state[0],
              record("cv2.0.0.conv.weight", Shape{32, 3, 3, 128}, StateKind::Parameter));
    EXPECT_EQ(state[10], record("cv2.0.2.weight", Shape{4, 1, 1, 32}, StateKind::Parameter));
    EXPECT_EQ(state[17],
              record("cv3.0.0.1.conv.weight", Shape{128, 1, 1, 128}, StateKind::Parameter));
    EXPECT_EQ(state[32], record("cv3.0.2.weight", Shape{3, 1, 1, 128}, StateKind::Parameter));
    EXPECT_EQ(head->ports()[1].shape, (Shape{4, 4, 3}));
}

/** @brief Tests that forward refuses two values for three levels and appends nothing. */
TEST(Unit_BlockDetectHead, ForwardRefusesAWrongInputCount)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(head->forward(Trace::evaluation(traced.builder),
                                    NN::ValueSpan(traced.features.data(), 2), outputs),
                      StatusCode::InvalidDescription, "input count is 3", "given is 2");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that create refuses no level, a rank-2 level and a class count of zero. */
TEST(Unit_BlockDetectHead, CreateRefusesLevelsOrClassesThatDoNotFit)
{
    const std::vector<Shape> none;
    const std::vector<Shape> flat = {Shape{80, 80, 64}, Shape{40, 128}};
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);

    SA_EXPECT_FAILURE(DetectHead::create(span_of(none), K_CLASSES, K_YOLO_NORM),
                      StatusCode::InvalidDescription, "at least one feature level");
    SA_EXPECT_FAILURE(DetectHead::create(span_of(flat), K_CLASSES, K_YOLO_NORM),
                      StatusCode::InvalidDescription, "level 1", "[40, 128]", "rank-3");
    SA_EXPECT_FAILURE(DetectHead::create(span_of(levels), 0, K_YOLO_NORM),
                      StatusCode::InvalidDescription, "at least one class", "0");
}

/**
 * @brief Tests that make_detect_node builds row 23 from its file row: twelve ports, the first
 *        and the last of them, and the norm constants on every traced norm.
 */
TEST(Unit_BlockDetectHead, TheFactoryBuildsRow23)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const Config::Value attributes =
        attributes_of(R"({"classes": 80, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_detect_node(span_of(levels), attributes,
                                                             K_CONTEXT));
    EXPECT_EQ(node->input_count(), K_LEVELS);
    ASSERT_EQ(node->ports().size(), 4 * K_LEVELS);
    EXPECT_EQ(node->ports()[0].name, "one2many.level0.boxes");
    EXPECT_EQ(node->ports()[0].shape, (Shape{80, 80, 4}));
    EXPECT_EQ(node->ports()[11].name, "one2one.level2.scores");
    EXPECT_EQ(node->ports()[11].shape, (Shape{20, 20, K_CLASSES}));

    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    // Both constants are stored as given, so they compare exactly.
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_BATCHNORM2D.id), 36u);
    for (const Graph::Node& traced_node : traced.graph.nodes())
    {
        if (traced_node.op == Ops::K_BATCHNORM2D.id)
            EXPECT_EQ(traced_node.attributes.values[0], K_YOLO_NORM.epsilon);
    }
    for (const auto& update : list.updates())
        EXPECT_EQ(update.momentum, K_YOLO_NORM.momentum);

    EXPECT_EQ(walk(*node, "model.23").size(), 204u);
}

/** @brief Tests that the factory refuses an unknown key, a missing class count and no level. */
TEST(Unit_BlockDetectHead, TheFactoryRefusesARowThatDoesNotFit)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const std::vector<Shape> none;
    const Config::Value unknown = attributes_of(R"({"classes": 80, "filters": 64})");
    const Config::Value no_classes = attributes_of(R"({"bn_epsilon": 0.001})");
    const Config::Value row = attributes_of(R"({"classes": 80})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(levels), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 23 'detect'", "'filters'");
    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(levels), no_classes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'classes'", "absent");
    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(none), row, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 23 'detect'",
                      "at least one feature level");
}
