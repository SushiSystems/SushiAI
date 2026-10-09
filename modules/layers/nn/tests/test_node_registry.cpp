/**************************************************************************/
/* test_node_registry.cpp                                                 */
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
 * @file test_node_registry.cpp
 * @brief Tests NodeRegistry, register_layers, make_node and build_model over the nn rows.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/upsample2d.hpp>
#include <common/test_status.hpp>
#include <common/model_text.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;
using NN::ModelGraph;
using NN::NodeRef;

namespace
{
    using Test::MinimalCase;
    using Test::TraceSummary;
    using Test::attributes_from;
    using Test::make_minimal;
    using Test::model_from;
    using Test::span_of;
    using Test::trace_summary;
    using Test::types_of;

    /** @brief Holds the epsilon and the momentum K_BRANCHING_MODEL's defaults state. */
    constexpr NN::NormOptions K_NORM{1e-3, 0.03};

    /** @brief Lists a minimal object for every node row of registered(), in row order. */
    const std::vector<MinimalCase> K_MINIMAL_CASES = {
        {"batchnorm2d", "{}", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"concat", "{}", {Shape{4, 4, 2}, Shape{4, 4, 3}}, Shape{4, 4, 5}},
        {"add", "{}", {Shape{4, 4, 3}, Shape{4, 4, 3}}, Shape{4, 4, 3}},
    };

    /** @brief Describes one layer row's minimal object and the shape it reads. */
    struct LayerCase
    {
        std::string_view type;       /**< The registry key. */
        std::string_view attributes; /**< The row without its "type" and "from". */
        Shape input;                 /**< The per-sample shape the row reads. */
    };

    /** @brief Lists a minimal object for every layer row of registered(), in row order. */
    const std::vector<LayerCase> K_LAYER_CASES = {
        {"linear", R"({"units": 5})", Shape{6}},
        {"gelu", "{}", Shape{6}},
        {"relu", "{}", Shape{6}},
        {"tanh", "{}", Shape{6}},
        {"sigmoid", "{}", Shape{6}},
        {"leaky_relu", "{}", Shape{6}},
        {"unflatten", R"({"height": 2, "width": 3, "channels": 4})", Shape{24}},
        {"conv2d", R"({"filters": 5, "kernel": 3})", Shape{8, 8, 2}},
        {"maxpool2d", R"({"kernel": 2})", Shape{8, 8, 2}},
        {"avgpool2d", R"({"kernel": 2})", Shape{8, 8, 2}},
        {"global_avgpool", "{}", Shape{8, 8, 2}},
        {"flatten", "{}", Shape{8, 8, 2}},
        {"upsample", "{}", Shape{8, 8, 2}},
        {"silu", "{}", Shape{8, 8, 2}},
        {"mish", "{}", Shape{8, 8, 2}},
        {"slice_channels", R"({"offset": 0, "count": 1})", Shape{8, 8, 2}},
    };

    /** @brief Holds a model whose row 2 joins the model input with the row above it. */
    constexpr std::string_view K_INPUT_JOINING_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8, 8, 3]},
        "layers": [
            {"type": "conv2d", "filters": 4, "kernel": 1},
            {"type": "silu"},
            {"type": "concat", "from": ["input", -1]}
        ]
    })";

    /**
     * @brief Holds a branch and a merge as a model file: rows 0 and 2 meet in row 3, and rows 5
     *        and 6 normalise and activate what row 4 mixed.
     */
    constexpr std::string_view K_BRANCHING_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [16, 16, 3]},
        "defaults": {"batchnorm2d": {"bn_epsilon": 0.001, "bn_momentum": 0.03}},
        "layers": [
            {"type": "conv2d", "filters": 8, "kernel": 3, "stride": 2, "padding": 1,
             "use_bias": false},
            {"type": "conv2d", "filters": 16, "kernel": 3, "stride": 2, "padding": 1,
             "use_bias": false},
            {"type": "upsample", "scale": 2},
            {"type": "concat", "from": [-1, 0]},
            {"type": "conv2d", "filters": 8, "kernel": 1, "use_bias": false},
            {"type": "batchnorm2d"},
            {"type": "silu"}
        ]
    })";

    /** @brief Builds a registry from register_layers. */
    Result<NN::NodeRegistry> registered()
    {
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));

        return Result<NN::NodeRegistry>(std::move(registry));
    }

    /** @brief Builds by hand the graph K_BRANCHING_MODEL describes, rows named by index. */
    Result<ModelGraph> build_branching_model_by_hand()
    {
        ModelGraph model(Shape{16, 16, 3});

        SA_TRY_ASSIGN(const NodeRef stem,
                      model.add("0", NN::Conv2d::same_padded(3, 8, 3, 2), model.input()));
        SA_TRY_ASSIGN(const NodeRef down,
                      model.add("1", NN::Conv2d::same_padded(8, 16, 3, 2), stem));
        SA_TRY_ASSIGN(const NodeRef up, model.add("2", NN::Upsample2d{2}, down));

        const NodeRef both[] = {up, stem};
        const Shape shapes[] = {model.shape_of(up), model.shape_of(stem)};
        SA_TRY_ASSIGN(std::unique_ptr<NN::Concat> concat,
                      NN::Concat::create(NN::ShapeSpan(shapes, 2)));
        SA_TRY_ASSIGN(const NodeRef merged,
                      model.add("3", std::unique_ptr<NN::Node>(std::move(concat)),
                                SushiRuntime::span<const NodeRef>(both, 2)));

        SA_TRY_ASSIGN(const NodeRef mix,
                      model.add("4", NN::Conv2d::same_padded(24, 8, 1, 1), merged));
        SA_TRY_ASSIGN(const NodeRef normed, model.add("5", NN::BatchNorm2d(8, K_NORM), mix));
        SA_TRY_ASSIGN(const NodeRef activated, model.add("6", NN::SiLU{}, normed));
        SA_TRY(model.output_ports(activated));

        return Result<ModelGraph>(std::move(model));
    }

    /** @brief Returns one output entry that publishes port "output" of @p row as @p name. */
    Config::OutputSpec named_output(std::int64_t row, std::string_view name)
    {
        Config::OutputSpec output;
        output.node = row;
        output.port = std::string(NN::K_SINGLE_PORT_NAME);
        output.name = std::string(name);
        return output;
    }

} // namespace

/** @brief Tests that register_layers adds the 16 layer types, then batchnorm2d, concat and add. */
TEST(Unit_NodeRegistry, RegisterLayersAddsTheRowsInOrder)
{
    const std::vector<std::string_view> expected_layers = {
        "linear",    "gelu",     "relu",      "tanh",      "sigmoid", "leaky_relu",
        "unflatten", "conv2d",   "maxpool2d", "avgpool2d", "global_avgpool",
        "flatten",   "upsample", "silu",      "mish",      "slice_channels",
    };
    const std::vector<std::string_view> expected_nodes = {"batchnorm2d", "concat", "add"};

    NN::NodeRegistry registry;
    SA_EXPECT_OK(NN::register_layers(registry));

    EXPECT_EQ(types_of(registry.layers()), expected_layers);
    EXPECT_EQ(types_of(registry.nodes()), expected_nodes);
}

/** @brief Tests that a node row is refused over a node row and over a layer row of its type. */
TEST(Unit_NodeRegistry, ATypeAddedTwiceIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::NodeRegistry registry, registered());
    const NN::NodeEntry* const concat = registry.find_node("concat");
    ASSERT_NE(concat, nullptr);
    ASSERT_NE(registry.find_layer("relu"), nullptr);
    const NN::NodeFactory factory = concat->make;
    const std::size_t node_count = registry.nodes().size();
    const std::size_t layer_count = registry.layers().size();

    SA_EXPECT_FAILURE(registry.add(NN::NodeEntry{"concat", factory}),
                      StatusCode::InvalidDescription, "the type 'concat'", "already");
    SA_EXPECT_FAILURE(registry.add(NN::NodeEntry{"relu", factory}),
                      StatusCode::InvalidDescription, "the type 'relu'", "already");

    EXPECT_EQ(registry.nodes().size(), node_count);
    EXPECT_EQ(registry.layers().size(), layer_count);
}

/** @brief Tests that make_node over an empty registry fails and lists no type. */
TEST(Unit_NodeRegistry, AnEmptyRegistryKnowsNoType)
{
    const NN::NodeRegistry registry;
    const std::vector<Shape> inputs = {Shape{8, 8, 2}};
    const std::string_view no_types = "the node types are: ; the layer types are: ";

    const Result<std::unique_ptr<NN::Node>> made = NN::make_node(
        registry, "silu", span_of(inputs), attributes_from("{}"), "layer 0");

    SA_EXPECT_FAILURE(made, StatusCode::InvalidDescription, "layer 0", "the type 'silu'",
                      "does not have", no_types);

    // The two lists close the message, so nothing may follow the label of the layer list.
    const std::string message(made.status().message());
    ASSERT_GE(message.size(), no_types.size());
    EXPECT_EQ(message.substr(message.size() - no_types.size()), no_types);
}

/** @brief Tests that every node row of registered() has a minimal case here, in row order. */
TEST(Unit_NodeRegistry, EveryRowHasAMinimalCase)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const SushiRuntime::span<const NN::NodeEntry> nodes = registry.nodes();
    ASSERT_EQ(K_MINIMAL_CASES.size(), nodes.size());

    for (std::size_t index = 0; index < nodes.size(); ++index)
        EXPECT_EQ(K_MINIMAL_CASES[index].type, nodes[index].type);
}

/** @brief Tests that each row builds a one-port node of the expected shape from its case. */
TEST(Unit_NodeRegistry, EachRowBuildsFromAMinimalObject)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    for (const MinimalCase& minimal : K_MINIMAL_CASES)
    {
        SCOPED_TRACE(std::string(minimal.type));

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                                make_minimal(registry, minimal, minimal.attributes));

        ASSERT_NE(node, nullptr);
        ASSERT_EQ(node->ports().size(), 1u);
        EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
        EXPECT_EQ(node->ports()[0].shape, minimal.output);
    }
}

/** @brief Tests that each row refuses a key it does not know and names the row and the key. */
TEST(Unit_NodeRegistry, EachRowRejectsAnUnknownKey)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    for (const MinimalCase& minimal : K_MINIMAL_CASES)
    {
        const std::string context = "layer 0 '" + std::string(minimal.type) + "'";

        SA_EXPECT_FAILURE(make_minimal(registry, minimal, R"({"no_such_key": 1})"),
                          StatusCode::InvalidDescription, context, "'no_such_key'");
    }
}

/** @brief Tests that a concat row over inputs of unequal H fails and names the row. */
TEST(Unit_NodeRegistry, AConcatRowNamesItselfWhenItsInputsDisagree)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> inputs = {Shape{4, 4, 2}, Shape{8, 4, 2}};

    SA_EXPECT_FAILURE(NN::make_node(registry, "concat", span_of(inputs), attributes_from("{}"),
                                    "layer 7 'concat'"),
                      StatusCode::InvalidDescription, "layer 7 'concat'", "[8, 4, 2]", "H and W");
}

/** @brief Tests that a node row's type builds the row's own node, not a LayerNode. */
TEST(Unit_NodeRegistry, ANodeTypeBuildsItsOwnNode)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> inputs = {Shape{4, 4, 2}, Shape{4, 4, 3}};

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_node(registry, "concat", span_of(inputs),
                                          attributes_from("{}"), "layer 0 'concat'"));

    EXPECT_NE(dynamic_cast<NN::Concat*>(node.get()), nullptr);
    EXPECT_EQ(dynamic_cast<NN::LayerNode*>(node.get()), nullptr);
}

/** @brief Tests that every layer row's type is reached through make_node as a LayerNode. */
TEST(Unit_NodeRegistry, ALayerTypeIsWrappedInALayerNode)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> inputs = {Shape{4, 4, 2}};

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_node(registry, "upsample", span_of(inputs),
                                          attributes_from(R"({"scale": 2})"),
                                          "layer 0 'upsample'"));

    ASSERT_NE(dynamic_cast<NN::LayerNode*>(node.get()), nullptr);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 8, 2}));

    const SushiRuntime::span<const NN::LayerEntry> layers = registry.layers();
    ASSERT_EQ(K_LAYER_CASES.size(), layers.size());
    for (std::size_t index = 0; index < layers.size(); ++index)
    {
        const LayerCase& minimal = K_LAYER_CASES[index];
        SCOPED_TRACE(std::string(minimal.type));
        ASSERT_EQ(minimal.type, layers[index].type);

        const std::vector<Shape> read = {minimal.input};
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> built,
                                NN::make_node(registry, minimal.type, span_of(read),
                                              attributes_from(minimal.attributes), "layer 0"));
        EXPECT_NE(dynamic_cast<NN::LayerNode*>(built.get()), nullptr);

        // An unknown key is refused by the layer's own factory, so the type was found.
        SA_EXPECT_FAILURE(NN::make_node(registry, minimal.type, span_of(read),
                                        attributes_from(R"({"no_such_key": 1})"), "layer 0"),
                          StatusCode::InvalidDescription, "'no_such_key'");
    }
}

/** @brief Tests that a layer type given two inputs fails and names the count and the type. */
TEST(Unit_NodeRegistry, ALayerTypeRefusesTwoInputs)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> inputs = {Shape{6}, Shape{6}};

    SA_EXPECT_FAILURE(NN::make_node(registry, "relu", span_of(inputs), attributes_from("{}"),
                                    "layer 3 'relu'"),
                      StatusCode::InvalidDescription, "layer 3 'relu'", "reads 2 values",
                      "reads one");
}

/** @brief Tests that what a layer factory throws comes back as a status with its message. */
TEST(Unit_NodeRegistry, ALayerFactorysRefusalComesBackAsAStatus)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> vector = {Shape{6}};
    const std::vector<Shape> image = {Shape{8, 8, 2}};

    SA_EXPECT_FAILURE(NN::make_node(registry, "linear", span_of(vector), attributes_from("{}"),
                                    "layer 4 'linear'"),
                      StatusCode::InvalidDescription, "layer 4 'linear'", "'units'");
    SA_EXPECT_FAILURE(NN::make_node(registry, "linear", span_of(image),
                                    attributes_from(R"({"units": 3})"), "layer 4 'linear'"),
                      StatusCode::InvalidDescription, "layer 4 'linear'", "[8, 8, 2]", "'flatten'");
}

/** @brief Tests that the maxpool2d row reads trailing_padding as an integer or as [h, w]. */
TEST(Unit_NodeRegistry, AMaxPoolRowReadsItsTrailingPadding)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> image = {Shape{13, 13, 2}};

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> both,
        NN::make_node(registry, "maxpool2d", span_of(image),
                      attributes_from(R"({"kernel": 2, "stride": 1, "trailing_padding": 1})"),
                      "layer 0 'maxpool2d'"));
    EXPECT_EQ(both->ports()[0].shape, (Shape{13, 13, 2}));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> pair,
        NN::make_node(registry, "maxpool2d", span_of(image),
                      attributes_from(R"({"kernel": 2, "stride": 1, "trailing_padding": [1, 0]})"),
                      "layer 0 'maxpool2d'"));
    EXPECT_EQ(pair->ports()[0].shape, (Shape{13, 12, 2}));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> absent,
        NN::make_node(registry, "maxpool2d", span_of(image),
                      attributes_from(R"({"kernel": 2, "stride": 1})"), "layer 0 'maxpool2d'"));
    EXPECT_EQ(absent->ports()[0].shape, (Shape{12, 12, 2}));
}

/** @brief Tests that the avgpool2d row refuses trailing_padding as a key it does not know. */
TEST(Unit_NodeRegistry, AnAvgPoolRowRefusesTrailingPadding)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> image = {Shape{13, 13, 2}};

    SA_EXPECT_FAILURE(NN::make_node(registry, "avgpool2d", span_of(image),
                                    attributes_from(R"({"kernel": 2, "trailing_padding": 1})"),
                                    "layer 0 'avgpool2d'"),
                      StatusCode::InvalidDescription, "layer 0 'avgpool2d'", "'trailing_padding'");
}

/** @brief Tests that the maxpool2d row refuses a negative trailing_padding and names it. */
TEST(Unit_NodeRegistry, AMaxPoolRowRefusesANegativeTrailingPadding)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> image = {Shape{13, 13, 2}};

    SA_EXPECT_FAILURE(NN::make_node(registry, "maxpool2d", span_of(image),
                                    attributes_from(R"({"kernel": 2, "trailing_padding": -1})"),
                                    "layer 0 'maxpool2d'"),
                      StatusCode::InvalidDescription, "layer 0 'maxpool2d'",
                      "non-negative trailing padding", "-1,-1");
}

/** @brief Tests that the slice_channels row keeps the channels its offset and count name. */
TEST(Unit_NodeRegistry, ASliceChannelsRowReadsItsOffsetAndCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> image = {Shape{4, 4, 64}};

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> upper_half,
                            NN::make_node(registry, "slice_channels", span_of(image),
                                          attributes_from(R"({"offset": 32, "count": 32})"),
                                          "layer 0 'slice_channels'"));
    EXPECT_EQ(upper_half->ports()[0].shape, (Shape{4, 4, 32}));

    SA_EXPECT_FAILURE(NN::make_node(registry, "slice_channels", span_of(image),
                                    attributes_from(R"({"offset": 48, "count": 32})"),
                                    "layer 0 'slice_channels'"),
                      StatusCode::InvalidDescription, "layer 0 'slice_channels'", "offset=48",
                      "count=32");

    SA_EXPECT_FAILURE(NN::make_node(registry, "slice_channels", span_of(image),
                                    attributes_from(R"({"offset": 32})"),
                                    "layer 0 'slice_channels'"),
                      StatusCode::InvalidDescription, "layer 0 'slice_channels'", "count");
}

/** @brief Tests that an unknown type fails naming the row, the type and every row's type. */
TEST(Unit_NodeRegistry, AnUnknownTypeListsBothTables)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const std::vector<Shape> inputs = {Shape{8, 8, 2}};
    const Result<std::unique_ptr<NN::Node>> made = NN::make_node(
        registry, "conv3d", span_of(inputs), attributes_from("{}"), "layer 5 'conv3d'");

    SA_EXPECT_FAILURE(made, StatusCode::InvalidDescription, "layer 5 'conv3d'",
                      "the type 'conv3d'", "does not have", "node types", "layer types");

    for (const NN::NodeEntry& entry : registry.nodes())
        SA_EXPECT_FAILURE(made, StatusCode::InvalidDescription,
                          "'" + std::string(entry.type) + "'");

    for (const NN::LayerEntry& entry : registry.layers())
        SA_EXPECT_FAILURE(made, StatusCode::InvalidDescription,
                          "'" + std::string(entry.type) + "'");
}

/**
 * @brief Tests that build_model of a file and the same graph built by hand trace to equal
 *        node lists, equal outputs and equal state names and shapes.
 */
TEST(Unit_NodeRegistry, AFileAndAHandBuiltGraphAgree)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const Config::ModelSpec spec = model_from(K_BRANCHING_MODEL);
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph from_file, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph by_hand, build_branching_model_by_hand());

    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary file_trace,
                            trace_summary(from_file, spec.input_shape, "model"));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary hand_trace,
                            trace_summary(by_hand, spec.input_shape, "model"));

    ASSERT_EQ(from_file.size(), 7u);
    EXPECT_EQ(from_file.size(), by_hand.size());
    EXPECT_FALSE(file_trace.ops.empty());
    EXPECT_EQ(file_trace.ops, hand_trace.ops);
    EXPECT_EQ(file_trace.outputs, hand_trace.outputs);
    EXPECT_EQ(file_trace.outputs, (std::vector<std::string>{"output [2, 8, 8, 8]"}));

    // Three conv2d weights, then the norm row's two parameters and two running buffers.
    ASSERT_EQ(file_trace.state.size(), 7u);
    EXPECT_EQ(file_trace.state, hand_trace.state);
    EXPECT_EQ(file_trace.state.front(), "model.0.weight [8, 3, 3, 3] parameter");
    EXPECT_EQ(file_trace.state.back(), "model.5.running_var [8] buffer");
}

/** @brief Tests that a row built from a file reads its sources in the order the file gave. */
TEST(Unit_NodeRegistry, ARowReadsItsSourcesInFileOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    SA_ASSERT_OK_AND_ASSIGN(const ModelGraph model,
                            NN::build_model(registry, model_from(K_BRANCHING_MODEL)));

    // Row 3 joins row 2's 16 channels, then row 0's 8, at row 0's 8 x 8 extent.
    EXPECT_EQ(model.node(3).ports()[0].shape, (Shape{8, 8, 24}));
    EXPECT_EQ(model.node(2).ports()[0].shape, (Shape{8, 8, 16}));
    EXPECT_EQ(model.node(1).ports()[0].shape, (Shape{4, 4, 16}));
}

/** @brief Tests that "outputs" publishes a named port under the stated name, in file order. */
TEST(Unit_NodeRegistry, OutputsArePublishedUnderTheirStatedNames)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    Config::ModelSpec spec = model_from(K_BRANCHING_MODEL);
    spec.outputs = {named_output(4, "mixed"), named_output(0, "stem")};

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, "model"));

    EXPECT_EQ(trace.outputs,
              (std::vector<std::string>{"mixed [2, 8, 8, 8]", "stem [2, 8, 8, 8]"}));
}

/** @brief Tests that a port entry with no name is published under the port's own name. */
TEST(Unit_NodeRegistry, AnOutputWithoutANameKeepsThePortsName)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    Config::ModelSpec spec = model_from(K_BRANCHING_MODEL);
    spec.outputs = {named_output(1, "")};

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, "model"));

    EXPECT_EQ(trace.outputs, (std::vector<std::string>{"output [2, 4, 4, 16]"}));
}

/** @brief Tests that a source may name a port, and that an unknown port names the row. */
TEST(Unit_NodeRegistry, ASourceMayNameAPort)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    Config::ModelSpec spec = model_from(K_BRANCHING_MODEL);
    spec.layers[1].from[0].port = std::string(NN::K_SINGLE_PORT_NAME);
    SA_EXPECT_OK(NN::build_model(registry, spec));

    spec.layers[1].from[0].port = "no_such_port";
    SA_EXPECT_FAILURE(NN::build_model(registry, spec), StatusCode::InvalidDescription,
                      "layer 1 'conv2d'", "'no_such_port'");
}

/** @brief Tests that a row below row 0 reads the model input, joined in the file's order. */
TEST(Unit_NodeRegistry, ALaterRowMayReadTheModelInput)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const Config::ModelSpec spec = model_from(K_INPUT_JOINING_MODEL);
    ASSERT_EQ(spec.layers[2].from.size(), 2u);
    ASSERT_EQ(spec.layers[2].from[0].node, Config::K_MODEL_INPUT);

    SA_ASSERT_OK_AND_ASSIGN(const ModelGraph model, NN::build_model(registry, spec));

    // Row 2 joins the input's 3 channels, then row 1's 4.
    EXPECT_EQ(model.node(2).ports()[0].shape, (Shape{8, 8, 7}));
}

/** @brief Tests that a source naming a port of the model input is refused and names the row. */
TEST(Unit_NodeRegistry, ASourceCannotNameAPortOfTheModelInput)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    Config::ModelSpec spec = model_from(K_INPUT_JOINING_MODEL);
    spec.layers[2].from[0].port = std::string(NN::K_SINGLE_PORT_NAME);

    SA_EXPECT_FAILURE(NN::build_model(registry, spec), StatusCode::InvalidDescription,
                      "layer 2 'concat'", "the model input has no port", "'output'");
}

/** @brief Tests that a row the factory refuses fails the build with the row's index and type. */
TEST(Unit_NodeRegistry, ARefusedRowNamesItsIndexAndType)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    Config::ModelSpec spec = model_from(K_BRANCHING_MODEL);
    spec.layers[4].attributes = attributes_from(R"({"filters": 0, "kernel": 1})");

    SA_EXPECT_FAILURE(NN::build_model(registry, spec), StatusCode::InvalidDescription,
                      "layer 4 'conv2d'", "positive channel counts", "out=0");
}

/** @brief Tests that a spec built by hand is refused when a row reads a row not above it. */
TEST(Unit_NodeRegistry, ARowThatReadsARowNotAboveItIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    Config::ModelSpec forward = model_from(K_BRANCHING_MODEL);
    forward.layers[1].from[0].node = 3;
    SA_EXPECT_FAILURE(NN::build_model(registry, forward), StatusCode::InvalidDescription,
                      "layer 1 'conv2d'", "row 3", "not above");

    Config::ModelSpec negative = model_from(K_BRANCHING_MODEL);
    negative.layers[2].from[0].node = -2;
    SA_EXPECT_FAILURE(NN::build_model(registry, negative), StatusCode::InvalidDescription,
                      "layer 2 'upsample'", "row -2");

    Config::ModelSpec sourceless = model_from(K_BRANCHING_MODEL);
    sourceless.layers[2].from.clear();
    SA_EXPECT_FAILURE(NN::build_model(registry, sourceless), StatusCode::InvalidDescription,
                      "layer 2 'upsample'", "no source");
}

/** @brief Tests that a hand-built spec with no outputs publishes its last row, as a file does. */
TEST(Unit_NodeRegistry, AnEmptyOutputListPublishesTheLastRowWhole)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    const Config::ModelSpec parsed = model_from(K_BRANCHING_MODEL);
    Config::ModelSpec silent = parsed;
    silent.outputs.clear();

    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph from_file, NN::build_model(registry, parsed));
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph by_hand, NN::build_model(registry, silent));

    EXPECT_EQ(by_hand.output_shape(parsed.input_shape), (Shape{8, 8, 8}));
    EXPECT_EQ(by_hand.output_shape(parsed.input_shape),
              from_file.output_shape(parsed.input_shape));
}

/** @brief Tests that a spec with no layer or a wrong output entry is refused. */
TEST(Unit_NodeRegistry, AWrongOutputListIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
    Config::ModelSpec empty = model_from(K_BRANCHING_MODEL);
    empty.layers.clear();
    SA_EXPECT_FAILURE(NN::build_model(registry, empty), StatusCode::InvalidDescription,
                      "no layers");

    Config::ModelSpec beyond = model_from(K_BRANCHING_MODEL);
    beyond.outputs = {named_output(7, "late")};
    SA_EXPECT_FAILURE(NN::build_model(registry, beyond), StatusCode::InvalidDescription, "output 0",
                      "row 7", "rows 0 to 6");

    Config::ModelSpec portless = model_from(K_BRANCHING_MODEL);
    portless.outputs = {named_output(4, "mixed")};
    portless.outputs[0].port.clear();
    SA_EXPECT_FAILURE(NN::build_model(registry, portless), StatusCode::InvalidDescription,
                      "output 0", "'mixed'", "without a port");

    Config::ModelSpec unknown = model_from(K_BRANCHING_MODEL);
    unknown.outputs = {named_output(4, "mixed")};
    unknown.outputs[0].port = "no_such_port";
    SA_EXPECT_FAILURE(NN::build_model(registry, unknown), StatusCode::InvalidDescription,
                      "output 0", "'no_such_port'");

    Config::ModelSpec twice = model_from(K_BRANCHING_MODEL);
    twice.outputs = {named_output(4, "mixed"), named_output(0, "mixed")};
    SA_EXPECT_FAILURE(NN::build_model(registry, twice), StatusCode::InvalidDescription, "output 1",
                      "'mixed'");
}
