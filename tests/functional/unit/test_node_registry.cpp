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
 * @brief Tests the node registry, make_node, build_model and the YOLO26n model file.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/blocks/conv_norm.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/upsample2d.hpp>
#include "../../common/reference_fixture.hpp"
#include "../../common/test_status.hpp"

using namespace SushiAI;
using NN::ModelGraph;
using NN::NodeRef;

namespace
{
    /** @brief Lists the operation of every node of a graph, in order. */
    using OpList = std::vector<decltype(Graph::Node::op)>;

    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NN::NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Holds the batch size every trace here uses. */
    constexpr int64_t K_BATCH = 2;

    /** @brief Holds the row count of the YOLO26n model file. */
    constexpr std::size_t K_YOLO_ROWS = 24;

    /** @brief Holds the row of the YOLO26n file that names the missing attention block. */
    constexpr std::size_t K_ATTENTION_ROW = 10;

    /** @brief Holds the row of the YOLO26n file whose inner block is missing. */
    constexpr std::size_t K_ATTENTION_INNER_ROW = 22;

    /** @brief Holds the absolute bound on a real compared here; each is exact in the file. */
    constexpr double K_TOLERANCE = 1e-12;

    /** @brief Describes one node registry row's minimal object and the shapes around it. */
    struct MinimalCase
    {
        std::string_view type;       /**< The registry key. */
        std::string_view attributes; /**< The row without its "type" and "from". */
        std::vector<Shape> inputs;   /**< The per-sample shapes the row reads. */
        Shape output;                /**< The per-sample shape of the node's one port. */
    };

    /** @brief Lists a minimal object for every row of K_NODE_REGISTRY, in the table's order. */
    const std::vector<MinimalCase> K_MINIMAL_CASES = {
        {"batchnorm2d", "{}", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"concat", "{}", {Shape{4, 4, 2}, Shape{4, 4, 3}}, Shape{4, 4, 5}},
        {"conv", R"({"filters": 5})", {Shape{8, 8, 2}}, Shape{8, 8, 5}},
        {"bottleneck", R"({"filters": 4, "hidden": 2})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"c3k", R"({"filters": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"c3k2", R"({"filters": 8, "hidden": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 8}},
        {"sppf", R"({"filters": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
    };

    /** @brief Describes one LAYER_REGISTRY row's minimal object and the shape it reads. */
    struct LayerCase
    {
        std::string_view type;       /**< The registry key. */
        std::string_view attributes; /**< The row without its "type" and "from". */
        Shape input;                 /**< The per-sample shape the row reads. */
    };

    /** @brief Lists a minimal object for every row of LAYER_REGISTRY, in the table's order. */
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
    };

    /** @brief Holds a model whose row 2 joins the model input with the row above it. */
    constexpr std::string_view K_INPUT_JOINING_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8, 8, 3]},
        "layers": [
            {"type": "conv", "filters": 4},
            {"type": "silu"},
            {"type": "concat", "from": ["input", -1]}
        ]
    })";

    /** @brief Holds a branch and a merge as a model file: rows 0 and 2 meet in row 3. */
    constexpr std::string_view K_BRANCHING_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [16, 16, 3]},
        "defaults": {"conv": {"bn_epsilon": 0.001, "bn_momentum": 0.03}},
        "layers": [
            {"type": "conv", "filters": 8, "kernel": 3, "stride": 2},
            {"type": "conv", "filters": 16, "kernel": 3, "stride": 2},
            {"type": "upsample", "scale": 2},
            {"type": "concat", "from": [-1, 0]},
            {"type": "conv", "filters": 8, "kernel": 1}
        ]
    })";

    /** @brief Holds what one training trace of a model graph left behind. */
    struct TraceSummary
    {
        OpList ops;                       /**< The operation of every traced node. */
        std::vector<std::string> outputs; /**< Each published output as "name [extents]". */
        std::vector<std::string> state;   /**< Each piece of state, as record() writes it. */
    };

    /** @brief Parses @p text as the attributes of one row. */
    Config::Value attributes_of(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }

    /** @brief Parses @p text as a model file. */
    Config::ModelSpec model_of(std::string_view text)
    {
        return Config::parse_model(Config::parse_json(text, "<test>"), "<test>");
    }

    /** @brief Views @p shapes as the inputs of one row. */
    NN::ShapeSpan span_of(const std::vector<Shape>& shapes)
    {
        return NN::ShapeSpan(shapes.data(), shapes.size());
    }

    /** @brief Builds the node @p minimal describes, as "layer 0 '<type>'". */
    Result<std::unique_ptr<NN::Node>> make_minimal(const MinimalCase& minimal,
                                                   std::string_view attributes)
    {
        const std::string context = "layer 0 '" + std::string(minimal.type) + "'";
        return NN::make_node(minimal.type, span_of(minimal.inputs), attributes_of(attributes),
                             context);
    }

    /** @brief Writes one piece of state as "name [extents] kind", for a readable mismatch. */
    std::string record(const std::string& name, const Shape& shape, bool is_buffer)
    {
        return name + " " + shape.to_string() + (is_buffer ? " buffer" : " parameter");
    }

    /** @brief Lists every piece of state of @p model under @p prefix, in walk order. */
    std::vector<std::string> state_records(ModelGraph& model, std::string_view prefix)
    {
        std::vector<std::string> records;
        const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
        {
            records.push_back(record(name, state.shape(), false));
        };
        const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
        {
            records.push_back(record(name, state.shape(), true));
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        model.walk_state(prefix, visitor);

        return records;
    }

    /**
     * @brief Traces @p model in Training mode over K_BATCH samples and summarises the result.
     * @param prefix The prefix the state names are listed under.
     */
    Result<TraceSummary> trace_summary(ModelGraph& model, const Shape& sample,
                                       std::string_view prefix)
    {
        std::vector<int64_t> extents{K_BATCH};
        for (int32_t axis = 0; axis < sample.rank(); ++axis)
            extents.push_back(sample[axis]);

        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(
            Shape(SushiRuntime::span<const int64_t>(extents.data(), extents.size())), "x");
        NN::BufferUpdateList updates;

        SA_TRY_ASSIGN(const NN::ModelOutputs outputs,
                      model.trace(NN::Trace::training(builder, updates), x));

        TraceSummary summary;
        for (const Graph::Node& node : graph.nodes())
            summary.ops.push_back(node.op);

        for (const NN::ModelOutputs::Entry& output : outputs.all())
            summary.outputs.push_back(output.first + " " +
                                      graph.value(output.second).shape.to_string());

        summary.state = state_records(model, prefix);
        return summary;
    }

    /** @brief Builds by hand the graph K_BRANCHING_MODEL describes, rows named by index. */
    Result<ModelGraph> build_branching_model_by_hand()
    {
        ModelGraph model(Shape{16, 16, 3});

        SA_TRY_ASSIGN(const NodeRef stem,
                      model.add("0", NN::Conv(3, 8, 3, 2, K_YOLO_NORM), model.input()));
        SA_TRY_ASSIGN(const NodeRef down, model.add("1", NN::Conv(8, 16, 3, 2, K_YOLO_NORM), stem));
        SA_TRY_ASSIGN(const NodeRef up, model.add("2", NN::Upsample2d{2}, down));

        const NodeRef both[] = {up, stem};
        const Shape shapes[] = {model.shape_of(up), model.shape_of(stem)};
        SA_TRY_ASSIGN(std::unique_ptr<NN::Concat> concat,
                      NN::Concat::create(NN::ShapeSpan(shapes, 2)));
        SA_TRY_ASSIGN(const NodeRef merged,
                      model.add("3", std::unique_ptr<NN::Node>(std::move(concat)),
                                SushiRuntime::span<const NodeRef>(both, 2)));

        SA_TRY_ASSIGN(const NodeRef mix,
                      model.add("4", NN::Conv(24, 8, 1, 1, K_YOLO_NORM), merged));
        SA_TRY(model.output_ports(mix));

        return Result<ModelGraph>(std::move(model));
    }

    /** @brief Returns the repository root, or an empty path when the fixture folder is unknown. */
    std::filesystem::path repository_root()
    {
        const std::string fixtures = Test::fixture_directory();
        if (fixtures.empty())
            return {};

        // SA_TEST_FIXTURE_DIR is <repository>/tests/fixtures.
        return std::filesystem::path(fixtures).parent_path().parent_path();
    }

    /** @brief Returns the path of the YOLO26n model file under @p root. */
    std::filesystem::path yolo_model_path(const std::filesystem::path& root)
    {
        return root / "examples" / "yolo26" / "model.json";
    }

    /** @brief Returns the path of the YOLO26 reference manifest under @p root. */
    std::filesystem::path manifest_path(const std::filesystem::path& root)
    {
        return root / "tests" / "fixtures" / Test::K_YOLO26_FIXTURE_FOLDER / "MANIFEST.json";
    }

    /** @brief Returns the entry of the manifest's "files" whose name is @p name. */
    const Config::Value& manifest_file(const Config::Value& manifest, std::string_view name)
    {
        for (const Config::Value& file : manifest.at("files", "manifest").as_array("files"))
        {
            if (file.at("name", "file").as_string("name") == name)
                return file;
        }

        SA_THROW_IF(true, "MANIFEST.json lists no file named '{}'", name);
        return manifest;
    }

    /** @brief Returns the row index of the manifest record @p name, "model.<row>.<rest>". */
    std::size_t record_row(const std::string& name)
    {
        const std::size_t first = name.find('.') + 1;
        const std::size_t length = name.find('.', first) - first;
        return static_cast<std::size_t>(std::stoul(name.substr(first, length)));
    }

    /**
     * @brief Lists the weights.sacp records of rows @p first to @p last, in manifest order.
     * @return Each record as record() writes it; a name ending in a running statistic is a
     *         buffer.
     */
    std::vector<std::string> manifest_records(const Config::Value& manifest, std::size_t first,
                                              std::size_t last)
    {
        std::vector<std::string> records;
        const Config::Value& weights = manifest_file(manifest, "weights.sacp");
        for (const Config::Value& tensor : weights.at("tensors", "weights").as_array("tensors"))
        {
            const std::string& name = tensor.at("name", "tensor").as_string("name");
            const std::size_t row = record_row(name);
            if (row < first || row > last)
                continue;

            std::vector<int64_t> extents;
            for (const Config::Value& extent : tensor.at("shape", name).as_array("shape"))
                extents.push_back(extent.as_int("extent"));

            const bool is_buffer = name.find(".running_") != std::string::npos;
            records.push_back(record(
                name, Shape(SushiRuntime::span<const int64_t>(extents.data(), extents.size())),
                is_buffer));
        }

        return records;
    }

    /** @brief Replaces row @p row of @p model with a row of @p type that reads the same sources. */
    void stand_in(Config::ModelSpec& model, std::size_t row, std::string_view type,
                  std::string_view attributes)
    {
        model.layers[row].type = std::string(type);
        model.layers[row].attributes = attributes_of(attributes);
    }

    /** @brief Keeps the first @p count rows of @p model and publishes the last of them whole. */
    void keep_first_rows(Config::ModelSpec& model, std::size_t count)
    {
        model.layers.resize(count);

        Config::OutputSpec last;
        last.node = static_cast<std::int64_t>(count) - 1;
        model.outputs.assign(1, last);
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

    /** @brief Lists the per-sample shape of rows 0 to 21 of YOLO26n over a [640, 640, 3] input. */
    const std::vector<Shape> K_YOLO_ROW_SHAPES = {
        Shape{320, 320, 16}, Shape{160, 160, 32}, Shape{160, 160, 64}, Shape{80, 80, 64},
        Shape{80, 80, 128},  Shape{40, 40, 128},  Shape{40, 40, 128},  Shape{20, 20, 256},
        Shape{20, 20, 256},  Shape{20, 20, 256},  Shape{20, 20, 256},  Shape{40, 40, 256},
        Shape{40, 40, 384},  Shape{40, 40, 128},  Shape{80, 80, 128},  Shape{80, 80, 256},
        Shape{80, 80, 64},   Shape{40, 40, 64},   Shape{40, 40, 192},  Shape{40, 40, 128},
        Shape{20, 20, 128},  Shape{20, 20, 384},
    };

} // namespace

/**
 * @brief Declares @p root as the repository root, or skips the test when it is unknown.
 * @pre Used as a statement in a function returning void.
 */
#define SA_FIND_REPOSITORY_OR_SKIP(root)                                               \
    const std::filesystem::path root = repository_root();                              \
    if (root.empty())                                                                  \
        GTEST_SKIP() << "SA_TEST_FIXTURE_DIR is not defined for this binary, so the "  \
                        "repository's files cannot be located"

/** @brief Tests that K_NODE_REGISTRY spells its seven type strings as model files do, in order. */
TEST(Unit_NodeRegistry, TheTypeStringsAreTheOnesModelFilesSpell)
{
    const std::vector<std::string_view> expected = {
        "batchnorm2d", "concat", "conv", "bottleneck", "c3k", "c3k2", "sppf",
    };

    std::vector<std::string_view> spelled;
    for (const NN::NodeEntry& entry : NN::K_NODE_REGISTRY)
        spelled.push_back(entry.type);

    EXPECT_EQ(spelled, expected);
}

/** @brief Tests that no type string is in both tables, so make_node's order decides nothing. */
TEST(Unit_NodeRegistry, NoTypeIsInBothTables)
{
    for (const NN::NodeEntry& node : NN::K_NODE_REGISTRY)
    {
        for (const NN::LayerEntry& layer : NN::LAYER_REGISTRY)
            EXPECT_NE(node.type, layer.type);
    }
}

/** @brief Tests that every row of K_NODE_REGISTRY has a minimal case here, in table order. */
TEST(Unit_NodeRegistry, EveryRowHasAMinimalCase)
{
    ASSERT_EQ(K_MINIMAL_CASES.size(), NN::K_NODE_REGISTRY.size());

    for (std::size_t index = 0; index < NN::K_NODE_REGISTRY.size(); ++index)
        EXPECT_EQ(K_MINIMAL_CASES[index].type, NN::K_NODE_REGISTRY[index].type);
}

/** @brief Tests that each row builds a one-port node of the expected shape from its case. */
TEST(Unit_NodeRegistry, EachRowBuildsFromAMinimalObject)
{
    for (const MinimalCase& minimal : K_MINIMAL_CASES)
    {
        SCOPED_TRACE(std::string(minimal.type));

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                                make_minimal(minimal, minimal.attributes));

        ASSERT_NE(node, nullptr);
        ASSERT_EQ(node->ports().size(), 1u);
        EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
        EXPECT_EQ(node->ports()[0].shape, minimal.output);
    }
}

/** @brief Tests that each row refuses a key it does not know and names the row and the key. */
TEST(Unit_NodeRegistry, EachRowRejectsAnUnknownKey)
{
    for (const MinimalCase& minimal : K_MINIMAL_CASES)
    {
        const std::string context = "layer 0 '" + std::string(minimal.type) + "'";

        SA_EXPECT_FAILURE(make_minimal(minimal, R"({"no_such_key": 1})"),
                          StatusCode::InvalidDescription, context, "'no_such_key'");
    }
}

/** @brief Tests that a concat row over inputs of unequal H fails and names the row. */
TEST(Unit_NodeRegistry, AConcatRowNamesItselfWhenItsInputsDisagree)
{
    const std::vector<Shape> inputs = {Shape{4, 4, 2}, Shape{8, 4, 2}};

    SA_EXPECT_FAILURE(
        NN::make_node("concat", span_of(inputs), attributes_of("{}"), "layer 7 'concat'"),
        StatusCode::InvalidDescription, "layer 7 'concat'", "[8, 4, 2]", "H and W");
}

/** @brief Tests that a K_NODE_REGISTRY type builds the block's own node, not a LayerNode. */
TEST(Unit_NodeRegistry, ANodeTypeBuildsItsOwnNode)
{
    const std::vector<Shape> inputs = {Shape{8, 8, 2}};

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_node("conv", span_of(inputs),
                                          attributes_of(R"({"filters": 5})"),
                                          "layer 0 'conv'"));

    EXPECT_NE(dynamic_cast<NN::ModuleNode<NN::Conv>*>(node.get()), nullptr);
    EXPECT_EQ(dynamic_cast<NN::LayerNode*>(node.get()), nullptr);
}

/** @brief Tests that every LAYER_REGISTRY type is reached through make_node as a LayerNode. */
TEST(Unit_NodeRegistry, ALayerTypeIsWrappedInALayerNode)
{
    const std::vector<Shape> inputs = {Shape{4, 4, 2}};

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_node("upsample", span_of(inputs),
                                          attributes_of(R"({"scale": 2})"),
                                          "layer 0 'upsample'"));

    ASSERT_NE(dynamic_cast<NN::LayerNode*>(node.get()), nullptr);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 8, 2}));

    ASSERT_EQ(K_LAYER_CASES.size(), NN::LAYER_REGISTRY.size());
    for (std::size_t index = 0; index < NN::LAYER_REGISTRY.size(); ++index)
    {
        const LayerCase& minimal = K_LAYER_CASES[index];
        SCOPED_TRACE(std::string(minimal.type));
        ASSERT_EQ(minimal.type, NN::LAYER_REGISTRY[index].type);

        const std::vector<Shape> read = {minimal.input};
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> built,
                                NN::make_node(minimal.type, span_of(read),
                                              attributes_of(minimal.attributes), "layer 0"));
        EXPECT_NE(dynamic_cast<NN::LayerNode*>(built.get()), nullptr);

        // An unknown key is refused by the layer's own factory, so the type was found.
        SA_EXPECT_FAILURE(NN::make_node(minimal.type, span_of(read),
                                        attributes_of(R"({"no_such_key": 1})"), "layer 0"),
                          StatusCode::InvalidDescription, "'no_such_key'");
    }
}

/** @brief Tests that a layer type given two inputs fails and names the count and the type. */
TEST(Unit_NodeRegistry, ALayerTypeRefusesTwoInputs)
{
    const std::vector<Shape> inputs = {Shape{6}, Shape{6}};

    SA_EXPECT_FAILURE(
        NN::make_node("relu", span_of(inputs), attributes_of("{}"), "layer 3 'relu'"),
        StatusCode::InvalidDescription, "layer 3 'relu'", "reads 2 values", "reads one");
}

/** @brief Tests that what a layer factory throws comes back as a status with its message. */
TEST(Unit_NodeRegistry, ALayerFactorysRefusalComesBackAsAStatus)
{
    const std::vector<Shape> vector = {Shape{6}};
    const std::vector<Shape> image = {Shape{8, 8, 2}};

    SA_EXPECT_FAILURE(
        NN::make_node("linear", span_of(vector), attributes_of("{}"), "layer 4 'linear'"),
        StatusCode::InvalidDescription, "layer 4 'linear'", "'units'");
    SA_EXPECT_FAILURE(NN::make_node("linear", span_of(image), attributes_of(R"({"units": 3})"),
                                    "layer 4 'linear'"),
                      StatusCode::InvalidDescription, "layer 4 'linear'", "[8, 8, 2]", "'flatten'");
}

/** @brief Tests that an unknown type fails naming the row, the type and both tables' types. */
TEST(Unit_NodeRegistry, AnUnknownTypeListsBothTables)
{
    const std::vector<Shape> inputs = {Shape{8, 8, 2}};
    const Result<std::unique_ptr<NN::Node>> made = NN::make_node(
        "conv3d", span_of(inputs), attributes_of("{}"), "layer 5 'conv3d'");

    SA_EXPECT_FAILURE(made, StatusCode::InvalidDescription, "layer 5 'conv3d'",
                      "the type 'conv3d'", "does not have", "node types", "layer types");

    for (const NN::NodeEntry& entry : NN::K_NODE_REGISTRY)
        SA_EXPECT_FAILURE(made, StatusCode::InvalidDescription,
                          "'" + std::string(entry.type) + "'");

    for (const NN::LayerEntry& entry : NN::LAYER_REGISTRY)
        SA_EXPECT_FAILURE(made, StatusCode::InvalidDescription,
                          "'" + std::string(entry.type) + "'");
}

/**
 * @brief Tests that build_model of a file and the same graph built by hand trace to equal
 *        node lists, equal outputs and equal state names and shapes.
 */
TEST(Unit_NodeRegistry, AFileAndAHandBuiltGraphAgree)
{
    const Config::ModelSpec spec = model_of(K_BRANCHING_MODEL);
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph from_file, NN::build_model(spec));
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph by_hand, build_branching_model_by_hand());

    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary file_trace,
                            trace_summary(from_file, spec.input_shape, "model"));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary hand_trace,
                            trace_summary(by_hand, spec.input_shape, "model"));

    ASSERT_EQ(from_file.size(), 5u);
    EXPECT_EQ(from_file.size(), by_hand.size());
    EXPECT_FALSE(file_trace.ops.empty());
    EXPECT_EQ(file_trace.ops, hand_trace.ops);
    EXPECT_EQ(file_trace.outputs, hand_trace.outputs);
    EXPECT_EQ(file_trace.outputs, (std::vector<std::string>{"output [2, 8, 8, 8]"}));

    // Three Conv rows, each one weight, two norm parameters and two running buffers.
    ASSERT_EQ(file_trace.state.size(), 15u);
    EXPECT_EQ(file_trace.state, hand_trace.state);
    EXPECT_EQ(file_trace.state.front(), "model.0.conv.weight [8, 3, 3, 3] parameter");
    EXPECT_EQ(file_trace.state.back(), "model.4.bn.running_var [8] buffer");
}

/** @brief Tests that a row built from a file reads its sources in the order the file gave. */
TEST(Unit_NodeRegistry, ARowReadsItsSourcesInFileOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelGraph model, NN::build_model(model_of(K_BRANCHING_MODEL)));

    // Row 3 joins row 2's 16 channels, then row 0's 8, at row 0's 8 x 8 extent.
    EXPECT_EQ(model.node(3).ports()[0].shape, (Shape{8, 8, 24}));
    EXPECT_EQ(model.node(2).ports()[0].shape, (Shape{8, 8, 16}));
    EXPECT_EQ(model.node(1).ports()[0].shape, (Shape{4, 4, 16}));
}

/** @brief Tests that "outputs" publishes a named port under the stated name, in file order. */
TEST(Unit_NodeRegistry, OutputsArePublishedUnderTheirStatedNames)
{
    Config::ModelSpec spec = model_of(K_BRANCHING_MODEL);
    spec.outputs = {named_output(4, "mixed"), named_output(0, "stem")};

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, "model"));

    EXPECT_EQ(trace.outputs,
              (std::vector<std::string>{"mixed [2, 8, 8, 8]", "stem [2, 8, 8, 8]"}));
}

/** @brief Tests that a port entry with no name is published under the port's own name. */
TEST(Unit_NodeRegistry, AnOutputWithoutANameKeepsThePortsName)
{
    Config::ModelSpec spec = model_of(K_BRANCHING_MODEL);
    spec.outputs = {named_output(1, "")};

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, "model"));

    EXPECT_EQ(trace.outputs, (std::vector<std::string>{"output [2, 4, 4, 16]"}));
}

/** @brief Tests that a source may name a port, and that an unknown port names the row. */
TEST(Unit_NodeRegistry, ASourceMayNameAPort)
{
    Config::ModelSpec spec = model_of(K_BRANCHING_MODEL);
    spec.layers[1].from[0].port = std::string(NN::K_SINGLE_PORT_NAME);
    SA_EXPECT_OK(NN::build_model(spec));

    spec.layers[1].from[0].port = "no_such_port";
    SA_EXPECT_FAILURE(NN::build_model(spec), StatusCode::InvalidDescription, "layer 1 'conv'",
                      "'no_such_port'");
}

/** @brief Tests that a row below row 0 reads the model input, joined in the file's order. */
TEST(Unit_NodeRegistry, ALaterRowMayReadTheModelInput)
{
    const Config::ModelSpec spec = model_of(K_INPUT_JOINING_MODEL);
    ASSERT_EQ(spec.layers[2].from.size(), 2u);
    ASSERT_EQ(spec.layers[2].from[0].node, Config::K_MODEL_INPUT);

    SA_ASSERT_OK_AND_ASSIGN(const ModelGraph model, NN::build_model(spec));

    // Row 2 joins the input's 3 channels, then row 1's 4.
    EXPECT_EQ(model.node(2).ports()[0].shape, (Shape{8, 8, 7}));
}

/** @brief Tests that a source naming a port of the model input is refused and names the row. */
TEST(Unit_NodeRegistry, ASourceCannotNameAPortOfTheModelInput)
{
    Config::ModelSpec spec = model_of(K_INPUT_JOINING_MODEL);
    spec.layers[2].from[0].port = std::string(NN::K_SINGLE_PORT_NAME);

    SA_EXPECT_FAILURE(NN::build_model(spec), StatusCode::InvalidDescription, "layer 2 'concat'",
                      "the model input has no port", "'output'");
}

/** @brief Tests that a row the factory refuses fails the build with the row's index and type. */
TEST(Unit_NodeRegistry, ARefusedRowNamesItsIndexAndType)
{
    Config::ModelSpec spec = model_of(K_BRANCHING_MODEL);
    spec.layers[4].attributes = attributes_of(R"({"filters": 0})");

    SA_EXPECT_FAILURE(NN::build_model(spec), StatusCode::InvalidDescription, "layer 4 'conv'",
                      "'filters'", "positive");
}

/** @brief Tests that a spec built by hand is refused when a row reads a row not above it. */
TEST(Unit_NodeRegistry, ARowThatReadsARowNotAboveItIsRefused)
{
    Config::ModelSpec forward = model_of(K_BRANCHING_MODEL);
    forward.layers[1].from[0].node = 3;
    SA_EXPECT_FAILURE(NN::build_model(forward), StatusCode::InvalidDescription, "layer 1 'conv'",
                      "row 3", "not above");

    Config::ModelSpec negative = model_of(K_BRANCHING_MODEL);
    negative.layers[2].from[0].node = -2;
    SA_EXPECT_FAILURE(NN::build_model(negative), StatusCode::InvalidDescription,
                      "layer 2 'upsample'", "row -2");

    Config::ModelSpec sourceless = model_of(K_BRANCHING_MODEL);
    sourceless.layers[2].from.clear();
    SA_EXPECT_FAILURE(NN::build_model(sourceless), StatusCode::InvalidDescription,
                      "layer 2 'upsample'", "no source");
}

/** @brief Tests that a hand-built spec with no outputs publishes its last row, as a file does. */
TEST(Unit_NodeRegistry, AnEmptyOutputListPublishesTheLastRowWhole)
{
    const Config::ModelSpec parsed = model_of(K_BRANCHING_MODEL);
    Config::ModelSpec silent = parsed;
    silent.outputs.clear();

    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph from_file, NN::build_model(parsed));
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph by_hand, NN::build_model(silent));

    EXPECT_EQ(by_hand.output_shape(parsed.input_shape), (Shape{8, 8, 8}));
    EXPECT_EQ(by_hand.output_shape(parsed.input_shape),
              from_file.output_shape(parsed.input_shape));
}

/** @brief Tests that a spec with no layer or a wrong output entry is refused. */
TEST(Unit_NodeRegistry, AWrongOutputListIsRefused)
{
    Config::ModelSpec empty = model_of(K_BRANCHING_MODEL);
    empty.layers.clear();
    SA_EXPECT_FAILURE(NN::build_model(empty), StatusCode::InvalidDescription, "no layers");

    Config::ModelSpec beyond = model_of(K_BRANCHING_MODEL);
    beyond.outputs = {named_output(5, "late")};
    SA_EXPECT_FAILURE(NN::build_model(beyond), StatusCode::InvalidDescription, "output 0",
                      "row 5", "rows 0 to 4");

    Config::ModelSpec portless = model_of(K_BRANCHING_MODEL);
    portless.outputs = {named_output(4, "mixed")};
    portless.outputs[0].port.clear();
    SA_EXPECT_FAILURE(NN::build_model(portless), StatusCode::InvalidDescription, "output 0",
                      "'mixed'", "without a port");

    Config::ModelSpec unknown = model_of(K_BRANCHING_MODEL);
    unknown.outputs = {named_output(4, "mixed")};
    unknown.outputs[0].port = "no_such_port";
    SA_EXPECT_FAILURE(NN::build_model(unknown), StatusCode::InvalidDescription, "output 0",
                      "'no_such_port'");

    Config::ModelSpec twice = model_of(K_BRANCHING_MODEL);
    twice.outputs = {named_output(4, "mixed"), named_output(0, "mixed")};
    SA_EXPECT_FAILURE(NN::build_model(twice), StatusCode::InvalidDescription, "output 1",
                      "'mixed'");
}

/**
 * @brief Tests that the YOLO26n file parses to its 24 rows: the types in order, the sources
 *        of the four joins and of the head, the merged defaults and the one output.
 */
TEST(Unit_NodeRegistry, TheYolo26FileHoldsItsTwentyFourRows)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    const Config::ModelSpec spec = Config::load_model(yolo_model_path(root));

    const std::vector<std::string> expected_types = {
        "conv", "conv",     "c3k2",   "conv", "c3k2",     "conv",   "c3k2", "conv",
        "c3k2", "sppf",     "c2psa",  "upsample", "concat", "c3k2", "upsample", "concat",
        "c3k2", "conv",     "concat", "c3k2", "conv",     "concat", "c3k2", "detect",
    };
    std::vector<std::string> types;
    for (const Config::LayerSpec& layer : spec.layers)
        types.push_back(layer.type);

    ASSERT_EQ(spec.layers.size(), K_YOLO_ROWS);
    EXPECT_EQ(types, expected_types);
    EXPECT_EQ(spec.input_shape, (Shape{640, 640, 3}));

    const auto sources_of = [&spec](std::size_t row)
    {
        std::vector<std::int64_t> sources;
        for (const Config::SourceSpec& source : spec.layers[row].from)
            sources.push_back(source.node);
        return sources;
    };
    EXPECT_EQ(sources_of(0), (std::vector<std::int64_t>{Config::K_MODEL_INPUT}));
    EXPECT_EQ(sources_of(12), (std::vector<std::int64_t>{11, 6}));
    EXPECT_EQ(sources_of(15), (std::vector<std::int64_t>{14, 4}));
    EXPECT_EQ(sources_of(18), (std::vector<std::int64_t>{17, 13}));
    EXPECT_EQ(sources_of(21), (std::vector<std::int64_t>{20, 10}));
    EXPECT_EQ(sources_of(23), (std::vector<std::int64_t>{16, 19, 22}));

    EXPECT_NEAR(spec.layers[0].attributes.at("bn_epsilon", "row 0").as_real("bn_epsilon"), 0.001,
                K_TOLERANCE);
    EXPECT_NEAR(spec.layers[9].attributes.at("bn_momentum", "row 9").as_real("bn_momentum"), 0.03,
                K_TOLERANCE);
    EXPECT_FALSE(spec.layers[11].attributes.has("bn_epsilon"));

    ASSERT_EQ(spec.outputs.size(), 1u);
    EXPECT_EQ(spec.outputs[0].node, 23);
    EXPECT_TRUE(spec.outputs[0].port.empty());
}

/**
 * @brief Tests that the YOLO26n file fails to build at row 10, then row 22, then row 23, each
 *        with a status that names the row and the missing type or inner block.
 */
TEST(Unit_NodeRegistry, TheYolo26FileNamesEachMissingBlock)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));

    SA_EXPECT_FAILURE(NN::build_model(spec), StatusCode::InvalidDescription, "layer 10 'c2psa'",
                      "the type 'c2psa'", "does not have", "'c3k2'", "'upsample'");

    // A silu row keeps [20, 20, 256], which is what C2PSA(256, 256) keeps.
    stand_in(spec, K_ATTENTION_ROW, "silu", "{}");
    SA_EXPECT_FAILURE(NN::build_model(spec), StatusCode::InvalidDescription, "layer 22 'c3k2'",
                      "'inner'", "'bottleneck_psa'", "bottleneck, c3k");

    stand_in(spec, K_ATTENTION_INNER_ROW, "conv",
             R"({"filters": 256, "bn_epsilon": 0.001, "bn_momentum": 0.03})");
    SA_EXPECT_FAILURE(NN::build_model(spec), StatusCode::InvalidDescription, "layer 23 'detect'",
                      "the type 'detect'", "does not have");
}

/**
 * @brief Tests that rows 2 and 4 of the YOLO26n file each trace one ADD, the shortcut of the
 *        one Bottleneck a C3k2 of the pinned Ultralytics commit builds with shortcut=True.
 */
TEST(Unit_NodeRegistry, Yolo26RowsTwoAndFourAddTheirBottlenecksInput)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    const std::size_t rows_through_four = 5;
    keep_first_rows(spec, rows_through_four);

    EXPECT_TRUE(spec.layers[2].attributes.at("shortcut", "row 2").as_bool("shortcut"));
    EXPECT_TRUE(spec.layers[4].attributes.at("shortcut", "row 4").as_bool("shortcut"));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary traced,
                            trace_summary(model, spec.input_shape, "model"));

    EXPECT_EQ(std::count(traced.ops.begin(), traced.ops.end(), Ops::ADD.id), 2);
}

/**
 * @brief Tests that rows 0 to 9 of the YOLO26n file build and yield exactly the records
 *        MANIFEST.json lists under model.0 to model.9: names, shapes and order.
 */
TEST(Unit_NodeRegistry, Yolo26RowsZeroToNineMatchTheManifest)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    keep_first_rows(spec, K_ATTENTION_ROW);
    const Config::Value manifest = Config::parse_json_file(manifest_path(root));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, "model"));

    const std::vector<std::string> expected = manifest_records(manifest, 0, 9);
    ASSERT_EQ(expected.size(), 165u);
    EXPECT_EQ(trace.state, expected);
    EXPECT_EQ(trace.outputs, (std::vector<std::string>{"output [2, 20, 20, 256]"}));
}

/**
 * @brief Tests that rows 0 to 9 and 11 to 21 of the YOLO26n file build to the per-sample
 *        shapes of the reference, with a silu row standing in for row 10.
 */
TEST(Unit_NodeRegistry, Yolo26RowsBuildToTheReferenceShapes)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    stand_in(spec, K_ATTENTION_ROW, "silu", "{}");
    keep_first_rows(spec, K_ATTENTION_INNER_ROW);

    SA_ASSERT_OK_AND_ASSIGN(const ModelGraph model, NN::build_model(spec));

    ASSERT_EQ(model.size(), K_YOLO_ROW_SHAPES.size());
    for (std::size_t row = 0; row < model.size(); ++row)
    {
        SCOPED_TRACE("row " + std::to_string(row));
        ASSERT_EQ(model.node(row).ports().size(), 1u);
        EXPECT_EQ(model.node(row).ports()[0].shape, K_YOLO_ROW_SHAPES[row]);
    }
}

/**
 * @brief Tests that rows 16 and 19 trace to the extents, widths and strides MANIFEST.json
 *        records for feature levels 0 and 1, and that rows 11 to 21 yield the manifest's
 *        records for rows 13 to 20.
 */
TEST(Unit_NodeRegistry, Yolo26NeckTracesToTheReferenceLevels)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    stand_in(spec, K_ATTENTION_ROW, "silu", "{}");
    keep_first_rows(spec, K_ATTENTION_INNER_ROW);
    spec.outputs = {named_output(16, "level0"), named_output(19, "level1"),
                    named_output(21, "joined")};
    const Config::Value manifest = Config::parse_json_file(manifest_path(root));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, "model"));

    const std::vector<Config::Value>& levels = manifest_file(manifest, "forward.sacp")
                                                   .at("context", "forward.sacp")
                                                   .at("levels", "context")
                                                   .as_array("levels");
    ASSERT_EQ(levels.size(), 3u);
    ASSERT_EQ(trace.outputs.size(), 3u);
    for (std::size_t level = 0; level < 2; ++level)
    {
        SCOPED_TRACE("level " + std::to_string(level));
        const std::int64_t height = levels[level].at("height", "level").as_int("height");
        const std::int64_t width = levels[level].at("width", "level").as_int("width");
        const std::int64_t channels = levels[level].at("channels", "level").as_int("channels");
        const Shape expected{K_BATCH, height, width, channels};

        EXPECT_EQ(trace.outputs[level],
                  "level" + std::to_string(level) + " " + expected.to_string());

        EXPECT_NEAR(levels[level].at("stride", "level").as_real("stride") *
                        static_cast<double>(height),
                    static_cast<double>(spec.input_shape[0]), K_TOLERANCE);
    }
    EXPECT_EQ(trace.outputs[2], "joined [2, 20, 20, 384]");

    std::vector<std::string> expected = manifest_records(manifest, 0, 9);
    const std::vector<std::string> neck = manifest_records(manifest, 11, 21);
    ASSERT_EQ(neck.size(), 145u);
    expected.insert(expected.end(), neck.begin(), neck.end());
    EXPECT_EQ(trace.state, expected);
}
