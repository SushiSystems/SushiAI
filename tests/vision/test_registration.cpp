/**************************************************************************/
/* test_registration.cpp                                                  */
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
 * @file test_registration.cpp
 * @brief Tests register_nodes and register_datasets, and builds the YOLO26n file over them.
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
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/registration.hpp>
#include "../common/reference_fixture.hpp"
#include "../common/test_status.hpp"
#include "../common/model_text.hpp"
#include "../common/test_utils.hpp"

using namespace SushiAI;
using NN::ModelGraph;

namespace
{
    using Test::K_TRACE_BATCH;
    using Test::MinimalCase;
    using Test::TraceSummary;
    using Test::attributes_from;
    using Test::make_minimal;
    using Test::model_from;
    using Test::state_record;
    using Test::trace_summary;
    using Test::types_of;

    /** @brief Holds the row count of the YOLO26n model file. */
    constexpr std::size_t K_YOLO_ROWS = 24;

    /** @brief Holds the row of the YOLO26n file that names the missing attention block. */
    constexpr std::size_t K_ATTENTION_ROW = 10;

    /** @brief Holds the row of the YOLO26n file whose inner block is missing. */
    constexpr std::size_t K_ATTENTION_INNER_ROW = 22;

    /** @brief Holds the absolute bound on a real compared here; each is exact in the file. */
    constexpr double K_TOLERANCE = 1e-12;

    /** @brief Lists a minimal object for every row of register_nodes, in the rows' order. */
    const std::vector<MinimalCase> K_MINIMAL_CASES = {
        {"conv", R"({"filters": 5})", {Shape{8, 8, 2}}, Shape{8, 8, 5}},
        {"bottleneck", R"({"filters": 4, "hidden": 2})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"c3k", R"({"filters": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"c3k2", R"({"filters": 8, "hidden": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 8}},
        {"sppf", R"({"filters": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
    };

    /** @brief Holds a model whose row 0 is a conv block. */
    constexpr std::string_view K_CONV_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [16, 16, 3]},
        "layers": [
            {"type": "conv", "filters": 8, "kernel": 3, "stride": 2}
        ]
    })";

    /** @brief Builds a registry from Vision::register_nodes alone. */
    Result<NN::NodeRegistry> blocks_alone()
    {
        NN::NodeRegistry registry;
        SA_TRY(Vision::register_nodes(registry));

        return Result<NN::NodeRegistry>(std::move(registry));
    }

    /** @brief Builds a registry from NN::register_layers, then Vision::register_nodes. */
    Result<NN::NodeRegistry> composed()
    {
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY(Vision::register_nodes(registry));

        return Result<NN::NodeRegistry>(std::move(registry));
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
     * @return Each record as state_record() writes it; a name ending in a running statistic is a
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
            records.push_back(state_record(
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
        model.layers[row].attributes = attributes_from(attributes);
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

/** @brief Tests that register_nodes adds conv, bottleneck, c3k, c3k2 and sppf, in order. */
TEST(Unit_VisionRegistration, RegisterNodesAddsTheFiveBlocks)
{
    const std::vector<std::string_view> expected = {
        "conv", "bottleneck", "c3k", "c3k2", "sppf",
    };

    NN::NodeRegistry registry;
    SA_EXPECT_OK(Vision::register_nodes(registry));

    EXPECT_EQ(types_of(registry.nodes()), expected);
    EXPECT_TRUE(registry.layers().empty());
}

/** @brief Tests that register_datasets adds the one row "mnist", with a factory. */
TEST(Unit_VisionRegistration, RegisterDatasetsAddsMnist)
{
    const std::vector<std::string_view> expected = {"mnist"};

    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Vision::register_datasets(registry));

    EXPECT_EQ(types_of(registry.rows()), expected);
    const Data::DatasetEntry* const mnist = registry.find("mnist");
    ASSERT_NE(mnist, nullptr);
    EXPECT_NE(mnist->make, nullptr);
}

/** @brief Tests that build_model over register_layers alone fails at the first conv row. */
TEST(Unit_VisionRegistration, ARegistryWithoutTheBlocksRefusesConv)
{
    NN::NodeRegistry registry;
    ASSERT_TRUE(SushiAI::Test::status_is_ok(NN::register_layers(registry)));

    SA_EXPECT_FAILURE(NN::build_model(registry, model_from(K_CONV_MODEL)),
                      StatusCode::InvalidDescription, "layer 0 'conv'", "the type 'conv'",
                      "does not have");
}

/** @brief Tests that every row of register_nodes has a minimal case here, in row order. */
TEST(Unit_VisionRegistration, EveryRowHasAMinimalCase)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, blocks_alone());
    const SushiRuntime::span<const NN::NodeEntry> nodes = registry.nodes();
    ASSERT_EQ(K_MINIMAL_CASES.size(), nodes.size());

    for (std::size_t index = 0; index < nodes.size(); ++index)
        EXPECT_EQ(K_MINIMAL_CASES[index].type, nodes[index].type);
}

/** @brief Tests that each row builds a one-port node of the expected shape from its case. */
TEST(Unit_VisionRegistration, EachRowBuildsFromAMinimalObject)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, blocks_alone());
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
TEST(Unit_VisionRegistration, EachRowRejectsAnUnknownKey)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, blocks_alone());
    for (const MinimalCase& minimal : K_MINIMAL_CASES)
    {
        const std::string context = "layer 0 '" + std::string(minimal.type) + "'";

        SA_EXPECT_FAILURE(make_minimal(registry, minimal, R"({"no_such_key": 1})"),
                          StatusCode::InvalidDescription, context, "'no_such_key'");
    }
}

/**
 * @brief Tests that the YOLO26n file parses to its 24 rows: the types in order, the sources
 *        of the four joins and of the head, the merged defaults and the one output.
 */
TEST(Unit_VisionRegistration, TheYolo26FileHoldsItsTwentyFourRows)
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
TEST(Unit_VisionRegistration, TheYolo26FileNamesEachMissingBlock)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));

    SA_EXPECT_FAILURE(NN::build_model(registry, spec), StatusCode::InvalidDescription,
                      "layer 10 'c2psa'", "the type 'c2psa'", "does not have", "'c3k2'",
                      "'upsample'");

    // A silu row keeps [20, 20, 256], which is what C2PSA(256, 256) keeps.
    stand_in(spec, K_ATTENTION_ROW, "silu", "{}");
    SA_EXPECT_FAILURE(NN::build_model(registry, spec), StatusCode::InvalidDescription,
                      "layer 22 'c3k2'", "'inner'", "'bottleneck_psa'", "bottleneck, c3k");

    stand_in(spec, K_ATTENTION_INNER_ROW, "conv",
             R"({"filters": 256, "bn_epsilon": 0.001, "bn_momentum": 0.03})");
    SA_EXPECT_FAILURE(NN::build_model(registry, spec), StatusCode::InvalidDescription,
                      "layer 23 'detect'", "the type 'detect'", "does not have");
}

/**
 * @brief Tests that rows 2 and 4 of the YOLO26n file each trace one ADD, the shortcut of the
 *        one Bottleneck a C3k2 of the pinned Ultralytics commit builds with shortcut=True.
 */
TEST(Unit_VisionRegistration, Yolo26RowsTwoAndFourAddTheirBottlenecksInput)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    const std::size_t rows_through_four = 5;
    keep_first_rows(spec, rows_through_four);

    EXPECT_TRUE(spec.layers[2].attributes.at("shortcut", "row 2").as_bool("shortcut"));
    EXPECT_TRUE(spec.layers[4].attributes.at("shortcut", "row 4").as_bool("shortcut"));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary traced,
                            trace_summary(model, spec.input_shape, "model"));

    EXPECT_EQ(std::count(traced.ops.begin(), traced.ops.end(), Ops::K_ADD.id), 2);
}

/**
 * @brief Tests that rows 0 to 9 of the YOLO26n file build and yield exactly the records
 *        MANIFEST.json lists under model.0 to model.9: names, shapes and order.
 */
TEST(Unit_VisionRegistration, Yolo26RowsZeroToNineMatchTheManifest)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    keep_first_rows(spec, K_ATTENTION_ROW);
    const Config::Value manifest = Config::parse_json_file(manifest_path(root));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
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
TEST(Unit_VisionRegistration, Yolo26RowsBuildToTheReferenceShapes)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    stand_in(spec, K_ATTENTION_ROW, "silu", "{}");
    keep_first_rows(spec, K_ATTENTION_INNER_ROW);

    SA_ASSERT_OK_AND_ASSIGN(const ModelGraph model, NN::build_model(registry, spec));

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
TEST(Unit_VisionRegistration, Yolo26NeckTracesToTheReferenceLevels)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    stand_in(spec, K_ATTENTION_ROW, "silu", "{}");
    keep_first_rows(spec, K_ATTENTION_INNER_ROW);
    spec.outputs = {named_output(16, "level0"), named_output(19, "level1"),
                    named_output(21, "joined")};
    const Config::Value manifest = Config::parse_json_file(manifest_path(root));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
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
        const Shape expected{K_TRACE_BATCH, height, width, channels};

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
