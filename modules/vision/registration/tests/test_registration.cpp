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
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/vision/datasets/reader_section.hpp>
#include <SushiAI/vision/registration.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>
#include <SushiAI/layer_traces/trace_summary.hpp>

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

    /**
     * @brief Lists a minimal object for every single-port row of register_nodes, in the rows'
     *        order; detect, the one row with several ports, has K_DETECT_CASE.
     */
    const std::vector<MinimalCase> K_MINIMAL_CASES = {
        {"conv", R"({"filters": 5})", {Shape{8, 8, 2}}, Shape{8, 8, 5}},
        {"bottleneck", R"({"filters": 4, "hidden": 2})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"c3k", R"({"filters": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"c3k2", R"({"filters": 8, "hidden": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 8}},
        {"sppf", R"({"filters": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
        {"c2psa", R"({"filters": 4, "hidden": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 4}},
    };

    /** @brief Holds the minimal object of segment_head, the row after detect. */
    const MinimalCase K_SEGMENT_HEAD_CASE = {
        "segment_head",
        R"({"classes": 3, "scale": 2})",
        {Shape{8, 8, 4}},
        Shape{16, 16, 3},
    };

    /**
     * @brief Holds the minimal object of detect, which reads three levels and has no single
     *        output shape; its output is unused, and its ports are counted by a test.
     */
    const MinimalCase K_DETECT_CASE = {
        "detect",
        R"({"classes": 3})",
        {Shape{8, 8, 4}, Shape{4, 4, 4}, Shape{2, 2, 4}},
        Shape{},
    };

    /** @brief Holds the number of class logits of the YOLO26n detect row. */
    constexpr std::int64_t K_YOLO_CLASSES = 80;

    /** @brief Holds the number of box coordinates per position of the detect row. */
    constexpr std::int64_t K_BOX_COORDINATES = 4;

    /** @brief Holds the feature side, in positions, of each level of the detect row. */
    const std::vector<std::int64_t> K_DETECT_SIDES = {80, 40, 20};

    /** @brief Holds the state name prefix every manifest record carries and ModelState drops. */
    constexpr std::string_view K_MODEL_PREFIX = "model.";

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
        return root / "applications" / "examples" / "yolo26" / "model.json";
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
     * @brief Lists the weights.sushi records of rows @p first to @p last, in manifest order.
     * @return Each record as state_record() writes it; a name ending in a running statistic is a
     *         buffer.
     */
    std::vector<std::string> manifest_records(const Config::Value& manifest, std::size_t first,
                                              std::size_t last)
    {
        std::vector<std::string> records;
        const Config::Value& weights = manifest_file(manifest, "weights.sushi");
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

    /**
     * @brief Removes the "model." prefix from each manifest record, as ModelState names lack it.
     * @param records Records as state_record() writes them, each starting with its name.
     * @return The same records with the prefix removed where a record carries it.
     */
    std::vector<std::string> without_model_prefix(std::vector<std::string> records)
    {
        for (std::string& record : records)
        {
            if (record.rfind(K_MODEL_PREFIX, 0) == 0)
                record.erase(0, K_MODEL_PREFIX.size());
        }

        return records;
    }

    /**
     * @brief Lists the twelve ports of the detect row over a [2, 640, 640, 3] input, in port
     *        order, as trace_summary() writes them: "<copy>.level<l>.<kind> [extents]".
     */
    std::vector<std::string> detect_outputs()
    {
        const std::vector<std::pair<std::string_view, std::int64_t>> kinds = {
            {"boxes", K_BOX_COORDINATES},
            {"scores", K_YOLO_CLASSES},
        };

        // A copy publishes the boxes of every level, then the scores of every level.
        std::vector<std::string> outputs;
        for (const std::string_view copy : {std::string_view("one2many"),
                                            std::string_view("one2one")})
        {
            for (const std::pair<std::string_view, std::int64_t>& kind : kinds)
            {
                for (std::size_t level = 0; level < K_DETECT_SIDES.size(); ++level)
                {
                    const std::int64_t side = K_DETECT_SIDES[level];
                    const Shape shape{K_TRACE_BATCH, side, side, kind.second};

                    outputs.push_back(std::string(copy) + ".level" + std::to_string(level) +
                                      "." + std::string(kind.first) + " " + shape.to_string());
                }
            }
        }

        return outputs;
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

/**
 * @brief Tests that register_nodes adds conv, bottleneck, c3k, c3k2, sppf, c2psa, detect and
 *        segment_head, in that order.
 */
TEST(Unit_VisionRegistration, RegisterNodesAddsTheEightBlocks)
{
    const std::vector<std::string_view> expected = {
        "conv", "bottleneck", "c3k", "c3k2", "sppf", "c2psa", "detect", "segment_head",
    };

    NN::NodeRegistry registry;
    SA_EXPECT_OK(Vision::register_nodes(registry));

    EXPECT_EQ(types_of(registry.nodes()), expected);
    EXPECT_TRUE(registry.layers().empty());
}

/** @brief Tests that register_datasets adds "mnist" and "reader", each with a factory. */
TEST(Unit_VisionRegistration, RegisterDatasetsAddsMNISTAndReader)
{
    const std::vector<std::string_view> expected = {"mnist", "reader"};

    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Vision::register_datasets(registry));

    EXPECT_EQ(types_of(registry.rows()), expected);
    const Data::DatasetEntry* const mnist = registry.find("mnist");
    ASSERT_NE(mnist, nullptr);
    EXPECT_NE(mnist->make, nullptr);

    const Data::DatasetEntry* const reader = registry.find(Config::K_READER_DATA_TYPE);
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(reader->make, &Vision::Datasets::make_reader_dataset);
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

/**
 * @brief Tests that every row of register_nodes has a minimal case here, in row order, with
 *        detect and segment_head last.
 */
TEST(Unit_VisionRegistration, EveryRowHasAMinimalCase)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, blocks_alone());
    const SushiRuntime::span<const NN::NodeEntry> nodes = registry.nodes();
    ASSERT_EQ(K_MINIMAL_CASES.size() + 2, nodes.size());

    for (std::size_t index = 0; index < K_MINIMAL_CASES.size(); ++index)
        EXPECT_EQ(K_MINIMAL_CASES[index].type, nodes[index].type);
    EXPECT_EQ(K_DETECT_CASE.type, nodes[K_MINIMAL_CASES.size()].type);
    EXPECT_EQ(K_SEGMENT_HEAD_CASE.type, nodes[K_MINIMAL_CASES.size() + 1].type);
}

/** @brief Tests that segment_head builds one port at twice the extents, a channel per class. */
TEST(Unit_VisionRegistration, SegmentHeadBuildsOnePortAtTheUpsampledSize)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, blocks_alone());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        make_minimal(registry, K_SEGMENT_HEAD_CASE, K_SEGMENT_HEAD_CASE.attributes));

    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, K_SEGMENT_HEAD_CASE.output);
}

/**
 * @brief Tests that detect over three levels builds four ports per level: the boxes and the
 *        scores of each of the two copies.
 */
TEST(Unit_VisionRegistration, DetectBuildsFourPortsPerLevel)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, blocks_alone());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            make_minimal(registry, K_DETECT_CASE, K_DETECT_CASE.attributes));

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->ports().size(), 4u * K_DETECT_CASE.inputs.size());
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

    SA_EXPECT_FAILURE(make_minimal(registry, K_DETECT_CASE, R"({"no_such_key": 1})"),
                      StatusCode::InvalidDescription, "layer 0 'detect'", "'no_such_key'");
}

/**
 * @brief Tests that the YOLO26n file parses to its 24 rows: the types in order, the sources
 *        of the four joins and of the head, the merged defaults and the one output.
 */
TEST(Unit_VisionRegistration, TheYOLO26FileHoldsItsTwentyFourRows)
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
 * @brief Tests that the YOLO26n file builds whole over the composed registry, with the state
 *        of MANIFEST.json's 594 records and the twelve ports of the detect row published in
 *        port order.
 */
TEST(Unit_VisionRegistration, TheYOLO26FileBuildsWhole)
{
    SA_FIND_REPOSITORY_OR_SKIP(root);
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(yolo_model_path(root));
    const Config::Value manifest = Config::parse_json_file(manifest_path(root));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, ""));

    EXPECT_EQ(trace.outputs, detect_outputs());

    const std::vector<std::string> expected =
        without_model_prefix(manifest_records(manifest, 0, K_YOLO_ROWS - 1));
    ASSERT_EQ(expected.size(), 594u);
    EXPECT_EQ(trace.state, expected);
}

/**
 * @brief Tests that rows 2 and 4 of the YOLO26n file each trace one ADD, the shortcut of the
 *        one Bottleneck a C3k2 of the pinned Ultralytics commit builds with shortcut=True.
 */
TEST(Unit_VisionRegistration, YOLO26RowsTwoAndFourAddTheirBottlenecksInput)
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
TEST(Unit_VisionRegistration, YOLO26RowsZeroToNineMatchTheManifest)
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
TEST(Unit_VisionRegistration, YOLO26RowsBuildToTheReferenceShapes)
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
TEST(Unit_VisionRegistration, YOLO26NeckTracesToTheReferenceLevels)
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

/** @brief Tests that the vision domain adds its three objectives after the layers' rows. */
TEST(Unit_VisionRegistration, RegistersTheAnchorAndThePixelObjectives)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Train::register_objectives(registry));
    const std::size_t layers = registry.rows().size();
    SA_EXPECT_OK(Vision::register_objectives(registry));

    ASSERT_EQ(registry.rows().size(), layers + 3);
    EXPECT_EQ(registry.rows()[0].type, "cross_entropy");
    EXPECT_EQ(registry.rows()[layers].type, "anchor");
    EXPECT_EQ(registry.rows()[layers + 1].type, "pixel_cross_entropy");
    EXPECT_EQ(registry.rows()[layers + 2].type, "dice");

    SA_EXPECT_FAILURE(Vision::register_objectives(registry), StatusCode::InvalidDescription,
                      "'anchor'", "already registered");
}

/** @brief Tests that the registered row refuses a model file that has no task section. */
TEST(Unit_VisionRegistration, TheAnchorRowReadsTheTaskOfTheModel)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Vision::register_objectives(registry));
    const Config::Value empty = Config::parse_json("{}", "<test>");

    SA_EXPECT_FAILURE(Train::make_objective(registry, "anchor", Train::ObjectiveContext{},
                                            empty, "the objective block"),
                      StatusCode::InvalidDescription, "the objective block", "'task'");
}

/** @brief Tests that each pixel row builds its objective over one [N, H, W, C] output. */
TEST(Unit_VisionRegistration, ThePixelRowsBuildOverOneLogitMap)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Vision::register_objectives(registry));
    Train::ObjectiveContext given;
    given.batch = 2;
    given.input_sample = Shape{4, 4, 3};
    given.outputs.push_back(Train::OutputShape{"output", Shape{2, 4, 4, 3}});
    const Config::Value section = Config::parse_json(R"({"ignore_index": 255})", "<test>");

    for (const std::string_view type : {"pixel_cross_entropy", "dice"})
    {
        SCOPED_TRACE(std::string(type));
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> objective,
                                Train::make_objective(registry, type, given, section,
                                                      "the objective block"));

        ASSERT_NE(objective, nullptr);
        EXPECT_EQ(objective->name(), type);
    }
}

/** @brief Tests that the vision domain adds the "darknet" importer and refuses a repeat. */
TEST(Unit_VisionRegistration, RegistersTheDarknetImporter)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(Vision::register_importers(registry));

    ASSERT_EQ(registry.rows().size(), 1u);
    EXPECT_EQ(registry.rows()[0].format, "darknet");

    SA_EXPECT_FAILURE(Vision::register_importers(registry), StatusCode::InvalidDescription,
                      "'darknet'", "already registered");
}

/** @brief Tests that the registered row reaches the Darknet importer, which counts its files. */
TEST(Unit_VisionRegistration, TheDarknetRowRefusesACallWithoutFiles)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(Vision::register_importers(registry));

    SA_EXPECT_FAILURE(IO::import_model(registry, "darknet", {}),
                      StatusCode::InvalidDescription, "'darknet'", ".cfg", "0 files");
}

namespace
{
    /** @brief Provides an engine for the dataset rows, whose factories take one. */
    class Unit_VisionRegistrationData : public SushiAI::Test::EngineTest
    {
    };
} // namespace

/** @brief Tests that the "reader" row reaches the reader table through the registry. */
TEST_F(Unit_VisionRegistrationData, TheReaderRowBuildsADataSection)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Vision::register_datasets(registry));

    const Config::Value section = SushiAI::Test::attributes_from(
        R"({"reader": "no_such_reader", "task": "detection", "root": "sushiai_absent_root"})");

    SA_EXPECT_FAILURE(Data::make_dataset(registry, *engine, Config::K_READER_DATA_TYPE, section,
                                         "the data block"),
                      StatusCode::InvalidDescription, "the data block 'reader'",
                      "'no_such_reader'", "'coco_json'", "'yolo_txt'");
}
