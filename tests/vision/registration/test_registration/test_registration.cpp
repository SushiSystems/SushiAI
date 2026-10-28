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
 * @brief Tests what the vision registration adds: its rows, the MNIST loader and the YOLO26n model
 *        file against the reference.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/error.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>
#include <SushiAI/layer_traces/trace_summary.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>
#include <SushiAI/vision/datasets/reader_section.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include "registration_test_support.hpp"

using namespace SushiAI;
using NN::ModelGraph;
using Vision::Datasets::MNISTLayout;
using Vision::Datasets::MNISTOptions;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns the array under @p key of @p value; a refusal fails the test run. */
    const std::vector<Config::Value>& array_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_array(key));
    }

    /** @brief Returns the string under @p key of @p value; a refusal fails the test run. */
    const std::string& string_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_string(key));
    }

    /** @brief Returns the whole number under @p key of @p value; a refusal fails the test run. */
    std::int64_t int_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_int(key));
    }

    /** @brief Returns the real under @p key of @p value; a refusal fails the test run. */
    double real_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_real(key));
    }

    /** @brief Returns the flag under @p key of @p value; a refusal fails the test run. */
    bool bool_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_bool(key));
    }

    using Test::K_TRACE_BATCH;
    using Test::composed;
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
        {"c2f", R"({"filters": 8, "hidden": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 8}},
        {"c3", R"({"filters": 8, "hidden": 4})", {Shape{8, 8, 4}}, Shape{8, 8, 8}},
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

    /** @brief Returns the root of the repository the program was built from. */
    std::filesystem::path repository_root()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT);
    }

    /** @brief Returns the path of the YOLO26n model file under @p root. */
    std::filesystem::path yolo_model_path(const std::filesystem::path& root)
    {
        return root / "applications" / "examples" / "yolo26" / "model.json";
    }

    /** @brief Returns the path of the YOLO26 reference manifest in the fixture folder. */
    std::filesystem::path manifest_path()
    {
        return std::filesystem::path(Test::fixture_directory()) / "yolo26" / "MANIFEST.json";
    }

    /** @brief Returns the entry of the manifest's "files" whose name is @p name. */
    const Config::Value& manifest_file(const Config::Value& manifest, std::string_view name)
    {
        for (const Config::Value& file : array_of(manifest, "files"))
        {
            if (string_of(file, "name") == name)
                return file;
        }

        SA_FATAL_IF(true, "MANIFEST.json lists no file named '{}'", name);
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
        for (const Config::Value& tensor : array_of(weights, "tensors"))
        {
            const std::string& name =
                string_of(tensor, "name");
            const std::size_t row = record_row(name);
            if (row < first || row > last)
                continue;

            std::vector<int64_t> extents;
            for (const Config::Value& extent : array_of(tensor, "shape"))
                extents.push_back(SaTest::require_value(extent.as_int("extent")));

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
 * @brief Tests that register_nodes adds conv, bottleneck, c3k, c3k2, c2f, c3, sppf, c2psa,
 *        detect and segment_head, in that order.
 */
TEST(Medium_VisionRegistration, RegisterNodesAddsTheTenBlocks)
{
    const std::vector<std::string_view> expected = {
        "conv", "bottleneck", "c3k", "c3k2", "c2f", "c3", "sppf", "c2psa", "detect",
        "segment_head",
    };

    NN::NodeRegistry registry;
    SA_EXPECT_OK(Vision::register_nodes(registry));

    EXPECT_EQ(types_of(registry.nodes()), expected);
    EXPECT_TRUE(registry.layers().empty());
}

/** @brief Tests that register_datasets adds "mnist" and "reader", each with a factory. */
TEST(Medium_VisionRegistration, RegisterDatasetsAddsMNISTAndReader)
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
TEST(Medium_VisionRegistration, ARegistryWithoutTheBlocksRefusesConv)
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
TEST(Medium_VisionRegistration, EveryRowHasAMinimalCase)
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
TEST(Medium_VisionRegistration, SegmentHeadBuildsOnePortAtTheUpsampledSize)
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
TEST(Medium_VisionRegistration, DetectBuildsFourPortsPerLevel)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, blocks_alone());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            make_minimal(registry, K_DETECT_CASE, K_DETECT_CASE.attributes));

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->ports().size(), 4u * K_DETECT_CASE.inputs.size());
}

/** @brief Tests that each row builds a one-port node of the expected shape from its case. */
TEST(Medium_VisionRegistration, EachRowBuildsFromAMinimalObject)
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
TEST(Medium_VisionRegistration, EachRowRejectsAnUnknownKey)
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
TEST(Medium_VisionRegistration, TheYOLO26FileHoldsItsTwentyFourRows)
{
    const std::filesystem::path root = repository_root();
    const Config::ModelSpec spec = SaTest::require_value(Config::load_model(yolo_model_path(root)));

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

    EXPECT_NEAR(real_of(spec.layers[0].attributes, "bn_epsilon"),
                0.001,
                K_TOLERANCE);
    EXPECT_NEAR(real_of(spec.layers[9].attributes, "bn_momentum"),
                0.03,
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
TEST(Medium_VisionRegistration, TheYOLO26FileBuildsWhole)
{
    const std::filesystem::path root = repository_root();
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = SaTest::require_value(Config::load_model(yolo_model_path(root)));
    const Config::Value manifest = SaTest::require_value(Config::parse_json_file(manifest_path()));

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
TEST(Medium_VisionRegistration, YOLO26RowsTwoAndFourAddTheirBottlenecksInput)
{
    const std::filesystem::path root = repository_root();
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = SaTest::require_value(Config::load_model(yolo_model_path(root)));
    const std::size_t rows_through_four = 5;
    keep_first_rows(spec, rows_through_four);

    EXPECT_TRUE(bool_of(spec.layers[2].attributes, "shortcut"));
    EXPECT_TRUE(bool_of(spec.layers[4].attributes, "shortcut"));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary traced,
                            trace_summary(model, spec.input_shape, "model"));

    EXPECT_EQ(std::count(traced.ops.begin(), traced.ops.end(), Ops::K_ADD.id), 2);
}

/**
 * @brief Tests that rows 0 to 9 of the YOLO26n file build and yield exactly the records
 *        MANIFEST.json lists under model.0 to model.9: names, shapes and order.
 */
TEST(Medium_VisionRegistration, YOLO26RowsZeroToNineMatchTheManifest)
{
    const std::filesystem::path root = repository_root();
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = SaTest::require_value(Config::load_model(yolo_model_path(root)));
    keep_first_rows(spec, K_ATTENTION_ROW);
    const Config::Value manifest = SaTest::require_value(Config::parse_json_file(manifest_path()));

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
TEST(Medium_VisionRegistration, YOLO26RowsBuildToTheReferenceShapes)
{
    const std::filesystem::path root = repository_root();
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = SaTest::require_value(Config::load_model(yolo_model_path(root)));
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
TEST(Medium_VisionRegistration, YOLO26NeckTracesToTheReferenceLevels)
{
    const std::filesystem::path root = repository_root();
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = SaTest::require_value(Config::load_model(yolo_model_path(root)));
    stand_in(spec, K_ATTENTION_ROW, "silu", "{}");
    keep_first_rows(spec, K_ATTENTION_INNER_ROW);
    spec.outputs = {named_output(16, "level0"), named_output(19, "level1"),
                    named_output(21, "joined")};
    const Config::Value manifest = SaTest::require_value(Config::parse_json_file(manifest_path()));

    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(const TraceSummary trace,
                            trace_summary(model, spec.input_shape, "model"));

    const std::vector<Config::Value>& levels =
        array_of(SaTest::require_value(manifest_file(manifest, "forward.sacp").at("context",
                                                                                  "forward.sacp")),
                 "levels");
    ASSERT_EQ(levels.size(), 3u);
    ASSERT_EQ(trace.outputs.size(), 3u);
    for (std::size_t level = 0; level < 2; ++level)
    {
        SCOPED_TRACE("level " + std::to_string(level));
        const std::int64_t height =
            int_of(levels[level], "height");
        const std::int64_t width =
            int_of(levels[level], "width");
        const std::int64_t channels =
            int_of(levels[level], "channels");
        const Shape expected{K_TRACE_BATCH, height, width, channels};

        EXPECT_EQ(trace.outputs[level],
                  "level" + std::to_string(level) + " " + expected.to_string());

        EXPECT_NEAR(real_of(levels[level], "stride") *
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

/** @brief Tests that the vision domain adds its four objectives after the layers' rows. */
TEST(Medium_VisionRegistration, RegistersTheDetectionAndThePixelObjectives)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Train::register_objectives(registry));
    const std::size_t layers = registry.rows().size();
    SA_EXPECT_OK(Vision::register_objectives(registry));

    ASSERT_EQ(registry.rows().size(), layers + 4);
    EXPECT_EQ(registry.rows()[0].type, "cross_entropy");
    EXPECT_EQ(registry.rows()[layers].type, "anchor");
    EXPECT_EQ(registry.rows()[layers + 1].type, "anchor_free");
    EXPECT_EQ(registry.rows()[layers + 2].type, "pixel_cross_entropy");
    EXPECT_EQ(registry.rows()[layers + 3].type, "dice");

    SA_EXPECT_FAILURE(Vision::register_objectives(registry), StatusCode::InvalidDescription,
                      "'anchor'", "already registered");
}

/** @brief Tests that the registered row refuses a model file that has no task section. */
TEST(Medium_VisionRegistration, TheAnchorRowReadsTheTaskOfTheModel)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Vision::register_objectives(registry));
    const Config::Value empty = SaTest::require_value(Config::parse_json("{}", "<test>"));

    SA_EXPECT_FAILURE(Train::make_objective(registry, "anchor", Train::ObjectiveContext{},
                                            empty, "the objective block"),
                      StatusCode::InvalidDescription, "the objective block", "'task'");
}

/** @brief Tests that each pixel row builds its objective over one [N, H, W, C] output. */
TEST(Medium_VisionRegistration, ThePixelRowsBuildOverOneLogitMap)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Vision::register_objectives(registry));
    Train::ObjectiveContext given;
    given.batch = 2;
    given.input_sample = Shape{4, 4, 3};
    given.outputs.push_back(Train::OutputShape{"output", Shape{2, 4, 4, 3}});
    const Config::Value section =
        SaTest::require_value(Config::parse_json(R"({"ignore_index": 255})", "<test>"));

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
TEST(Medium_VisionRegistration, RegistersTheDarknetImporter)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(Vision::register_importers(registry));

    ASSERT_EQ(registry.rows().size(), 1u);
    EXPECT_EQ(registry.rows()[0].format, "darknet");

    SA_EXPECT_FAILURE(Vision::register_importers(registry), StatusCode::InvalidDescription,
                      "'darknet'", "already registered");
}

/** @brief Tests that the registered row reaches the Darknet importer, which counts its files. */
TEST(Medium_VisionRegistration, TheDarknetRowRefusesACallWithoutFiles)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(Vision::register_importers(registry));

    SA_EXPECT_FAILURE(IO::import_model(registry, "darknet", {}),
                      StatusCode::InvalidDescription, "'darknet'", ".cfg", "0 files");
}

namespace
{
    /** @brief Provides an engine for the dataset rows, whose factories take one. */
    class Medium_VisionRegistrationData : public SushiAI::Test::EngineTest
    {
    };
} // namespace

/** @brief Tests that the "reader" row reaches the reader table through the registry. */
TEST_F(Medium_VisionRegistrationData, TheReaderRowBuildsADataSection)
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

namespace
{
    /** @brief Provides an engine and a dataset registry that holds the vision rows. */
    class Medium_VisionMnist : public SaTest::EngineTest
    {
        protected:
            /** @brief Adds the "mnist" row the vision registry holds. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                SA_EXPECT_OK(Vision::register_datasets(datasets));
            }

            /** @brief Holds the dataset rows the tests build from. */
            Data::DatasetRegistry datasets;
    };

} // namespace

/** @brief Tests that the dataset registry carries a row named "mnist". */
TEST_F(Medium_VisionMnist, TheDatasetRegistryListsMNIST)
{
    EXPECT_NE(datasets.find("mnist"), nullptr);

    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "coco", attributes_from("{}"), "data"),
        StatusCode::InvalidDescription, "'mnist'");
}

/** @brief Tests that "mnist" rejects a key it does not know before it looks for any file. */
TEST_F(Medium_VisionMnist, AnUnknownMNISTKeyIsRejectedByItsOwnFactory)
{
    SA_EXPECT_FAILURE(Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "batch_sizes": 16})"),
            "data"), StatusCode::InvalidDescription, "data", "'batch_sizes'", "'batch_size'");
}

/** @brief Tests that "mnist" fails, naming the directory and its files, when they are absent. */
TEST_F(Medium_VisionMnist, MNISTRefusesADirectoryWithoutTheFiles)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "mnist",
                           attributes_from(R"({"directory": "sushiai_absent_mnist"})"),
                           "data"),
        StatusCode::FileNotFound, "data", "sushiai_absent_mnist", "MNIST train split",
        "train-images-idx3-ubyte, labels", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte");
}

/** @brief Tests that the "test" split of "mnist" looks for the t10k file pair. */
TEST_F(Medium_VisionMnist, MNISTTestSplitLooksForTheT10kFiles)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "split": "test"})"),
            "data"),
        StatusCode::FileNotFound, "data", "sushiai_absent_mnist", "MNIST test split",
        "t10k-images-idx3-ubyte, labels");
}

/** @brief Tests that "mnist" throws when the required "directory" key is absent. */
TEST_F(Medium_VisionMnist, MNISTRequiresADirectory)
{
    SA_EXPECT_FAILURE(Data::make_dataset(datasets, *engine, "mnist",
                                 attributes_from(R"({"split": "test"})"), "data"),
        StatusCode::InvalidDescription, "data", "has no", "'directory'");
}

/** @brief Tests that "mnist" rejects a split other than "train" or "test". */
TEST_F(Medium_VisionMnist, MNISTRefusesASplitItDoesNotKnow)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "split": "validation"})"),
            "data"),
        StatusCode::InvalidDescription, "data", "'split'", "'validation'", "'train'", "'test'");
}

/** @brief Tests that "mnist" refuses a layout other than "flat" or "image" before any file. */
TEST_F(Medium_VisionMnist, AnMNISTLayoutOtherThanFlatOrImageIsRefused)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "layout": "tiled"})"),
            "data"),
        StatusCode::InvalidDescription, "data", "'layout'", "'tiled'", "'flat'", "'image'");
}

/** @brief Tests that the "image" layout yields [N, rows, cols, 1] batches from the IDX files. */
TEST_F(Medium_VisionMnist, AnImageLayoutYieldsRankFourBatches)
{
    MNISTOptions options;
    options.layout = MNISTLayout::Image;
    options.batch_size = 16;
    options.max_samples = 32;
    if (!Vision::Datasets::mnist_available(options))
        GTEST_SKIP() << "MNIST IDX files are absent from " << options.directory.string();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            Vision::Datasets::try_load_mnist(*engine, options));

    EXPECT_EQ(dataset->sample_shape(), (Shape{28, 28, 1}));
    EXPECT_EQ(dataset->sample_shape().num_elements(), 784);

    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets().front().role, Data::K_ROLE_LABELS);
    EXPECT_EQ(dataset->targets().front().sample_shape, Shape{});
    EXPECT_EQ(dataset->targets().front().dtype, DType::Int32);
    EXPECT_EQ(dataset->targets().front().classes, 10);

    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    const Data::BatchTarget* labels =
        Data::find_target(dataset_batch.targets, Data::K_ROLE_LABELS);
    ASSERT_NE(labels, nullptr);
    ASSERT_EQ(labels->values.rank, 1);
    EXPECT_EQ(labels->values.shape[0], 16);
    EXPECT_EQ(labels->values.dtype, DType::Int32);

    const SushiBLAS::Tensor& inputs = dataset_batch.inputs;
    ASSERT_EQ(inputs.rank, 4);
    EXPECT_EQ(inputs.shape[0], 16);
    EXPECT_EQ(inputs.shape[1], 28);
    EXPECT_EQ(inputs.shape[2], 28);
    EXPECT_EQ(inputs.shape[3], 1);
}

/** @brief Tests that an absent directory is unavailable, refused and named. */
TEST_F(Medium_VisionMnist, MNISTFallsBackToSyntheticWhenTheFilesAreAbsent)
{
    MNISTOptions options;
    options.directory = "this-directory-does-not-exist";

    EXPECT_FALSE(Vision::Datasets::mnist_available(options));
    SA_EXPECT_FAILURE(Vision::Datasets::try_load_mnist(*engine, options),
                      StatusCode::FileNotFound, "this-directory-does-not-exist");
    EXPECT_FALSE(Vision::Datasets::mnist_expected_paths(options).empty());
}

namespace
{
    class Medium_Yolo26Forward : public SaTest::EngineTest {};

    using SaTest::ReferenceFile;
    using SaTest::ReferenceTensor;
    using SaTest::near_reference;
    using Vision::Detect::Free::FeatureLevel;
    using Vision::Detect::Free::LevelPlacement;
    using Vision::Detect::PredictionShape;

    /** @brief Holds every published tensor of one forward pass, by its published name. */
    using ForwardOutputs = std::map<std::string, std::vector<float>, std::less<>>;

    /** @brief Names a host decode of `[B, A, 4]` distances about `[A, 2]` anchor points. */
    using BoxDecode = void (*)(SushiRuntime::span<const float>, SushiRuntime::span<const float>,
                               SushiRuntime::span<float>);

    /** @brief Names a load of weights.sushi into the state of one model. */
    using WeightLoad = Status (*)(NN::ModelState&);

    /** @brief Names what an observed row of the model file is to the model. */
    enum class RowUse : std::uint8_t
    {
        HeadLevel, /**< The row is one of the three feature maps the head reads. */
        Inner,     /**< The row is read by other rows only. */
    };

    /** @brief Pairs a row of the model file with the record of forward.sacp of its output. */
    struct ObservedRow
    {
        std::int64_t row;        /**< The index of the row in the model file. */
        std::string_view record; /**< The record name, which is also the published name. */
        RowUse use;              /**< What the row is to the model. */
    };

    /** @brief Holds the absolute bound on a feature map and on a raw head output. */
    constexpr float K_ACTIVATION_TOLERANCE = 1e-4f;

    /** @brief Holds the absolute bound on a decoded box in pixels: two float32 steps at 640. */
    constexpr float K_DECODE_TOLERANCE = 1.3e-4f;

    /** @brief Holds the absolute bound on a class probability. */
    constexpr float K_PROBABILITY_TOLERANCE = 1e-6f;

    /** @brief Holds the class count of the detect row of the model file. */
    constexpr std::int64_t K_CLASSES = 80;

    /** @brief Holds the coordinates of one box. */
    constexpr std::int64_t K_BOX_VALUES = 4;

    /** @brief Holds the values of one detection row: four corners, a score and a class. */
    constexpr std::size_t K_DETECTION_VALUES = 6;

    /** @brief Holds the column of a detection row that carries its score. */
    constexpr std::size_t K_SCORE_COLUMN = 4;

    /** @brief Holds the rule that turns a record name of weights.sushi into a state name. */
    constexpr IO::RenameRule K_WEIGHT_RULES[] = {{K_MODEL_PREFIX, std::string_view()}};

    /** @brief Lists the rows the forward pass publishes beside the head, as MANIFEST.json has. */
    constexpr ObservedRow K_OBSERVED_ROWS[] = {
        {9, "features.row9", RowUse::Inner},        {10, "features.row10", RowUse::Inner},
        {16, "features.level0", RowUse::HeadLevel}, {19, "features.level1", RowUse::HeadLevel},
        {21, "features.row21", RowUse::Inner},      {22, "features.level2", RowUse::HeadLevel},
    };

    /** @brief Holds the row of the model file that is the c2psa block. */
    constexpr std::int64_t K_C2PSA_ROW = 10;

    /** @brief Holds the levels MANIFEST.json records under "levels" for a 640 px input. */
    const std::vector<FeatureLevel> K_LEVELS{{80, 80, 8.0f}, {40, 40, 16.0f}, {20, 20, 32.0f}};

    /** @brief Holds the joined boxes and probabilities of one copy of the head. */
    struct DecodedBranch
    {
        std::vector<float> boxes;  /**< `[1, A, 4]` in pixels. */
        std::vector<float> scores; /**< `[1, A, C]` after the sigmoid. */
    };

    /** @brief Returns the record K_OBSERVED_ROWS pairs with @p row, or an empty name. */
    constexpr std::string_view record_of(std::int64_t row) noexcept
    {
        for (const ObservedRow& observed : K_OBSERVED_ROWS)
        {
            if (observed.row == row)
                return observed.record;
        }

        return std::string_view();
    }

    /** @brief Returns the published name of @p kind of @p branch at @p level. */
    std::string port_name(std::string_view branch, std::size_t level, std::string_view kind)
    {
        std::string name(branch);
        name += ".level" + std::to_string(level) + ".";
        name += kind;
        return name;
    }

    /** @brief Returns the logistic function of @p logit, computed in double. */
    float sigmoid(float logit)
    {
        return static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(logit))));
    }

    /** @brief Builds a registry from NN::register_layers, then Vision::register_nodes. */
    Result<NN::NodeRegistry> compose_registry()
    {
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY(Vision::register_nodes(registry));

        return Result<NN::NodeRegistry>(std::move(registry));
    }

    /** @brief Returns the path of weights.sushi in the fixture directory. */
    std::filesystem::path weights_path()
    {
        return std::filesystem::path(SaTest::fixture_directory()) / "yolo26" / "weights.sushi";
    }

    /** @brief Loads the YOLO26n model file as it is written. */
    Config::ModelSpec load_spec()
    {
        return SaTest::require_value(Config::load_model(yolo_model_path(repository_root())));
    }

    /**
     * @brief Builds a graph that holds row @p row of the model file alone, named by its index.
     * @param sample The per-sample shape the row reads.
     * @param published The name the row's single port is published under.
     * @return The graph, or the failure of the registry, of the row's factory or of the graph.
     */
    Result<NN::ModelGraph> build_row_alone(std::int64_t row, const Shape& sample,
                                           const std::string& published)
    {
        SA_TRY_ASSIGN(const NN::NodeRegistry registry, compose_registry());
        const Config::ModelSpec spec = load_spec();
        const Config::LayerSpec& layer = spec.layers.at(static_cast<std::size_t>(row));
        const std::string name = std::to_string(row);

        NN::ModelGraph model(sample);
        SA_TRY_ASSIGN(std::unique_ptr<NN::Node> node,
                      NN::make_node(registry, layer.type, NN::ShapeSpan(&sample, 1),
                                    layer.attributes, name));
        const NN::NodeRef input = model.input();
        SA_TRY_ASSIGN(const NN::NodeRef made,
                      model.add(name, std::move(node),
                                SushiRuntime::span<const NN::NodeRef>(&input, 1)));
        SA_TRY(model.output(published, made));

        return Result<NN::ModelGraph>(std::move(model));
    }

    /**
     * @brief Loads the records of weights.sushi that rename to an entry of @p state and skips
     *        the others.
     * @return FileNotFound for an unreadable file; Incompatible for a record that repeats or
     *         differs from its entry in dtype or shape, and for an entry left unsupplied.
     */
    Status import_matching_weights(NN::ModelState& state)
    {
        const std::filesystem::path path = weights_path();
        std::ifstream in(path, std::ios::binary);
        SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                   path.string());
        SA_TRY_ASSIGN(IO::CheckpointReader reader, IO::CheckpointReader::open(in));

        std::set<std::string> supplied;
        while (reader.read_count() < reader.count())
        {
            SA_TRY_ASSIGN(const IO::TensorRecord record, reader.next());
            const std::string target = IO::renamed(record.name, K_WEIGHT_RULES);
            const Result<SushiBLAS::Tensor*> found = state.tensor(target);
            if (!found.has_value())
            {
                in.seekg(static_cast<std::streamoff>(record.payload_bytes()), std::ios::cur);
                continue;
            }

            SushiBLAS::Tensor* const tensor = found.value();
            SA_FAIL_IF(record.dtype != tensor->dtype || record.shape != shape_of(*tensor),
                       StatusCode::Incompatible, "'{}' holds {} {}, but the entry '{}' is {} {}",
                       record.name, dtype_name(record.dtype), record.shape.to_string(), target,
                       dtype_name(tensor->dtype), shape_of(*tensor).to_string());
            SA_FAIL_IF(!supplied.insert(target).second, StatusCode::Incompatible,
                       "'{}' supplies '{}' twice", path.string(), target);
            SA_TRY(reader.read_payload(record, tensor->data()));
        }

        for (const NN::StateEntry& entry : state.entries())
        {
            SA_FAIL_IF(supplied.count(entry.name) == 0, StatusCode::Incompatible,
                       "'{}' does not supply '{}'", path.string(), entry.name);
        }

        return Status();
    }

    /**
     * @brief Runs one evaluation step of @p model over @p values with the weights @p load reads.
     * @param batch_shape The extents of the input, batch first.
     * @param values The float32 elements of a @p batch_shape tensor.
     * @return Every published tensor by name, or the first failure of the trace or the run.
     */
    Result<ForwardOutputs> run_model(SushiBLAS::Engine& engine, NN::ModelGraph& model,
                                     const Shape& batch_shape, const std::vector<float>& values,
                                     WeightLoad load)
    {
        SA_TRY_ASSIGN(const Train::TracedStep traced,
                      Train::build_evaluation_graph(model, batch_shape));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        SA_TRY(load(state));

        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                      Train::EvaluationStep::make(engine, traced, state, pool));

        SA_TRY_ASSIGN(SushiBLAS::Tensor input, allocate_tensor(engine, batch_shape));
        SA_FAIL_IF(static_cast<std::size_t>(input.num_elements) != values.size(),
                   StatusCode::Incompatible, "the input holds {} values, but {} needs {}",
                   values.size(), batch_shape.to_string(), input.num_elements);
        std::memcpy(input.data_as<float>(), values.data(), values.size() * sizeof(float));
        SA_TRY(step->run(input));

        ForwardOutputs outputs;
        for (const NN::ModelOutputs::Entry& published : traced.outputs.all())
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, step->output(published.first));
            const float* const values = tensor->data_as<float>();
            outputs.emplace(published.first,
                            std::vector<float>(values, values + tensor->num_elements));
        }

        return Result<ForwardOutputs>(std::move(outputs));
    }

    /**
     * @brief Checks the output published as @p name against the record of the same name.
     * @return Success, or a failure naming what is missing or the first and the worst element.
     */
    ::testing::AssertionResult matches_reference(const ForwardOutputs& ours, ReferenceFile& file,
                                                 const std::string& name, float tolerance)
    {
        const auto found = ours.find(name);
        if (found == ours.end())
            return ::testing::AssertionFailure() << "the model publishes no '" << name << "'";

        const ReferenceTensor reference = file.read(name);
        if (!reference.found)
            return ::testing::AssertionFailure() << reference.reason;

        return near_reference(found->second, reference.floats(), tolerance);
    }

    /** @brief Expects every row of K_OBSERVED_ROWS that is a @p use to match its record. */
    void expect_rows_match(const ForwardOutputs& ours, ReferenceFile& file, RowUse use)
    {
        for (const ObservedRow& observed : K_OBSERVED_ROWS)
        {
            if (observed.use != use)
                continue;

            const std::string name(observed.record);
            SCOPED_TRACE(name);
            EXPECT_TRUE(matches_reference(ours, file, name, K_ACTIVATION_TOLERANCE));
        }
    }

    /**
     * @brief Joins the three levels of @p kind of @p branch into `[1, A, channels]`.
     * @param joined Receives the rows, levels in order; resized here.
     * @return Success, or a failure naming the port that is missing or has another size.
     */
    ::testing::AssertionResult join_levels(const ForwardOutputs& ours, std::string_view branch,
                                           std::string_view kind, std::int64_t channels,
                                           std::vector<float>& joined)
    {
        const std::int64_t anchors =
            SaTest::require_value(Vision::Detect::Free::count_anchors(K_LEVELS));
        joined.assign(static_cast<std::size_t>(anchors * channels), 0.0f);

        std::int64_t offset = 0;
        for (std::size_t level = 0; level < K_LEVELS.size(); ++level)
        {
            const std::string name = port_name(branch, level, kind);
            const auto found = ours.find(name);
            if (found == ours.end())
                return ::testing::AssertionFailure() << "the model publishes no '" << name << "'";

            const std::int64_t cells = K_LEVELS[level].height * K_LEVELS[level].width;
            const std::size_t expected = static_cast<std::size_t>(cells * channels);
            if (found->second.size() != expected)
                return ::testing::AssertionFailure() << "'" << name << "' holds "
                                                     << found->second.size() << " values, not "
                                                     << expected;

            SA_EXPECT_OK(Vision::Detect::Free::join_level(found->second,
                                                          LevelPlacement{1, anchors, channels,
                                                                         offset, cells},
                                                          joined));
            offset += cells;
        }

        return ::testing::AssertionSuccess();
    }

    /**
     * @brief Decodes the raw outputs of @p branch on the host: boxes in pixels through
     *        @p decode, class probabilities through the sigmoid.
     * @return Success, or the failure of join_levels().
     */
    ::testing::AssertionResult decode_branch(const ForwardOutputs& ours, std::string_view branch,
                                             BoxDecode decode, DecodedBranch& decoded)
    {
        const ::testing::AssertionResult boxes =
            join_levels(ours, branch, "boxes", K_BOX_VALUES, decoded.boxes);
        if (!boxes)
            return boxes;

        const ::testing::AssertionResult scores =
            join_levels(ours, branch, "scores", K_CLASSES, decoded.scores);
        if (!scores)
            return scores;

        const std::size_t anchors = static_cast<std::size_t>(
            SaTest::require_value(Vision::Detect::Free::count_anchors(K_LEVELS)));
        std::vector<float> points(anchors * Vision::Detect::Free::K_POINT_VALUES);
        std::vector<float> strides(anchors);
        SA_EXPECT_OK(Vision::Detect::Free::make_anchors(K_LEVELS, points, strides));

        decode(decoded.boxes, points, decoded.boxes);
        Vision::Detect::Free::scale_boxes(decoded.boxes, strides, decoded.boxes);
        for (float& score : decoded.scores)
            score = sigmoid(score);

        return ::testing::AssertionSuccess();
    }

    /** @brief Returns column K_SCORE_COLUMN of `[k, 6]` detection rows. */
    std::vector<float> score_column(const std::vector<float>& detections)
    {
        std::vector<float> scores;
        scores.reserve(detections.size() / K_DETECTION_VALUES);
        for (std::size_t at = 0; at + K_DETECTION_VALUES <= detections.size();
             at += K_DETECTION_VALUES)
            scores.push_back(detections[at + K_SCORE_COLUMN]);

        return scores;
    }

    /** @brief Holds the extents of "image" in input.sacp, batch first. */
    const Shape K_INPUT_SHAPE{1, 640, 640, 3};

    /** @brief Loads the YOLO26n model file and publishes K_OBSERVED_ROWS beside the head. */
    Config::ModelSpec load_spec_with_features()
    {
        Config::ModelSpec spec = load_spec();
        for (const ObservedRow& observed : K_OBSERVED_ROWS)
        {
            Config::OutputSpec output;
            output.node = observed.row;
            output.port = std::string(NN::K_SINGLE_PORT_NAME);
            output.name = std::string(observed.record);
            spec.outputs.push_back(std::move(output));
        }

        return spec;
    }

    /** @brief Loads weights.sushi into @p state, which every record must match exactly once. */
    Status import_every_weight(NN::ModelState& state)
    {
        return IO::import_weights(weights_path(), state, K_WEIGHT_RULES);
    }

    /**
     * @brief Runs one evaluation step of the whole model over @p image with every weight.
     * @param image The float32 elements of a K_INPUT_SHAPE tensor.
     * @return Every published tensor by name, or the first failure of the build or the run.
     */
    Result<ForwardOutputs> run_forward(SushiBLAS::Engine& engine, const std::vector<float>& image)
    {
        SA_TRY_ASSIGN(const NN::NodeRegistry registry, compose_registry());
        SA_TRY_ASSIGN(NN::ModelGraph model,
                      NN::build_model(registry, load_spec_with_features()));

        return run_model(engine, model, K_INPUT_SHAPE, image, &import_every_weight);
    }

    /**
     * @brief Returns the outputs of the one forward pass the cases of this file share.
     * @return The outputs of the first successful run_forward(), or the failure of this call.
     */
    Result<const ForwardOutputs*> shared_forward(SushiBLAS::Engine& engine,
                                                 const std::vector<float>& image)
    {
        static ForwardOutputs outputs;
        if (outputs.empty())
        {
            SA_TRY_ASSIGN(outputs, run_forward(engine, image));
        }

        return Result<const ForwardOutputs*>(&outputs);
    }

} // namespace

/**
 * @brief Declares @p reference as forward.sacp and @p ours as the shared forward outputs, or
 *        skips the test naming the reference file that is missing.
 * @pre Used as statements in a TEST_F of Medium_Yolo26Forward.
 */
#define SA_RUN_FORWARD_OR_SKIP(ours, reference)                                        \
    SA_OPEN_REFERENCE_OR_SKIP(reference, "yolo26/forward.sacp");                       \
    SA_OPEN_REFERENCE_OR_SKIP(reference##_weights, "yolo26/weights.sushi");            \
    SA_OPEN_REFERENCE_OR_SKIP(reference##_input, "yolo26/input.sacp");                 \
    SA_READ_REFERENCE_OR_FAIL(reference##_image, reference##_input, "image");          \
    SA_ASSERT_OK_AND_ASSIGN(const ForwardOutputs* const ours,                          \
                            shared_forward(*engine, reference##_image.floats()))

/** @brief Tests that rows 16, 19 and 22 compute the reference's three feature maps. */
TEST_F(Medium_Yolo26Forward, TheFeatureRowsMatchTheReferenceLevels)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    expect_rows_match(*ours, forward, RowUse::HeadLevel);
}

/**
 * @brief Tests that rows 9, 10 and 21 of the whole-model pass compute the reference's records
 *        of them, which pins the input and the output of c2psa and the concat row 22 reads.
 */
TEST_F(Medium_Yolo26Forward, TheInnerRowsMatchTheReferenceRows)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    expect_rows_match(*ours, forward, RowUse::Inner);
}

/**
 * @brief Tests that row 10 built alone, with its own weights and the reference's row 9 as its
 *        input, computes the reference's row 10, which pins c2psa apart from the rows above it.
 */
TEST_F(Medium_Yolo26Forward, C2PSAAloneMapsTheReferenceRow9ToRow10)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
    SA_OPEN_REFERENCE_OR_SKIP(weights, "yolo26/weights.sushi");
    SA_READ_REFERENCE_OR_FAIL(input, forward, record_of(K_C2PSA_ROW - 1));
    ASSERT_EQ(input.shape.rank(), 4);

    const std::string produced(record_of(K_C2PSA_ROW));
    const Shape sample{input.shape[1], input.shape[2], input.shape[3]};
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph block, build_row_alone(K_C2PSA_ROW, sample, produced));
    SA_ASSERT_OK_AND_ASSIGN(const ForwardOutputs ours,
                            run_model(*engine, block, input.shape, input.floats(),
                                      &import_matching_weights));

    EXPECT_TRUE(matches_reference(ours, forward, produced, K_ACTIVATION_TOLERANCE));
}

/** @brief Tests that the twelve ports of the head compute the reference's raw outputs. */
TEST_F(Medium_Yolo26Forward, TheTwelvePortsMatchTheReferenceRawOutputs)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    for (const std::string_view branch : {std::string_view("one2many"),
                                          std::string_view("one2one")})
    {
        for (std::size_t level = 0; level < K_LEVELS.size(); ++level)
        {
            for (const std::string_view kind : {std::string_view("boxes"),
                                                std::string_view("scores")})
            {
                const std::string name = port_name(branch, level, kind);
                SCOPED_TRACE(name);
                EXPECT_TRUE(matches_reference(*ours, forward, name, K_ACTIVATION_TOLERANCE));
            }
        }
    }
}

/**
 * @brief Tests that the host decode of our raw outputs gives the reference's decoded boxes and
 *        probabilities: one-to-many as centre and size, one-to-one as corners.
 */
TEST_F(Medium_Yolo26Forward, TheHostDecodeMatchesTheReferenceDecodes)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    const std::pair<std::string_view, BoxDecode> branches[] = {
        {"one2many", &Vision::Detect::Free::decode_xywh},
        {"one2one", &Vision::Detect::Free::decode_xyxy},
    };
    for (const std::pair<std::string_view, BoxDecode>& branch : branches)
    {
        SCOPED_TRACE(std::string(branch.first));
        const std::string prefix = std::string(branch.first) + ".decoded.";
        SA_READ_REFERENCE_OR_FAIL(boxes, forward, prefix + "boxes");
        SA_READ_REFERENCE_OR_FAIL(scores, forward, prefix + "scores");

        DecodedBranch decoded;
        ASSERT_TRUE(decode_branch(*ours, branch.first, branch.second, decoded));

        EXPECT_TRUE(near_reference(decoded.boxes, boxes.floats(), K_DECODE_TOLERANCE));
        EXPECT_TRUE(near_reference(decoded.scores, scores.floats(), K_PROBABILITY_TOLERANCE));
    }
}

/**
 * @brief Tests that the top-k selection over our one-to-one decode gives the scores of the
 *        reference detections, best first.
 */
TEST_F(Medium_Yolo26Forward, TheTopScoresMatchTheReferenceDetections)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);
    SA_READ_REFERENCE_OR_FAIL(reference, forward, "one2one.detections");

    DecodedBranch decoded;
    ASSERT_TRUE(decode_branch(*ours, "one2one", &Vision::Detect::Free::decode_xyxy, decoded));

    const std::int64_t anchors =
        SaTest::require_value(Vision::Detect::Free::count_anchors(K_LEVELS));
    const PredictionShape shape{1, anchors, K_CLASSES};
    const std::int64_t rows = SaTest::require_value(
        Vision::Detect::count_top_detections(shape, Vision::Detect::K_MAX_DETECTIONS));
    std::vector<float> detections(static_cast<std::size_t>(rows) * K_DETECTION_VALUES);
    SA_EXPECT_OK(Vision::Detect::select_top_detections(decoded.boxes, decoded.scores, shape,
                                                       Vision::Detect::K_MAX_DETECTIONS,
                                                       detections));

    EXPECT_TRUE(near_reference(score_column(detections), score_column(reference.floats()),
                               K_PROBABILITY_TOLERANCE));
}
