/**************************************************************************/
/* test_model_writer.cpp                                                  */
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
 * @file test_model_writer.cpp
 * @brief Tests that write_model writes a model file parse_model reads back unchanged.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>

using namespace SushiAI;
using Config::Kind;
using Config::LayerSpec;
using Config::ModelSpec;
using Config::OutputSpec;
using Config::SourceSpec;
using Config::Value;
using Config::write_model;
using Test::model_from;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Holds a chain of three rows that states neither "from" nor "outputs". */
    constexpr std::string_view K_CHAIN_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [28, 28, 1]},
        "layers": [
            {"type": "conv2d", "filters": 8, "kernel": 3},
            {"type": "relu"},
            {"type": "linear", "units": 10}
        ]
    })";

    /**
     * @brief Holds a model with an attribute of every JSON kind, every form of a "from" entry,
     *        every form of an "outputs" entry and a "defaults" block.
     */
    constexpr std::string_view K_FULL_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [32, 32, 3]},
        "defaults": {
            "conv": {"bn_epsilon": 0.001, "bn_momentum": 0.03, "stride": 1},
            "leaky_relu": {"slope": 1e-3}
        },
        "layers": [
            {"type": "conv", "filters": 16, "kernel": 3, "stride": 2},
            {"type": "conv2d", "from": "input", "filters": 8, "kernel": 3, "stride": 1,
             "padding": 1, "dilation": 1, "use_bias": true},
            {"type": "leaky_relu"},
            {"type": "c3k2", "from": 0, "filters": 64, "hidden": 16, "repeats": 1,
             "inner": "bottleneck", "shortcut": false},
            {"type": "upsample", "scale": 2, "from": -1},
            {"type": "maxpool2d", "kernel": 2, "stride": 2, "padding": 0,
             "from": {"node": 3}},
            {"type": "concat", "from": [-1, 0, "input", {"node": 4, "port": "features"}]},
            {"type": "conv", "filters": 32, "bn_momentum": -0.5, "from": [6]},
            {"type": "detect", "from": [{"node": 3, "port": "boxes"}, 5, 7], "classes": 80,
             "anchors": [[0.024, 0.031], [0.055, 0.072], []],
             "suppression": {"criterion": "iou", "threshold": null, "limits": {"boxes": 300}},
             "label": "a \"quoted\" name\twith a tab"},
            {"type": "flatten"},
            {"type": "linear", "units": 10, "scale": 2.0, "offset": -7}
        ],
        "outputs": [
            {"node": 8},
            {"node": 8, "port": "boxes"},
            {"node": 10, "port": "logits", "name": "predictions"}
        ]
    })";

    /** @brief Holds a chain whose "task" carries a value of every JSON kind. */
    constexpr std::string_view K_TASK_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [28, 28, 1]},
        "layers": [
            {"type": "conv2d", "filters": 8, "kernel": 3},
            {"type": "relu"}
        ],
        "outputs": [{"node": 1, "port": "values", "name": "head_0"}],
        "task": {
            "type": "anchor_detection",
            "classes": 80,
            "anchors": [[0.024, 0.031], [0.055, 0.072]],
            "heads": [{"output": "head_0", "anchors": [0, 1], "scale_xy": 1.05}],
            "suppression": {"criterion": "iou", "limit": null, "agnostic": false}
        }
    })";

    /** @brief Reports whether @p left and @p right are the same double, bit for bit. */
    bool same_bits(double left, double right)
    {
        std::uint64_t left_bits = 0;
        std::uint64_t right_bits = 0;
        std::memcpy(&left_bits, &left, sizeof(left_bits));
        std::memcpy(&right_bits, &right, sizeof(right_bits));
        return left_bits == right_bits;
    }

    /**
     * @brief Expects @p found to equal @p wanted: kind, value, entry order and key order.
     *
     * A real is compared bit for bit: the tolerance is zero, absolute and relative.
     * @param where Names the value in a failure, as "layer 3 attributes.anchors[1]".
     */
    void expect_same_value(const Value& found, const Value& wanted, const std::string& where)
    {
        ASSERT_EQ(found.kind(), wanted.kind()) << where;

        switch (wanted.kind())
        {
            case Kind::Null:
                return;

            case Kind::Bool:
                EXPECT_EQ(require_value(found.as_bool(where)),
                          require_value(wanted.as_bool(where))) << where;
                return;

            case Kind::Int:
                EXPECT_EQ(require_value(found.as_int(where)),
                          require_value(wanted.as_int(where))) << where;
                return;

            case Kind::Real:
                EXPECT_TRUE(same_bits(require_value(found.as_real(where)),
                                      require_value(wanted.as_real(where))))
                    << where << " is " << require_value(found.as_real(where)) << ", wanted "
                    << require_value(wanted.as_real(where));
                return;

            case Kind::String:
                EXPECT_EQ(require_value(found.as_string(where)),
                          require_value(wanted.as_string(where))) << where;
                return;

            case Kind::Array:
            {
                const std::vector<Value>& found_entries = require_value(found.as_array(where));
                const std::vector<Value>& wanted_entries = require_value(wanted.as_array(where));
                ASSERT_EQ(found_entries.size(), wanted_entries.size()) << where;

                for (std::size_t i = 0; i < wanted_entries.size(); ++i)
                    expect_same_value(found_entries[i], wanted_entries[i],
                                      where + "[" + std::to_string(i) + "]");
                return;
            }

            case Kind::Object:
                ASSERT_EQ(found.keys(), wanted.keys()) << where;

                for (const std::string& key : wanted.keys())
                    expect_same_value(require_value(found.at(key, where)),
                                      require_value(wanted.at(key, where)),
                                      where + "." + key);
                return;
        }
    }

    /** @brief Expects every field parse_model fills to be equal in @p found and @p wanted. */
    void expect_same_model(const ModelSpec& found, const ModelSpec& wanted)
    {
        EXPECT_TRUE(found.input_shape == wanted.input_shape) << "the input shapes differ";
        EXPECT_EQ(found.input_dtype, wanted.input_dtype) << "the input element types differ";
        EXPECT_EQ(found.input_fit, wanted.input_fit) << "the input fits differ";

        ASSERT_EQ(found.layers.size(), wanted.layers.size());
        for (std::size_t i = 0; i < wanted.layers.size(); ++i)
        {
            const std::string where = "layer " + std::to_string(i);
            const LayerSpec& found_layer = found.layers[i];
            const LayerSpec& wanted_layer = wanted.layers[i];

            EXPECT_EQ(found_layer.type, wanted_layer.type) << where;
            expect_same_value(found_layer.attributes, wanted_layer.attributes,
                              where + " attributes");

            ASSERT_EQ(found_layer.from.size(), wanted_layer.from.size()) << where;
            for (std::size_t k = 0; k < wanted_layer.from.size(); ++k)
            {
                EXPECT_EQ(found_layer.from[k].node, wanted_layer.from[k].node)
                    << where << " source " << k;
                EXPECT_EQ(found_layer.from[k].port, wanted_layer.from[k].port)
                    << where << " source " << k;
            }
        }

        ASSERT_EQ(found.outputs.size(), wanted.outputs.size());
        for (std::size_t i = 0; i < wanted.outputs.size(); ++i)
        {
            EXPECT_EQ(found.outputs[i].node, wanted.outputs[i].node) << "output " << i;
            EXPECT_EQ(found.outputs[i].port, wanted.outputs[i].port) << "output " << i;
            EXPECT_EQ(found.outputs[i].name, wanted.outputs[i].name) << "output " << i;
        }

        expect_same_value(found.task, wanted.task, "task");
    }

    /** @brief Expects parse_model of write_model of @p model to equal @p model. */
    void expect_round_trip(const ModelSpec& model)
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_model(model));
        expect_same_model(model_from(text), model);
    }

    /** @brief Returns the root of the repository the program was built from. */
    std::filesystem::path repository_root()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT);
    }

    /** @brief Returns the model file of the example @p name under @p root. */
    std::filesystem::path example_model_path(const std::filesystem::path& root,
                                             std::string_view name)
    {
        return root / "applications" / "examples" / std::filesystem::path(name) / "model.json";
    }

} // namespace

/** @brief Tests that an int32 input is written as "dtype" and a float32 one is left out. */
TEST(Small_ModelWriter, WritesTheInputElementTypeOnlyWhenItIsInt32)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    SA_ASSERT_OK_AND_ASSIGN(const std::string floating, write_model(model));
    EXPECT_EQ(floating.find("dtype"), std::string::npos);

    model.input_dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(const std::string indices, write_model(model));
    EXPECT_NE(indices.find("\"dtype\": \"int32\""), std::string::npos) << indices;
    EXPECT_EQ(model_from(indices).input_dtype, DType::Int32);

    model.input_dtype = DType::Float64;
    SA_EXPECT_FAILURE(write_model(model), StatusCode::InvalidDescription, "float64");
}

/** @brief Tests that a stated fit is written as "fit" and read back, and an empty one left out. */
TEST(Small_ModelWriter, WritesTheInputFitOnlyWhenOneIsStated)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    SA_ASSERT_OK_AND_ASSIGN(const std::string unstated, write_model(model));
    EXPECT_EQ(unstated.find("\"fit\""), std::string::npos);

    model.input_fit = "resize";
    SA_ASSERT_OK_AND_ASSIGN(const std::string stated, write_model(model));
    EXPECT_NE(stated.find("\"fit\": \"resize\""), std::string::npos) << stated;
    EXPECT_EQ(model_from(stated).input_fit, "resize");
}

TEST(Small_ModelWriter, RoundTripsTheMNISTExample)
{
    const std::filesystem::path root = repository_root();

    expect_round_trip(require_value(Config::load_model(example_model_path(root, "mnist_cnn"))));
}

TEST(Small_ModelWriter, RoundTripsTheYOLO26Example)
{
    const std::filesystem::path root = repository_root();

    expect_round_trip(require_value(Config::load_model(example_model_path(root, "yolo26"))));
}

TEST(Small_ModelWriter, RoundTripsEveryAttributeKindAndEveryFormOfFromAndOutputs)
{
    expect_round_trip(model_from(K_FULL_MODEL));
}

TEST(Small_ModelWriter, WritesTheSameTextFromWhatItWrote)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::string first, write_model(model_from(K_FULL_MODEL)));
    SA_ASSERT_OK_AND_ASSIGN(const std::string second, write_model(model_from(first)));

    EXPECT_EQ(first, second);
}

TEST(Small_ModelWriter, LeavesOutTheFromAndTheOutputsTheParserFillsIn)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_model(model_from(K_CHAIN_MODEL)));

    EXPECT_EQ(text,
              "{\n"
              "    \"format_version\": 2,\n"
              "    \"input\": {\n"
              "        \"shape\": [28, 28, 1]\n"
              "    },\n"
              "    \"layers\": [\n"
              "        { \"type\": \"conv2d\", \"filters\": 8, \"kernel\": 3 },\n"
              "        { \"type\": \"relu\" },\n"
              "        { \"type\": \"linear\", \"units\": 10 }\n"
              "    ]\n"
              "}\n");
}

TEST(Small_ModelWriter, WritesTypeThenFromThenTheAttributesAndTheOutputsLast)
{
    const ModelSpec model = model_from(R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"units": 4, "type": "linear"},
            {"type": "relu", "from": "input"},
            {"axis": 0, "from": [0, {"node": 1, "port": "values"}], "type": "concat"}
        ],
        "outputs": [{"node": 2, "port": "joined", "name": "result"}, {"node": 0}]
    })");

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_model(model));
    EXPECT_EQ(text,
              "{\n"
              "    \"format_version\": 2,\n"
              "    \"input\": {\n"
              "        \"shape\": [8]\n"
              "    },\n"
              "    \"layers\": [\n"
              "        { \"type\": \"linear\", \"units\": 4 },\n"
              "        { \"type\": \"relu\", \"from\": \"input\" },\n"
              "        { \"type\": \"concat\", \"from\": [0, { \"node\": 1, \"port\": "
              "\"values\" }], \"axis\": 0 }\n"
              "    ],\n"
              "    \"outputs\": [\n"
              "        { \"node\": 2, \"port\": \"joined\", \"name\": \"result\" },\n"
              "        { \"node\": 0 }\n"
              "    ]\n"
              "}\n");
}

TEST(Small_ModelWriter, WritesADefaultIntoEveryRowAndNoDefaultsBlock)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_model(model_from(K_FULL_MODEL)));

    EXPECT_EQ(text.find("\"defaults\""), std::string::npos);
    EXPECT_NE(text.find("{ \"type\": \"conv\", \"filters\": 16, \"kernel\": 3, \"stride\": 2, "
                        "\"bn_epsilon\": 0.001, \"bn_momentum\": 0.03 }"),
              std::string::npos)
        << text;
    EXPECT_NE(text.find("{ \"type\": \"leaky_relu\", \"slope\": 0.001 }"), std::string::npos)
        << text;
}

TEST(Small_ModelWriter, RoundTripsAPortOfTheModelInputAtRowZero)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    model.layers[0].from = {SourceSpec{Config::K_MODEL_INPUT, "image"}};

    expect_round_trip(model);
}

TEST(Small_ModelWriter, AnEmptyOutputsListReadsBackAsTheLastRowPublishedWhole)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    model.outputs.clear();

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_model(model));
    EXPECT_EQ(text.find("\"outputs\""), std::string::npos);

    const ModelSpec read = model_from(text);
    ASSERT_EQ(read.outputs.size(), 1u);
    EXPECT_EQ(read.outputs[0].node, 2);
    EXPECT_TRUE(read.outputs[0].port.empty());
    EXPECT_TRUE(read.outputs[0].name.empty());
}

TEST(Small_ModelWriter, RefusesAModelWithoutLayers)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    model.layers.clear();
    model.outputs.clear();

    SA_EXPECT_FAILURE(write_model(model), StatusCode::InvalidDescription, "no layers");
}

TEST(Small_ModelWriter, RefusesAnInputShapeAModelFileCannotState)
{
    ModelSpec scalar = model_from(K_CHAIN_MODEL);
    scalar.input_shape = Shape{};
    SA_EXPECT_FAILURE(write_model(scalar), StatusCode::InvalidDescription,
                      "input shape", "no extents");

    ModelSpec vast = model_from(K_CHAIN_MODEL);
    vast.input_shape = Shape{std::numeric_limits<std::int64_t>::max(), 2};
    SA_EXPECT_FAILURE(write_model(vast), StatusCode::InvalidDescription,
                      "input shape", "extent 1");
}

TEST(Small_ModelWriter, RefusesAttributesThatAreNotAnObject)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    model.layers[1].attributes = Value{};

    SA_EXPECT_FAILURE(write_model(model), StatusCode::InvalidDescription,
                      "layer 1", "'relu'", "null");
}

TEST(Small_ModelWriter, RefusesAnAttributeNamedAsARowKey)
{
    ModelSpec typed = model_from(K_CHAIN_MODEL);
    typed.layers[1].attributes = Value::object({"type"}, {Value::string("gelu")});
    SA_EXPECT_FAILURE(write_model(typed), StatusCode::InvalidDescription,
                      "layer 1", "'type'");

    ModelSpec sourced = model_from(K_CHAIN_MODEL);
    sourced.layers[2].attributes = Value::object({"from"}, {Value::integer(0)});
    SA_EXPECT_FAILURE(write_model(sourced), StatusCode::InvalidDescription,
                      "layer 2", "'from'");
}

TEST(Small_ModelWriter, RefusesARowThatReadsNothingOrARowNotAboveIt)
{
    ModelSpec empty = model_from(K_CHAIN_MODEL);
    empty.layers[1].from.clear();
    SA_EXPECT_FAILURE(write_model(empty), StatusCode::InvalidDescription,
                      "layer 1", "reads nothing");

    ModelSpec own = model_from(K_CHAIN_MODEL);
    own.layers[1].from = {SourceSpec{1, ""}};
    SA_EXPECT_FAILURE(write_model(own), StatusCode::InvalidDescription,
                      "layer 1", "row 1", "not above");

    ModelSpec later = model_from(K_CHAIN_MODEL);
    later.layers[0].from = {SourceSpec{2, ""}};
    SA_EXPECT_FAILURE(write_model(later), StatusCode::InvalidDescription,
                      "layer 0", "row 2", "not above");

    ModelSpec negative = model_from(K_CHAIN_MODEL);
    negative.layers[2].from = {SourceSpec{0, ""}, SourceSpec{-2, ""}};
    SA_EXPECT_FAILURE(write_model(negative), StatusCode::InvalidDescription,
                      "layer 2", "row -2", "not above");
}

TEST(Small_ModelWriter, RefusesAPortOfTheModelInputBelowRowZero)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    model.layers[2].from = {SourceSpec{Config::K_MODEL_INPUT, "image"}};

    SA_EXPECT_FAILURE(write_model(model), StatusCode::InvalidDescription,
                      "layer 2", "'image'", "row 0");
}

TEST(Small_ModelWriter, RefusesAnOutputAModelFileCannotState)
{
    ModelSpec beyond = model_from(K_CHAIN_MODEL);
    beyond.outputs = {OutputSpec{3, "", ""}};
    SA_EXPECT_FAILURE(write_model(beyond), StatusCode::InvalidDescription,
                      "output 0", "row 3", "0 to 2");

    ModelSpec negative = model_from(K_CHAIN_MODEL);
    negative.outputs = {OutputSpec{2, "", ""}, OutputSpec{-1, "", ""}};
    SA_EXPECT_FAILURE(write_model(negative), StatusCode::InvalidDescription,
                      "output 1", "row -1");

    ModelSpec named = model_from(K_CHAIN_MODEL);
    named.outputs = {OutputSpec{2, "", "predictions"}};
    SA_EXPECT_FAILURE(write_model(named), StatusCode::InvalidDescription,
                      "output 0", "'predictions'", "without a port");
}

TEST(Small_ModelWriter, ForwardsTheRefusalOfAnAttributeJSONCannotSpell)
{
    ModelSpec model = model_from(K_CHAIN_MODEL);
    model.layers[2].attributes = Value::object(
        {"rate"}, {Value::real(std::numeric_limits<double>::infinity())});

    SA_EXPECT_FAILURE(write_model(model), StatusCode::InvalidDescription,
                      "key 'layers'", "entry 2", "key 'rate'", "infinite");
}

TEST(Small_ModelWriter, RoundTripsATaskOfEveryValueKind)
{
    const ModelSpec model = model_from(K_TASK_MODEL);
    ASSERT_EQ(model.task.kind(), Kind::Object);

    expect_round_trip(model);
}

TEST(Small_ModelWriter, WritesTheTaskAfterTheOutputsWithItsKeysInOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_model(model_from(K_TASK_MODEL)));

    const std::size_t outputs = text.find("\n    \"outputs\": ");
    const std::size_t task = text.find("\n    \"task\": ");
    ASSERT_NE(outputs, std::string::npos) << text;
    ASSERT_NE(task, std::string::npos) << text;
    EXPECT_LT(outputs, task) << text;

    EXPECT_EQ(model_from(text).task.keys(),
              (std::vector<std::string>{"type", "classes", "anchors", "heads", "suppression"}));
}

TEST(Small_ModelWriter, LeavesOutANullTaskAndReadsItBackNull)
{
    const ModelSpec model = model_from(K_CHAIN_MODEL);
    ASSERT_TRUE(model.task.is_null());

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_model(model));
    EXPECT_EQ(text.find("\"task\""), std::string::npos);
    EXPECT_TRUE(model_from(text).task.is_null());
}

TEST(Small_ModelWriter, RefusesATaskThatIsNotAnObjectWithAStringType)
{
    ModelSpec listed = model_from(K_CHAIN_MODEL);
    listed.task = Value::array({Value::string("anchor_detection")});
    SA_EXPECT_FAILURE(write_model(listed), StatusCode::InvalidDescription,
                      "task", "an array", "object");

    ModelSpec untyped = model_from(K_CHAIN_MODEL);
    untyped.task = Value::object({"classes"}, {Value::integer(80)});
    SA_EXPECT_FAILURE(write_model(untyped), StatusCode::InvalidDescription,
                      "task", "no 'type'");

    ModelSpec numbered = model_from(K_CHAIN_MODEL);
    numbered.task = Value::object({"type"}, {Value::integer(3)});
    SA_EXPECT_FAILURE(write_model(numbered), StatusCode::InvalidDescription,
                      "task", "'type'", "a whole number");

    ModelSpec unnamed = model_from(K_CHAIN_MODEL);
    unnamed.task = Value::object({"type"}, {Value::string("")});
    SA_EXPECT_FAILURE(write_model(unnamed), StatusCode::InvalidDescription,
                      "task", "'type'", "empty string");
}
