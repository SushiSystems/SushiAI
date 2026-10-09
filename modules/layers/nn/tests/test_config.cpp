/**************************************************************************/
/* test_config.cpp                                                        */
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
 * @file test_config.cpp
 * @brief Tests how the config layer parses its JSON documents and which ones it refuses.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <common/test_status.hpp>
#include <common/test_throws.hpp>
#include <common/test_utils.hpp>

using SushiAI::Config::Kind;
using SushiAI::Config::Value;
using SushiAI::Config::parse_json;

namespace
{
    /** @brief A span over a literal allowed-key list, for reject_unknown_keys. */
    template<std::size_t N>
    SushiRuntime::span<const std::string_view> allowed(const std::string_view (&keys)[N])
    {
        return SushiRuntime::span<const std::string_view>(keys, N);
    }

    using SushiAI::Test::make_layer_registry;
}

TEST(Unit_Config, ParsesEveryScalarKind)
{
    const Value root = parse_json(
        R"({"a": 1, "b": 2.5, "c": "text", "d": true, "e": null, "f": [1, 2]})", "<test>");

    ASSERT_EQ(root.kind(), Kind::Object);
    EXPECT_EQ(root.at("a", "<test>").as_int("a"), 1);
    EXPECT_DOUBLE_EQ(root.at("b", "<test>").as_real("b"), 2.5);
    EXPECT_EQ(root.at("c", "<test>").as_string("c"), "text");
    EXPECT_TRUE(root.at("d", "<test>").as_bool("d"));
    EXPECT_TRUE(root.at("e", "<test>").is_null());
    EXPECT_EQ(root.at("f", "<test>").as_array("f").size(), 2u);
}

TEST(Unit_Config, KeepsTheKeyOrderTheFileSpelled)
{
    // Key order follows the file; see tests/functional/unit/README.md.
    const Value root = parse_json(R"({"zebra": 1, "apple": 2, "mango": 3})", "<test>");
    EXPECT_EQ(root.keys(), (std::vector<std::string>{"zebra", "apple", "mango"}));
}

TEST(Unit_Config, AnIntegerReadsAsARealButARealIsNotAWholeNumber)
{
    // An integer widens to a real, never the reverse; see tests/functional/unit/README.md.
    const Value root = parse_json(R"({"lr": 3, "units": 2.5})", "<test>");

    EXPECT_DOUBLE_EQ(root.at("lr", "<test>").as_real("lr"), 3.0);
    EXPECT_EQ(root.at("lr", "<test>").as_int("lr"), 3);
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("'units'"),
                    "'units'", "whole number");
}

TEST(Unit_Config, AWrongKindNamesBothTheContextAndTheKindFound)
{
    const Value root = parse_json(R"({"units": "many"})", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("layer 0 'units'"),
                    "layer 0 'units'", "a string");
}

TEST(Unit_Config, AMissingKeyNamesTheKeyAndListsWhatIsPresent)
{
    const Value root = parse_json(R"({"alpha": 1, "beta": 2})", "model.json");
    SA_EXPECT_ERROR((void)root.at("gamma", "model.json"),
                    "model.json", "'gamma'", "'alpha'", "'beta'");
}

TEST(Unit_Config, ReadingAKeyOffANonObjectSaysWhatItActuallyIs)
{
    const Value root = parse_json(R"([1, 2, 3])", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "the model"), "the model", "an array");
}

TEST(Unit_Config, AnUnknownKeyIsAnErrorThatListsWhatIsAllowed)
{
    const Value root = parse_json(R"({"type": "linear", "unitss": 256})", "model.json");
    const std::string_view keys[] = {"type", "units"};

    SA_EXPECT_ERROR(root.reject_unknown_keys("layer 0 'linear'", allowed(keys)),
                    "layer 0 'linear'", "'unitss'", "'units'", "'type'");
}

TEST(Unit_Config, AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything)
{
    const Value with_keys = parse_json(R"({"type": "linear", "units": 256})", "<test>");
    const std::string_view keys[] = {"type", "units"};
    EXPECT_NO_THROW(with_keys.reject_unknown_keys("layer 0", allowed(keys)));

    // An empty allowed set refuses every key; see tests/functional/unit/README.md.
    const std::string_view none[] = {"type"};
    const Value stray = parse_json(R"({"type": "gelu", "units": 8})", "<test>");
    SA_EXPECT_ERROR(stray.reject_unknown_keys("layer 1 'gelu'", allowed(none)),
                    "layer 1 'gelu'", "'units'");
}

TEST(Unit_Config, WithoutRemovesTheDiscriminatorAndLeavesTheRest)
{
    const Value layer = parse_json(R"({"type": "linear", "units": 256, "use_bias": false})",
                                   "<test>");
    const Value attributes = layer.without("type");

    EXPECT_FALSE(attributes.has("type"));
    EXPECT_TRUE(attributes.has("units"));
    EXPECT_EQ(attributes.keys(), (std::vector<std::string>{"units", "use_bias"}));

    // Removing an absent key is not an error; see tests/functional/unit/README.md.
    EXPECT_EQ(attributes.without("type").keys().size(), 2u);
}

TEST(Unit_Config, AtOrFallsBackWithoutThrowing)
{
    const Value root = parse_json(R"({"units": 256})", "<test>");
    const Value fallback = Value::boolean(true);

    EXPECT_EQ(root.at_or("units", fallback).as_int("units"), 256);
    EXPECT_TRUE(root.at_or("use_bias", fallback).as_bool("use_bias"));

    // A non-object falls back instead of throwing; see tests/functional/unit/README.md.
    const Value not_an_object = Value::integer(1);
    EXPECT_TRUE(not_an_object.at_or("use_bias", fallback).as_bool("use_bias"));
}

TEST(Unit_Config, MalformedJsonNamesTheOriginAndTheComplaint)
{
    SA_EXPECT_ERROR((void)parse_json("{ not json", "train.json"),
                    "train.json", "not valid JSON");
}

TEST(Unit_Config, AnObjectBuiltFromMismatchedVectorsIsRefused)
{
    std::vector<std::string> keys{"a", "b"};
    std::vector<Value> values{Value::integer(1)};
    SA_EXPECT_ERROR((void)Value::object(std::move(keys), std::move(values)), "2 keys", "1 value");
}

namespace
{
    /** @brief Parses @p text as a model document called "model.json". */
    SushiAI::Config::ModelSpec model_from(std::string_view text)
    {
        return SushiAI::Config::parse_model(parse_json(text, "model.json"), "model.json");
    }

    constexpr std::string_view K_GOOD_MODEL = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 8},
                   {"type": "gelu"},
                   {"type": "linear", "units": 4}]
    })";
}

TEST(Unit_Config, ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes)
{
    const SushiAI::Config::ModelSpec model = model_from(K_GOOD_MODEL);

    EXPECT_EQ(model.input_shape.num_elements(), 6);
    ASSERT_EQ(model.layers.size(), 3u);
    EXPECT_EQ(model.layers[0].type, "linear");
    EXPECT_EQ(model.layers[1].type, "gelu");
    EXPECT_EQ(model.layers[2].type, "linear");

    // The attributes no longer carry "type"; see tests/functional/unit/README.md.
    EXPECT_FALSE(model.layers[0].attributes.has("type"));
    EXPECT_TRUE(model.layers[0].attributes.has("units"));
    EXPECT_EQ(model.layers[1].attributes.keys().size(), 0u);
}

TEST(Unit_Config, ChainsLayerWidthsFromTheInputFeatures)
{
    SushiAI::NN::LayerStack stack =
        SushiAI::NN::build_layer_stack(make_layer_registry(), model_from(K_GOOD_MODEL));

    EXPECT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.output_features(), 4);

    // An activation keeps the width it was handed; see tests/functional/unit/README.md.
    EXPECT_EQ(stack.layer(0).output_features(), 8);
    EXPECT_EQ(stack.layer(1).output_features(), 8);
}

TEST(Unit_Config, AConfiguredStackTracesToTheDeclaredOutputWidth)
{
    SushiAI::NN::LayerStack stack =
        SushiAI::NN::build_layer_stack(make_layer_registry(), model_from(K_GOOD_MODEL));

    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);
    const SushiAI::Graph::ValueId x = builder.input(SushiAI::Shape{4, 6}, "inputs");

    EXPECT_EQ(graph.value(stack.forward(builder, x)).shape, (SushiAI::Shape{4, 4}));
}

TEST(Unit_Config, AnUnknownLayerTypeListsEveryTypeThisBuildHas)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "hologram", "units": 8}]
    })");

    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), model),
                    "hologram", "layer type", "'linear'", "'gelu'");
}

TEST(Unit_Config, AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory)
{
    const SushiAI::Config::ModelSpec typo = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "unitss": 8}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), typo),
                    "layer 0 'linear'", "'unitss'");

    // An activation accepts no attribute; see tests/functional/unit/README.md.
    const SushiAI::Config::ModelSpec stray = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "gelu", "units": 8}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), stray),
                    "layer 0 'gelu'", "'units'");
}

TEST(Unit_Config, ALayerFactoryRefusesANonPositiveWidth)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 0}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), model),
                    "layer 0 'linear'", "'units'");
}

TEST(Unit_Config, UseBiasDefaultsToTrueAndCanBeTurnedOff)
{
    const SushiAI::NN::NodeRegistry registry = make_layer_registry();
    SushiAI::NN::LayerStack with_bias = SushiAI::NN::build_layer_stack(registry, model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "linear", "units": 4}]
    })"));
    SushiAI::NN::LayerStack without_bias = SushiAI::NN::build_layer_stack(registry, model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "linear", "units": 4, "use_bias": false}]
    })"));

    SushiAI::Graph::Graph a;
    SushiAI::Graph::Graph b;
    const SushiAI::Graph::GraphBuilder builder_a(a);
    const SushiAI::Graph::GraphBuilder builder_b(b);
    (void)with_bias.forward(builder_a, builder_a.input(SushiAI::Shape{2, 6}, "x"));
    (void)without_bias.forward(builder_b, builder_b.input(SushiAI::Shape{2, 6}, "x"));

    EXPECT_EQ(SushiAI::NN::parameter_count(with_bias), 2u);
    EXPECT_EQ(SushiAI::NN::parameter_count(without_bias), 1u);
}

TEST(Unit_Config, AMissingModelVersionIsAnErrorThatSaysWhy)
{
    SA_EXPECT_ERROR((void)model_from(R"({"input": {"features": 6}, "layers": []})"),
                    "sushiai_model", "misread");
}

TEST(Unit_Config, AFutureModelVersionIsAnError)
{
    SA_EXPECT_ERROR(
        (void)model_from(R"({"sushiai_model": 2, "input": {"features": 6}, "layers": []})"),
        "version 2", "version 1");
}

TEST(Unit_Config, AnUnknownTopLevelModelKeyIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "gelu"}], "optimizer": {"type": "adamw"}
    })"), "'optimizer'", "'layers'");
}

TEST(Unit_Config, AModelWithNoLayersIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "layers": []
    })"), "no layers");
}

TEST(Unit_Config, ANonPositiveInputWidthIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 0}, "layers": [{"type": "gelu"}]
    })"), "'features'", "positive");
}

TEST(Unit_Config, ALayerThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "layers": ["linear"]
    })"), "layer 0", "a string");
}

namespace
{
    using SushiAI::Config::K_MODEL_INPUT;
    using SushiAI::Config::ModelSpec;

    /** @brief Holds the version 1 convolutional classifier, the shape of the MNIST example. */
    constexpr std::string_view K_FIRST_VERSION_MODEL = R"({
        "sushiai_model": 1,
        "input": {"features": 784},
        "layers": [{"type": "unflatten", "height": 28, "width": 28, "channels": 1},
                   {"type": "conv2d", "filters": 8, "kernel": 3, "use_bias": true},
                   {"type": "relu"},
                   {"type": "flatten"},
                   {"type": "linear", "units": 10}]
    })";

    /** @brief Holds K_FIRST_VERSION_MODEL rewritten by hand as version 2. */
    constexpr std::string_view K_REWRITTEN_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [784]},
        "layers": [{"type": "unflatten", "height": 28, "width": 28, "channels": 1, "from": -1},
                   {"type": "conv2d", "filters": 8, "kernel": 3, "use_bias": true, "from": -1},
                   {"type": "relu", "from": -1},
                   {"type": "flatten", "from": -1},
                   {"type": "linear", "units": 10, "from": -1}]
    })";

    /** @brief Renders one scalar attribute as text, so two specs compare value by value. */
    std::string scalar_text(const Value& value)
    {
        if (value.kind() == Kind::Int)
            return "int " + std::to_string(value.as_int("attribute"));

        if (value.kind() == Kind::Bool)
            return value.as_bool("attribute") ? "bool true" : "bool false";

        if (value.kind() == Kind::String)
            return "string " + value.as_string("attribute");

        return std::string(SushiAI::Config::kind_name(value.kind()));
    }

    /** @brief Renders a whole spec as text: shape, rows with sources and attributes, outputs. */
    std::string spec_text(const ModelSpec& model)
    {
        std::string text = "input " + model.input_shape.to_string() + " features "
                         + std::to_string(model.input_shape.num_elements()) + "\n";

        for (const SushiAI::Config::LayerSpec& layer : model.layers)
        {
            text += layer.type + " from";
            for (const SushiAI::Config::SourceSpec& source : layer.from)
                text += " " + std::to_string(source.node) + ":'" + source.port + "'";

            for (const std::string& key : layer.attributes.keys())
                text += " " + key + "=" + scalar_text(layer.attributes.at(key, "attributes"));
            text += "\n";
        }

        for (const SushiAI::Config::OutputSpec& output : model.outputs)
            text += "output " + std::to_string(output.node) + ":'" + output.port + "' as '"
                  + output.name + "'\n";
        return text;
    }

    /** @brief Wraps @p layers and @p rest in a version 2 document with a rank-1 input. */
    ModelSpec current_model_from(std::string_view layers, std::string_view rest = "")
    {
        return model_from(R"({"format_version": 2, "input": {"shape": [6]}, "layers": )"
                          + std::string(layers) + std::string(rest) + "}");
    }
}

/** @brief Tests that a version 1 document parses to the spec its version 2 rewrite parses to. */
TEST(Unit_Config, AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite)
{
    const ModelSpec upgraded = model_from(K_FIRST_VERSION_MODEL);
    const ModelSpec rewritten = model_from(K_REWRITTEN_MODEL);

    EXPECT_EQ(spec_text(upgraded), spec_text(rewritten));
    EXPECT_EQ(spec_text(upgraded),
              "input [784] features 784\n"
              "unflatten from -1:'' height=int 28 width=int 28 channels=int 1\n"
              "conv2d from 0:'' filters=int 8 kernel=int 3 use_bias=bool true\n"
              "relu from 1:''\n"
              "flatten from 2:''\n"
              "linear from 3:'' units=int 10\n"
              "output 4:'' as ''\n");
}

/** @brief Tests that a version 1 document still fills the shape and the feature count. */
TEST(Unit_Config, AVersionOneDocumentFillsTheInputShapeAndTheFeatureCount)
{
    const ModelSpec model = model_from(K_GOOD_MODEL);

    EXPECT_EQ(model.input_shape, (SushiAI::Shape{6}));
    EXPECT_EQ(model.input_shape.num_elements(), 6);
    EXPECT_EQ(SushiAI::Config::K_MODEL_VERSION, 2);
}

/** @brief Tests that a version 2 input shape of rank 3 is kept and its elements counted. */
TEST(Unit_Config, AVersionTwoInputShapeKeepsItsRankAndCountsItsElements)
{
    const ModelSpec model = model_from(R"({
        "format_version": 2, "input": {"shape": [28, 28, 1]}, "layers": [{"type": "relu"}]
    })");

    EXPECT_EQ(model.input_shape, (SushiAI::Shape{28, 28, 1}));
    EXPECT_EQ(model.input_shape.num_elements(), 784);
}

/** @brief Tests that an input shape that is empty or holds a non-positive extent is refused. */
TEST(Unit_Config, AnEmptyOrNonPositiveInputShapeIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "format_version": 2, "input": {"shape": []}, "layers": [{"type": "relu"}]
    })"), "'input' 'shape'", "empty");

    SA_EXPECT_ERROR((void)model_from(R"({
        "format_version": 2, "input": {"shape": [28, 0, 1]}, "layers": [{"type": "relu"}]
    })"), "'shape' entry 1", "positive");

    SA_EXPECT_ERROR((void)model_from(R"({
        "format_version": 2, "input": {"features": 6}, "layers": [{"type": "relu"}]
    })"), "'features'", "'shape'");
}

/** @brief Tests that an input shape above the rank limit or the element limit is refused. */
TEST(Unit_Config, AnInputShapeAboveTheRankOrTheElementLimitIsAnError)
{
    std::string extents = "1";
    for (std::int32_t i = 0; i < SushiAI::K_MAX_RANK; ++i)
        extents += ", 1";

    SA_EXPECT_ERROR((void)model_from(R"({"format_version": 2, "input": {"shape": [)" + extents
                                     + R"(]}, "layers": [{"type": "relu"}]})"),
                    "'shape'", "at most");

    SA_EXPECT_ERROR((void)model_from(R"({
        "format_version": 2, "input": {"shape": [4294967296, 4294967296]},
        "layers": [{"type": "relu"}]
    })"), "'shape'", "entry 1", "more elements");
}

/** @brief Tests that a model document whose root is not an object is refused by kind. */
TEST(Unit_Config, AModelRootThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_ERROR((void)model_from("[1]"), "model.json", "'format_version'", "an array");
}

/** @brief Tests that a version above the newest is refused with both numbers named. */
TEST(Unit_Config, AVersionAboveTheNewestIsRefusedWithBothNumbers)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "format_version": 3, "input": {"shape": [6]}, "layers": [{"type": "relu"}]
    })"), "model.json", "version 3", "version 2");
}

/** @brief Tests that format_version 1, which no file was ever written with, is refused. */
TEST(Unit_Config, FormatVersionOneIsRefusedAndNamesTheVersionOneKey)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "format_version": 1, "input": {"features": 6}, "layers": [{"type": "relu"}]
    })"), "'format_version'", "'sushiai_model'");
}

/** @brief Tests that a version 1 document cannot use a version 2 key. */
TEST(Unit_Config, AVersionOneDocumentRefusesTheVersionTwoKeys)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "relu"}, {"type": "relu", "from": 0}]
    })"), "layer 1", "'from'", "version 1");

    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "defaults": {},
        "layers": [{"type": "relu"}]
    })"), "'defaults'", "'layers'");
}

/** @brief Tests that each form of a from entry resolves to the absolute row it names. */
TEST(Unit_Config, EachFromFormResolvesToAnAbsoluteRow)
{
    const ModelSpec model = current_model_from(R"([
        {"type": "relu"},
        {"type": "relu", "from": -1},
        {"type": "relu", "from": "input"},
        {"type": "concat", "from": [-1, 0]},
        {"type": "relu", "from": {"node": 3, "port": "boxes"}},
        {"type": "concat", "from": [{"node": 1}, "input", {"node": -1, "port": "scores"}]}
    ])");

    EXPECT_EQ(spec_text(model),
              "input [6] features 6\n"
              "relu from -1:''\n"
              "relu from 0:''\n"
              "relu from -1:''\n"
              "concat from 2:'' 0:''\n"
              "relu from 3:'boxes'\n"
              "concat from 1:'' -1:'' 4:'scores'\n"
              "output 5:'' as ''\n");
    EXPECT_EQ(model.layers[0].from[0].node, K_MODEL_INPUT);

    // "from" chooses the sources and is not handed to the factory as an attribute.
    EXPECT_FALSE(model.layers[3].attributes.has("from"));
}

/** @brief Tests that a negative from entry other than -1 fails with the row named. */
TEST(Unit_Config, ANegativeFromOtherThanMinusOneNamesTheRow)
{
    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu"}, {"type": "relu", "from": -2}
    ])"), "layer 2 of 'model.json'", "'from'", "row -2");

    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu"}, {"type": "concat", "from": [0, {"node": -3}]}
    ])"), "layer 2 of 'model.json'", "row -3");
}

/** @brief Tests that a from entry naming its own row or a later one fails with the row named. */
TEST(Unit_Config, AForwardReferenceNamesTheRow)
{
    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": 2}, {"type": "relu"}
    ])"), "layer 1 of 'model.json'", "row 2", "above row 1");

    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": [0, 1]}
    ])"), "layer 1 of 'model.json'", "row 1");
}

/** @brief Tests that an unknown key or an unknown form in a from entry fails with the row named. */
TEST(Unit_Config, AnUnknownKeyOrFormInAFromEntryNamesTheRow)
{
    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": {"node": 0, "prot": "boxes"}}
    ])"), "layer 1 of 'model.json'", "'prot'", "'port'");

    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": "previous"}
    ])"), "layer 1 of 'model.json'", "'previous'", "'input'");

    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": []}
    ])"), "layer 1 of 'model.json'", "empty");

    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": 0.5}
    ])"), "layer 1 of 'model.json'", "it is a number");

    SA_EXPECT_ERROR((void)current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": {"node": 0, "port": ""}}
    ])"), "layer 1 of 'model.json'", "'port'", "empty string");
}

/** @brief Tests that an unknown top-level key of a version 2 document is refused. */
TEST(Unit_Config, AnUnknownTopLevelKeyOfAVersionTwoDocumentIsAnError)
{
    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "relu"}])", R"(, "output": [])"),
                    "'output'", "'outputs'", "'defaults'");
}

/** @brief Tests that a default reaches a row that omits the key and no row that states it. */
TEST(Unit_Config, DefaultsFillAnOmittedKeyAndNeverOverrideAStatedOne)
{
    const ModelSpec model = current_model_from(R"([
        {"type": "conv", "filters": 16},
        {"type": "conv", "filters": 32, "bn_momentum": 7},
        {"type": "relu"}
    ])", R"(, "defaults": {"conv": {"bn_momentum": 3, "kernel": 5}})");

    EXPECT_EQ(model.layers[0].attributes.keys(),
              (std::vector<std::string>{"filters", "bn_momentum", "kernel"}));
    EXPECT_EQ(model.layers[0].attributes.at("bn_momentum", "row 0").as_int("bn_momentum"), 3);
    EXPECT_EQ(model.layers[0].attributes.at("kernel", "row 0").as_int("kernel"), 5);

    EXPECT_EQ(model.layers[1].attributes.keys(),
              (std::vector<std::string>{"filters", "bn_momentum", "kernel"}));
    EXPECT_EQ(model.layers[1].attributes.at("bn_momentum", "row 1").as_int("bn_momentum"), 7);

    // A default belongs to one type; a row of another type receives none of its keys.
    EXPECT_EQ(model.layers[2].attributes.keys().size(), 0u);
}

/** @brief Tests that a defaults block of the wrong shape is refused with the type named. */
TEST(Unit_Config, AMalformedDefaultsBlockIsAnError)
{
    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "conv"}])", R"(, "defaults": [])"),
                    "'defaults'", "an array");

    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "conv"}])",
                                             R"(, "defaults": {"conv": 3})"),
                    "'defaults' for 'conv'", "object");

    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "conv"}])",
                                             R"(, "defaults": {"conv": {"from": 0}})"),
                    "'defaults' for 'conv'", "'from'");

    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "conv"}])",
                                             R"(, "defaults": {"conv": {"type": "relu"}})"),
                    "'defaults' for 'conv'", "'type'");
}

/** @brief Tests that a defaults entry for a type no row states is refused with both named. */
TEST(Unit_Config, ADefaultsEntryForATypeNoLayerStatesIsAnError)
{
    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "conv"}, {"type": "relu"}])",
                                             R"(, "defaults": {"covn": {"bn_momentum": 3}})"),
                    "'defaults'", "'covn'", "'conv', 'relu'");
}

/** @brief Tests that absent outputs publish the last row and stated ones keep their fields. */
TEST(Unit_Config, OutputsDefaultToTheLastRowAndKeepWhatTheFileStates)
{
    const ModelSpec defaulted = current_model_from(R"([{"type": "relu"}, {"type": "relu"}])");
    ASSERT_EQ(defaulted.outputs.size(), 1u);
    EXPECT_EQ(defaulted.outputs[0].node, 1);
    EXPECT_TRUE(defaulted.outputs[0].port.empty());
    EXPECT_TRUE(defaulted.outputs[0].name.empty());

    const ModelSpec stated = current_model_from(
        R"([{"type": "relu"}, {"type": "relu"}])",
        R"(, "outputs": [{"node": 0}, {"node": 1, "port": "boxes"},
                         {"node": 1, "port": "boxes", "name": "predictions"}])");
    EXPECT_EQ(spec_text(stated),
              "input [6] features 6\n"
              "relu from -1:''\n"
              "relu from 0:''\n"
              "output 0:'' as ''\n"
              "output 1:'boxes' as ''\n"
              "output 1:'boxes' as 'predictions'\n");
}

/** @brief Tests that an output of the wrong kind, row, key or text, or a lone name, fails. */
TEST(Unit_Config, AMalformedOutputIsAnError)
{
    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "relu"}])",
                                             R"(, "outputs": [{"node": 1}])"),
                    "'outputs' entry 0", "row 1", "0 to 0");

    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "relu"}])",
                                             R"(, "outputs": [{"node": 0, "nme": "x"}])"),
                    "'outputs' entry 0", "'nme'", "'name'");

    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "relu"}])",
                                             R"(, "outputs": [{"node": 0, "name": "x"}])"),
                    "'outputs' entry 0", "'name'", "'port'");

    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "relu"}])", R"(, "outputs": [])"),
                    "'outputs'", "empty");

    SA_EXPECT_ERROR((void)current_model_from(R"([{"type": "relu"}])", R"(, "outputs": [3])"),
                    "'outputs' entry 0", "a whole number");

    SA_EXPECT_ERROR(
        (void)current_model_from(R"([{"type": "relu"}])",
                                 R"(, "outputs": [{"node": 0, "port": "", "name": "x"}])"),
        "'outputs' entry 0 'port'", "empty string");

    SA_EXPECT_ERROR(
        (void)current_model_from(R"([{"type": "relu"}])",
                                 R"(, "outputs": [{"node": 0, "port": "p", "name": ""}])"),
        "'outputs' entry 0 'name'", "empty string");
}

namespace
{
    /** @brief Parses @p text as a train document, resolving paths against ".". */
    SushiAI::Config::TrainSpec train_from(std::string_view text)
    {
        return SushiAI::Config::parse_train(parse_json(text, "train.json"), "train.json", ".");
    }

    constexpr std::string_view K_GOOD_TRAIN = R"({
        "sushiai_train": 1,
        "model": "model.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings", "batch_size": 8, "batches": 2, "classes": 4},
        "optimizer": {"type": "adamw", "learning_rate": 0.003},
        "epochs": 3,
        "checkpoint": "out/model.ckpt"
    })";

    /**
     * @class TemporaryDirectory
     * @brief A directory that removes itself, so a failing test leaves nothing behind.
     */
    class TemporaryDirectory
    {
        public:
            explicit TemporaryDirectory(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::filesystem::remove_all(path_);
                std::filesystem::create_directories(path_);
            }

            ~TemporaryDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            TemporaryDirectory(const TemporaryDirectory&) = delete;
            TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

            const std::filesystem::path& path() const noexcept { return path_; }

            /** @brief Writes @p text into this directory as @p name. */
            std::filesystem::path write(const char* name, std::string_view text) const
            {
                const std::filesystem::path file = path_ / name;
                std::ofstream out(file, std::ios::binary);
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                return file;
            }

        private:
            std::filesystem::path path_;
    };
}

TEST(Unit_Config, ParsesATrainDocumentIntoItsFourTypedBlocks)
{
    const SushiAI::Config::TrainSpec spec = train_from(K_GOOD_TRAIN);

    EXPECT_EQ(spec.objective_type, "cross_entropy");
    EXPECT_EQ(spec.data.type, "rings");
    EXPECT_EQ(spec.optimizer_type, "adamw");
    EXPECT_EQ(spec.epochs, 3);

    // Each block's "type" is separated out, so no factory has to allow it.
    EXPECT_FALSE(spec.optimizer_attributes.has("type"));
    EXPECT_DOUBLE_EQ(spec.optimizer_attributes.at("learning_rate", "optimizer")
                                              .as_real("learning_rate"), 0.003);
    EXPECT_EQ(spec.data.attributes.at("batch_size", "data").as_int("batch_size"), 8);
}

TEST(Unit_Config, ResolvesRelativePathsAgainstTheConfigsOwnDirectory)
{
    const TemporaryDirectory directory("sushiai_config_paths");
    const std::filesystem::path config = directory.write("train.json", K_GOOD_TRAIN);

    const SushiAI::Config::TrainSpec spec = SushiAI::Config::load_train(config);

    EXPECT_EQ(spec.model_path, directory.path() / "model.json");
    EXPECT_EQ(spec.checkpoint, directory.path() / "out" / "model.ckpt");
}

TEST(Unit_Config, LeavesAnAbsolutePathAlone)
{
    const std::filesystem::path absolute =
        std::filesystem::temp_directory_path() / "elsewhere" / "model.json";

    // generic_string() writes '/', which JSON does not escape; see tests/functional/unit/README.md.
    const std::string text =
        R"({"sushiai_train": 1, "model": ")" + absolute.generic_string() +
        R"(", "objective": {"type": "cross_entropy"},
            "data": {"type": "rings"}, "optimizer": {"type": "adamw"}})";

    const SushiAI::Config::TrainSpec spec =
        SushiAI::Config::parse_train(parse_json(text, "train.json"), "train.json",
                                     "/some/other/place");
    EXPECT_EQ(spec.model_path, absolute);
}

TEST(Unit_Config, EpochsDefaultsToOneAndMustBePositive)
{
    const SushiAI::Config::TrainSpec defaulted = train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })");
    EXPECT_EQ(defaulted.epochs, 1);
    EXPECT_TRUE(defaulted.checkpoint.empty());

    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "epochs": 0
    })"), "'epochs'", "positive");
}

TEST(Unit_Config, AnUnknownTopLevelTrainKeyIsAnError)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "epochz": 3
    })"), "'epochz'", "'epochs'");
}

TEST(Unit_Config, AMissingTrainBlockNamesIt)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "optimizer": {"type": "adamw"}
    })"), "'data'");
}

TEST(Unit_Config, ATypedBlockThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": "cross_entropy",
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"), "'objective'", "a string");
}

TEST(Unit_Config, AFutureTrainVersionIsAnError)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 2, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"), "version 2", "version 1");
}

TEST(Unit_Config, ParsesAnEvalDocument)
{
    const SushiAI::Config::DataSpec data = SushiAI::Config::parse_data(
        parse_json(R"({
            "sushiai_data": 1,
            "data": {"type": "rings", "batch_size": 512, "batches": 1, "seed": 3735928575}
        })", "eval.json"), "eval.json");

    EXPECT_EQ(data.type, "rings");
    EXPECT_EQ(data.attributes.at("batch_size", "data").as_int("batch_size"), 512);
    EXPECT_EQ(data.attributes.at("seed", "data").as_int("seed"), 3735928575);
}

TEST(Unit_Config, AnEvalDocumentCarriesNoModelBlock)
{
    // An eval document naming a model is an error; see tests/functional/unit/README.md.
    SA_EXPECT_ERROR((void)SushiAI::Config::parse_data(
        parse_json(R"({"sushiai_data": 1, "data": {"type": "rings"}, "model": "m.json"})",
                   "eval.json"), "eval.json"), "'model'", "'data'");
}

TEST(Unit_Config, LoadingAnAbsentFileNamesThePath)
{
    SA_EXPECT_ERROR((void)SushiAI::Config::load_train("no/such/train.json"),
                    "no/such/train.json");
}
