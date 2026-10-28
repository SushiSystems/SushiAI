/**************************************************************************/
/* test_model_config.cpp                                                  */
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
 * @file test_model_config.cpp
 * @brief Tests how a model document parses into a ModelSpec and which documents it refuses.
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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/model_text/model_text.hpp>

using SushiAI::Result;
using SushiAI::StatusCode;
using SushiAI::Config::Kind;
using SushiAI::Config::Value;
using SushiAI::Config::parse_json;

namespace
{
    /** @brief Parses @p text as a model document called "model.json". */
    Result<SushiAI::Config::ModelSpec> model_from(std::string_view text)
    {
        SA_TRY_ASSIGN(const Value root, parse_json(text, "model.json"));
        return SushiAI::Config::parse_model(root, "model.json");
    }

    constexpr std::string_view K_GOOD_MODEL = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 8},
                   {"type": "gelu"},
                   {"type": "linear", "units": 4}]
    })";

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
    Result<std::string> scalar_text(const Value& value)
    {
        if (value.kind() == Kind::Int)
        {
            SA_TRY_ASSIGN(const std::int64_t whole, value.as_int("attribute"));
            return "int " + std::to_string(whole);
        }

        if (value.kind() == Kind::Bool)
        {
            SA_TRY_ASSIGN(const bool flag, value.as_bool("attribute"));
            return std::string(flag ? "bool true" : "bool false");
        }

        if (value.kind() == Kind::String)
        {
            SA_TRY_ASSIGN(const std::string& text, value.as_string("attribute"));
            return "string " + text;
        }

        return std::string(SushiAI::Config::kind_name(value.kind()));
    }

    /** @brief Renders a whole spec as text: shape, rows with sources and attributes, outputs. */
    Result<std::string> spec_text(const ModelSpec& model)
    {
        std::string text = "input " + model.input_shape.to_string() + " features "
                         + std::to_string(model.input_shape.num_elements()) + "\n";

        for (const SushiAI::Config::LayerSpec& layer : model.layers)
        {
            text += layer.type + " from";
            for (const SushiAI::Config::SourceSpec& source : layer.from)
                text += " " + std::to_string(source.node) + ":'" + source.port + "'";

            for (const std::string& key : layer.attributes.keys())
            {
                SA_TRY_ASSIGN(const Value& attribute, layer.attributes.at(key, "attributes"));
                SA_TRY_ASSIGN(const std::string rendered, scalar_text(attribute));
                text += " " + key + "=" + rendered;
            }
            text += "\n";
        }

        for (const SushiAI::Config::OutputSpec& output : model.outputs)
            text += "output " + std::to_string(output.node) + ":'" + output.port + "' as '"
                  + output.name + "'\n";
        return text;
    }

    /** @brief Wraps @p layers and @p rest in a version 2 document with a rank-1 input. */
    Result<ModelSpec> current_model_from(std::string_view layers, std::string_view rest = "")
    {
        return model_from(R"({"format_version": 2, "input": {"shape": [6]}, "layers": )"
                          + std::string(layers) + std::string(rest) + "}");
    }

} // namespace

TEST(Small_ModelConfig, ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::ModelSpec model, model_from(K_GOOD_MODEL));

    EXPECT_EQ(model.input_shape.num_elements(), 6);
    ASSERT_EQ(model.layers.size(), 3u);
    EXPECT_EQ(model.layers[0].type, "linear");
    EXPECT_EQ(model.layers[1].type, "gelu");
    EXPECT_EQ(model.layers[2].type, "linear");

    // The attributes no longer carry "type"; see tests/core/config/README.md.
    EXPECT_FALSE(model.layers[0].attributes.has("type"));
    EXPECT_TRUE(model.layers[0].attributes.has("units"));
    EXPECT_EQ(model.layers[1].attributes.keys().size(), 0u);
}

TEST(Small_ModelConfig, AMissingModelVersionIsAnErrorThatSaysWhy)
{
    SA_EXPECT_FAILURE(model_from(R"({"input": {"features": 6}, "layers": []})"),
                      StatusCode::InvalidFile, "sushiai_model", "misread");
}

TEST(Small_ModelConfig, AFutureModelVersionIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({"sushiai_model": 2, "input": {"features": 6}, "layers": []})"),
                      StatusCode::InvalidFile, "version 2", "version 1");
}

TEST(Small_ModelConfig, AnUnknownTopLevelModelKeyIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "gelu"}], "optimizer": {"type": "adamw"}
    })"), StatusCode::InvalidDescription, "'optimizer'", "'layers'");
}

TEST(Small_ModelConfig, AModelWithNoLayersIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "layers": []
    })"), StatusCode::InvalidDescription, "no layers");
}

TEST(Small_ModelConfig, ANonPositiveInputWidthIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "sushiai_model": 1, "input": {"features": 0}, "layers": [{"type": "gelu"}]
    })"), StatusCode::InvalidDescription, "'features'", "positive");
}

TEST(Small_ModelConfig, ALayerThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "layers": ["linear"]
    })"), StatusCode::InvalidDescription, "layer 0", "a string");
}

/** @brief Tests that a version 1 document parses to the spec its version 2 rewrite parses to. */
TEST(Small_ModelConfig, AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec upgraded, model_from(K_FIRST_VERSION_MODEL));
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec rewritten, model_from(K_REWRITTEN_MODEL));

    SA_ASSERT_OK_AND_ASSIGN(const std::string upgraded_text, spec_text(upgraded));
    SA_ASSERT_OK_AND_ASSIGN(const std::string rewritten_text, spec_text(rewritten));
    EXPECT_EQ(upgraded_text, rewritten_text);
    EXPECT_EQ(upgraded_text,
              "input [784] features 784\n"
              "unflatten from -1:'' height=int 28 width=int 28 channels=int 1\n"
              "conv2d from 0:'' filters=int 8 kernel=int 3 use_bias=bool true\n"
              "relu from 1:''\n"
              "flatten from 2:''\n"
              "linear from 3:'' units=int 10\n"
              "output 4:'' as ''\n");
}

/** @brief Tests that a version 1 document still fills the shape and the feature count. */
TEST(Small_ModelConfig, AVersionOneDocumentFillsTheInputShapeAndTheFeatureCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec model, model_from(K_GOOD_MODEL));

    EXPECT_EQ(model.input_shape, (SushiAI::Shape{6}));
    EXPECT_EQ(model.input_shape.num_elements(), 6);
    EXPECT_EQ(SushiAI::Config::K_MODEL_VERSION, 2);
}

/** @brief Tests that a version 2 input shape of rank 3 is kept and its elements counted. */
TEST(Small_ModelConfig, AVersionTwoInputShapeKeepsItsRankAndCountsItsElements)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec model, model_from(R"({
        "format_version": 2, "input": {"shape": [28, 28, 1]}, "layers": [{"type": "relu"}]
    })"));

    EXPECT_EQ(model.input_shape, (SushiAI::Shape{28, 28, 1}));
    EXPECT_EQ(model.input_shape.num_elements(), 784);
}

/** @brief Tests that an input shape that is empty or holds a non-positive extent is refused. */
TEST(Small_ModelConfig, AnEmptyOrNonPositiveInputShapeIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"shape": []}, "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'input' 'shape'", "empty");

    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"shape": [28, 0, 1]}, "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'shape' entry 1", "positive");

    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"features": 6}, "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'features'", "'shape'");
}

/** @brief Tests that a file without "dtype" has a float32 input and that "int32" is read. */
TEST(Small_ModelConfig, TheInputElementTypeIsFloat32UnlessTheFileStatesInt32)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::ModelSpec floating, model_from(R"({
        "format_version": 2, "input": {"shape": [8]}, "layers": [{"type": "relu"}]
    })"));
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::ModelSpec stated, model_from(R"({
        "format_version": 2, "input": {"shape": [8], "dtype": "float32"},
        "layers": [{"type": "relu"}]
    })"));
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::ModelSpec indices, model_from(R"({
        "format_version": 2, "input": {"shape": [8], "dtype": "int32"},
        "layers": [{"type": "relu"}]
    })"));

    EXPECT_EQ(floating.input_dtype, SushiAI::DType::Float32);
    EXPECT_EQ(stated.input_dtype, SushiAI::DType::Float32);
    EXPECT_EQ(indices.input_dtype, SushiAI::DType::Int32);
}

/** @brief Tests that a stated "fit" is kept as written and an absent one is the empty name. */
TEST(Small_ModelConfig, TheInputFitIsKeptAsTheFileStatesIt)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::ModelSpec unstated, model_from(R"({
        "format_version": 2, "input": {"shape": [8]}, "layers": [{"type": "relu"}]
    })"));
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::ModelSpec stated, model_from(R"({
        "format_version": 2, "input": {"shape": [4, 4, 3], "fit": "resize"},
        "layers": [{"type": "relu"}]
    })"));

    EXPECT_TRUE(unstated.input_fit.empty());
    EXPECT_EQ(stated.input_fit, "resize");
}

/** @brief Tests that a "fit" outside the two names, or no string, is refused. */
TEST(Small_ModelConfig, AnInputFitOutsideTheListIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"shape": [8], "fit": "crop"},
        "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'input' 'fit'", "'crop'", "'resize', 'letterbox'");

    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"shape": [8], "fit": 1},
        "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'input' 'fit'");
}

/** @brief Tests that an input element type the reader does not list, or no string, is refused. */
TEST(Small_ModelConfig, AnInputElementTypeOutsideTheListIsAnError)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"shape": [8], "dtype": "int64"},
        "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'input' 'dtype'", "'int64'", "'float32', 'int32'");

    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"shape": [8], "dtype": 32},
        "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'input' 'dtype'");
}

/** @brief Tests that an input shape above the rank limit or the element limit is refused. */
TEST(Small_ModelConfig, AnInputShapeAboveTheRankOrTheElementLimitIsAnError)
{
    std::string extents = "1";
    for (std::int32_t i = 0; i < SushiAI::K_MAX_RANK; ++i)
        extents += ", 1";

    SA_EXPECT_FAILURE(model_from(R"({"format_version": 2, "input": {"shape": [)" + extents
                                 + R"(]}, "layers": [{"type": "relu"}]})"),
                      StatusCode::InvalidDescription, "'shape'", "at most");

    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 2, "input": {"shape": [4294967296, 4294967296]},
        "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'shape'", "entry 1", "more elements");
}

/** @brief Tests that a model document whose root is not an object is refused by kind. */
TEST(Small_ModelConfig, AModelRootThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_FAILURE(model_from("[1]"), StatusCode::InvalidFile, "model.json",
                      "'format_version'", "an array");
}

/** @brief Tests that a version above the newest is refused with both numbers named. */
TEST(Small_ModelConfig, AVersionAboveTheNewestIsRefusedWithBothNumbers)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 3, "input": {"shape": [6]}, "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidFile, "model.json", "version 3", "version 2");
}

/** @brief Tests that format_version 1, which no file was ever written with, is refused. */
TEST(Small_ModelConfig, FormatVersionOneIsRefusedAndNamesTheVersionOneKey)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "format_version": 1, "input": {"features": 6}, "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidFile, "'format_version'", "'sushiai_model'");
}

/** @brief Tests that a version 1 document cannot use a version 2 key. */
TEST(Small_ModelConfig, AVersionOneDocumentRefusesTheVersionTwoKeys)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "relu"}, {"type": "relu", "from": 0}]
    })"), StatusCode::InvalidDescription, "layer 1", "'from'", "version 1");

    SA_EXPECT_FAILURE(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "defaults": {},
        "layers": [{"type": "relu"}]
    })"), StatusCode::InvalidDescription, "'defaults'", "'layers'");
}

/** @brief Tests that each form of a from entry resolves to the absolute row it names. */
TEST(Small_ModelConfig, EachFromFormResolvesToAnAbsoluteRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec model, current_model_from(R"([
        {"type": "relu"},
        {"type": "relu", "from": -1},
        {"type": "relu", "from": "input"},
        {"type": "concat", "from": [-1, 0]},
        {"type": "relu", "from": {"node": 3, "port": "boxes"}},
        {"type": "concat", "from": [{"node": 1}, "input", {"node": -1, "port": "scores"}]}
    ])"));

    SA_ASSERT_OK_AND_ASSIGN(const std::string model_text, spec_text(model));
    EXPECT_EQ(model_text,
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
TEST(Small_ModelConfig, ANegativeFromOtherThanMinusOneNamesTheRow)
{
    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu"}, {"type": "relu", "from": -2}
    ])"), StatusCode::InvalidDescription, "layer 2 of 'model.json'", "'from'", "row -2");

    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu"}, {"type": "concat", "from": [0, {"node": -3}]}
    ])"), StatusCode::InvalidDescription, "layer 2 of 'model.json'", "row -3");
}

/** @brief Tests that a from entry naming its own row or a later one fails with the row named. */
TEST(Small_ModelConfig, AForwardReferenceNamesTheRow)
{
    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": 2}, {"type": "relu"}
    ])"), StatusCode::InvalidDescription, "layer 1 of 'model.json'", "row 2", "above row 1");

    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": [0, 1]}
    ])"), StatusCode::InvalidDescription, "layer 1 of 'model.json'", "row 1");
}

/** @brief Tests that an unknown key or an unknown form in a from entry fails with the row named. */
TEST(Small_ModelConfig, AnUnknownKeyOrFormInAFromEntryNamesTheRow)
{
    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": {"node": 0, "prot": "boxes"}}
    ])"), StatusCode::InvalidDescription, "layer 1 of 'model.json'", "'prot'", "'port'");

    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": "previous"}
    ])"), StatusCode::InvalidDescription, "layer 1 of 'model.json'", "'previous'", "'input'");

    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": []}
    ])"), StatusCode::InvalidDescription, "layer 1 of 'model.json'", "empty");

    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": 0.5}
    ])"), StatusCode::InvalidDescription, "layer 1 of 'model.json'", "it is a number");

    SA_EXPECT_FAILURE(current_model_from(R"([
        {"type": "relu"}, {"type": "relu", "from": {"node": 0, "port": ""}}
    ])"), StatusCode::InvalidDescription, "layer 1 of 'model.json'", "'port'", "empty string");
}

/** @brief Tests that an unknown top-level key of a version 2 document is refused. */
TEST(Small_ModelConfig, AnUnknownTopLevelKeyOfAVersionTwoDocumentIsAnError)
{
    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "output": [])"),
                      StatusCode::InvalidDescription, "'output'", "'outputs'",
                      "'defaults'");
}

/** @brief Tests that a default reaches a row that omits the key and no row that states it. */
TEST(Small_ModelConfig, DefaultsFillAnOmittedKeyAndNeverOverrideAStatedOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec model, current_model_from(R"([
        {"type": "conv", "filters": 16},
        {"type": "conv", "filters": 32, "bn_momentum": 7},
        {"type": "relu"}
    ])", R"(, "defaults": {"conv": {"bn_momentum": 3, "kernel": 5}})"));

    EXPECT_EQ(model.layers[0].attributes.keys(),
              (std::vector<std::string>{"filters", "bn_momentum", "kernel"}));
    SA_ASSERT_OK_AND_ASSIGN(const Value& stated_momentum,
                            model.layers[0].attributes.at("bn_momentum", "row 0"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t momentum, stated_momentum.as_int("bn_momentum"));
    EXPECT_EQ(momentum, 3);
    SA_ASSERT_OK_AND_ASSIGN(const Value& stated_kernel,
                            model.layers[0].attributes.at("kernel", "row 0"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t kernel, stated_kernel.as_int("kernel"));
    EXPECT_EQ(kernel, 5);

    EXPECT_EQ(model.layers[1].attributes.keys(),
              (std::vector<std::string>{"filters", "bn_momentum", "kernel"}));
    SA_ASSERT_OK_AND_ASSIGN(const Value& second_momentum,
                            model.layers[1].attributes.at("bn_momentum", "row 1"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t second, second_momentum.as_int("bn_momentum"));
    EXPECT_EQ(second, 7);

    // A default belongs to one type; a row of another type receives none of its keys.
    EXPECT_EQ(model.layers[2].attributes.keys().size(), 0u);
}

/** @brief Tests that a defaults block of the wrong shape is refused with the type named. */
TEST(Small_ModelConfig, AMalformedDefaultsBlockIsAnError)
{
    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "conv"}])", R"(, "defaults": [])"),
                      StatusCode::InvalidDescription, "'defaults'", "an array");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "conv"}])", R"(, "defaults": {"conv": 3})"),
                      StatusCode::InvalidDescription, "'defaults' for 'conv'", "object");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "conv"}])",
                                         R"(, "defaults": {"conv": {"from": 0}})"),
                      StatusCode::InvalidDescription, "'defaults' for 'conv'", "'from'");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "conv"}])",
                                         R"(, "defaults": {"conv": {"type": "relu"}})"),
                      StatusCode::InvalidDescription, "'defaults' for 'conv'", "'type'");
}

/** @brief Tests that a defaults entry for a type no row states is refused with both named. */
TEST(Small_ModelConfig, ADefaultsEntryForATypeNoLayerStatesIsAnError)
{
    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "conv"}, {"type": "relu"}])",
                                         R"(, "defaults": {"covn": {"bn_momentum": 3}})"),
                      StatusCode::InvalidDescription, "'defaults'", "'covn'", "'conv', 'relu'");
}

/** @brief Tests that absent outputs publish the last row and stated ones keep their fields. */
TEST(Small_ModelConfig, OutputsDefaultToTheLastRowAndKeepWhatTheFileStates)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec defaulted,
                            current_model_from(R"([{"type": "relu"}, {"type": "relu"}])"));
    ASSERT_EQ(defaulted.outputs.size(), 1u);
    EXPECT_EQ(defaulted.outputs[0].node, 1);
    EXPECT_TRUE(defaulted.outputs[0].port.empty());
    EXPECT_TRUE(defaulted.outputs[0].name.empty());

    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec stated, current_model_from(
        R"([{"type": "relu"}, {"type": "relu"}])",
        R"(, "outputs": [{"node": 0}, {"node": 1, "port": "boxes"},
                         {"node": 1, "port": "boxes", "name": "predictions"}])"));
    SA_ASSERT_OK_AND_ASSIGN(const std::string stated_text, spec_text(stated));
    EXPECT_EQ(stated_text,
              "input [6] features 6\n"
              "relu from -1:''\n"
              "relu from 0:''\n"
              "output 0:'' as ''\n"
              "output 1:'boxes' as ''\n"
              "output 1:'boxes' as 'predictions'\n");
}

/** @brief Tests that an output of the wrong kind, row, key or text, or a lone name, fails. */
TEST(Small_ModelConfig, AMalformedOutputIsAnError)
{
    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "outputs": [{"node": 1}])"),
                      StatusCode::InvalidDescription, "'outputs' entry 0", "row 1", "0 to 0");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])",
                                         R"(, "outputs": [{"node": 0, "nme": "x"}])"),
                      StatusCode::InvalidDescription, "'outputs' entry 0", "'nme'", "'name'");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])",
                                         R"(, "outputs": [{"node": 0, "name": "x"}])"),
                      StatusCode::InvalidDescription, "'outputs' entry 0", "'name'", "'port'");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "outputs": [])"),
                      StatusCode::InvalidDescription, "'outputs'", "empty");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "outputs": [3])"),
                      StatusCode::InvalidDescription, "'outputs' entry 0",
                      "a whole number");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])",
                                         R"(, "outputs": [{"node": 0, "port": "", "name": "x"}])"),
                      StatusCode::InvalidDescription, "'outputs' entry 0 'port'", "empty string");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])",
                                         R"(, "outputs": [{"node": 0, "port": "p", "name": ""}])"),
                      StatusCode::InvalidDescription, "'outputs' entry 0 'name'", "empty string");
}

/** @brief Tests that a file without a task parses to a null task. */
TEST(Small_ModelConfig, AModelWithoutATaskHoldsANullTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec current, current_model_from(R"([{"type": "relu"}])"));
    EXPECT_TRUE(current.task.is_null());

    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec first, model_from(K_FIRST_VERSION_MODEL));
    EXPECT_TRUE(first.task.is_null());
}

/** @brief Tests that a task is kept as the file states it, with keys this module does not read. */
TEST(Small_ModelConfig, ATaskIsKeptWholeWithItsKeysInFileOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const ModelSpec model, current_model_from(
        R"([{"type": "relu"}])",
        R"(, "task": {"classes": 80, "type": "anchor_detection", "heads": [{"output": "a"}]})"));

    ASSERT_EQ(model.task.kind(), Kind::Object);
    EXPECT_EQ(model.task.keys(), (std::vector<std::string>{"classes", "type", "heads"}));
    SA_ASSERT_OK_AND_ASSIGN(const Value& type, model.task.at("type", "task"));
    SA_ASSERT_OK_AND_ASSIGN(const std::string& type_name, type.as_string("type"));
    EXPECT_EQ(type_name, "anchor_detection");
    SA_ASSERT_OK_AND_ASSIGN(const Value& classes, model.task.at("classes", "task"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t class_count, classes.as_int("classes"));
    EXPECT_EQ(class_count, 80);
    SA_ASSERT_OK_AND_ASSIGN(const Value& heads, model.task.at("heads", "task"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Value>& head_rows, heads.as_array("heads"));
    EXPECT_EQ(head_rows.size(), 1u);
}

/** @brief Tests that a task that is not an object with a string type fails with the key named. */
TEST(Small_ModelConfig, AMalformedTaskIsAnError)
{
    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "task": [])"),
                      StatusCode::InvalidDescription, "'model.json's 'task'", "an array");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "task": null)"),
                      StatusCode::InvalidDescription, "'model.json's 'task'", "null");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "task": {"classes": 80})"),
                      StatusCode::InvalidDescription, "'model.json's 'task'", "no 'type'",
                      "'classes'");

    SA_EXPECT_FAILURE(current_model_from(R"([{"type": "relu"}])", R"(, "task": {"type": 3})"),
                      StatusCode::InvalidDescription, "'task' 'type'", "a whole number");
}

/** @brief Tests that a version 1 document that states a task is refused. */
TEST(Small_ModelConfig, AVersionOneDocumentRefusesATask)
{
    SA_EXPECT_FAILURE(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "layers": [{"type": "gelu"}],
        "task": {"type": "anchor_detection"}
    })"), StatusCode::InvalidDescription, "'task'", "'layers'");
}
