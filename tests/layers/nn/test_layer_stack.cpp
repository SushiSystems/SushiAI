/**************************************************************************/
/* test_layer_stack.cpp                                                   */
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
 * @file test_layer_stack.cpp
 * @brief Tests how a layer stack is built from a model document and the errors its factories raise.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>

using namespace SushiAI;
using NN::GELU;
using NN::LayerAdapter;
using NN::LayerStack;
using NN::Linear;
using NN::Sequential;
using SushiAI::Config::Kind;
using SushiAI::Config::Value;
using SushiAI::Config::parse_json;

namespace
{
    using Test::make_layer_registry;

    /** @brief Traces @p model and returns its parameters as "name:extents". */
    template<typename M>
    std::vector<std::string> traced_names(M& model, int64_t batch, int64_t features)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{batch, features}, "inputs");
        model.forward(builder, x);

        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });
        return names;
    }

    /** @brief 6 -> 8 -> gelu -> 4, as a runtime stack. */
    LayerStack make_stack()
    {
        LayerStack stack;
        stack.push(std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8));
        stack.push(std::make_unique<LayerAdapter<GELU>>(GELU{}, 8));
        stack.push(std::make_unique<LayerAdapter<Linear>>(Linear{8, 4}, 4));
        return stack;
    }

    /** @brief Returns one row of @p type with the keys of @p attributes that reads @p source. */
    Config::LayerSpec chain_row(std::string type, std::string_view attributes,
                                std::int64_t source)
    {
        Config::LayerSpec row;
        row.type = std::move(type);
        row.attributes = Config::parse_json(attributes, "<test>");
        row.from.push_back(Config::SourceSpec{source, std::string()});
        return row;
    }

    /** @brief Builds 6 -> linear 8 -> gelu -> linear 4 as a spec no parser filled. */
    Config::ModelSpec hand_built_chain()
    {
        Config::ModelSpec model;
        model.input_shape = Shape{6};
        model.layers.push_back(chain_row("linear", R"({"units": 8})", Config::K_MODEL_INPUT));
        model.layers.push_back(chain_row("gelu", "{}", 0));
        model.layers.push_back(chain_row("linear", R"({"units": 4})", 1));
        return model;
    }
}

TEST(Unit_LayerStack, NamesItsParametersExactlyAsSequentialDoes)
{
    Sequential compile_time{Linear{6, 8}, GELU{}, Linear{8, 4}};
    LayerStack run_time = make_stack();

    const std::vector<std::string> expected = traced_names(compile_time, 4, 6);
    EXPECT_EQ(traced_names(run_time, 4, 6), expected);

    // Literals keep both paths from going empty together; see tests/layers/nn/README.md.
    EXPECT_EQ(expected, (std::vector<std::string>{
        "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]", "2.bias:[4]"}));
}

TEST(Unit_LayerStack, KeepsTheIndexGapAParameterlessActivationLeaves)
{
    // Indices track layer positions, not parameter positions; see tests/layers/nn/README.md.
    LayerStack stack = make_stack();
    const std::vector<std::string> names = traced_names(stack, 4, 6);

    ASSERT_EQ(names.size(), 4u);
    EXPECT_EQ(names[2].substr(0, 2), "2.");
}

TEST(Unit_LayerStack, ReportsTheLastLayersOutputWidth)
{
    LayerStack stack = make_stack();
    EXPECT_EQ(stack.size(), 3u);
    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(stack.output_features(), 4);
}

TEST(Unit_LayerStack, TracesTheLayersInOrder)
{
    LayerStack stack = make_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");
    const Graph::ValueID y = stack.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 4}));
}

TEST(Unit_LayerStack, RefusesToTraceWhenEmpty)
{
    LayerStack stack;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");

    SA_EXPECT_ERROR((void)stack.forward(builder, x), "no layers", "identity");
}

TEST(Unit_LayerStack, RefusesAnOutputWidthWhenEmpty)
{
    const LayerStack stack;
    SA_EXPECT_ERROR((void)stack.output_features(), "no layers");
}

TEST(Unit_LayerStack, RefusesANullLayer)
{
    LayerStack stack;
    SA_EXPECT_ERROR(stack.push(nullptr), "null layer", "index 0");
}

TEST(Unit_LayerStack, ReportsAnOutOfRangeLayerIndex)
{
    const LayerStack stack = make_stack();
    EXPECT_NO_THROW((void)stack.layer(2));
    SA_EXPECT_ERROR((void)stack.layer(3), "layer 3", "holds 3");
}

TEST(Unit_LayerStack, TracingTwiceDoesNotDuplicateParameters)
{
    // Both traces go into one graph; see tests/layers/nn/README.md.
    LayerStack stack = make_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");

    (void)stack.forward(builder, x);
    EXPECT_EQ(NN::parameter_count(stack), 4u);

    (void)stack.forward(builder, x);
    EXPECT_EQ(NN::parameter_count(stack), 4u);
}

/** @brief Tests that an adapter built from a Shape reports that shape unchanged. */
TEST(Unit_LayerStack, AnAdapterBuiltFromAShapeReportsIt)
{
    const LayerAdapter<GELU> adapter(GELU{}, Shape{4, 5, 3});
    EXPECT_EQ(adapter.output_shape(), (Shape{4, 5, 3}));
}

/** @brief Tests that an adapter built from a width reports the rank-1 shape of that width. */
TEST(Unit_LayerStack, AnAdapterBuiltFromAWidthReportsARankOneShape)
{
    const LayerAdapter<Linear> adapter(Linear{6, 8}, 8);
    EXPECT_EQ(adapter.output_shape(), (Shape{8}));
    EXPECT_EQ(adapter.output_features(), 8);
}

/** @brief Tests that output_features() is the element count of output_shape(). */
TEST(Unit_LayerStack, OutputFeaturesEqualsTheElementCountOfTheOutputShape)
{
    const LayerAdapter<GELU> adapter(GELU{}, Shape{4, 5, 3});
    const NN::Layer& layer = adapter;

    EXPECT_EQ(layer.output_features(), 4 * 5 * 3);
    EXPECT_EQ(layer.output_features(), layer.output_shape().num_elements());
}

/** @brief Tests that a stack reports the last layer's per-sample shape and its element count. */
TEST(Unit_LayerStack, ReportsTheLastLayersOutputShape)
{
    EXPECT_EQ(make_stack().output_shape(), (Shape{4}));

    LayerStack image_stack = make_stack();
    image_stack.push(std::make_unique<LayerAdapter<GELU>>(GELU{}, Shape{2, 2, 1}));
    EXPECT_EQ(image_stack.output_shape(), (Shape{2, 2, 1}));
    EXPECT_EQ(image_stack.output_features(), 4);
}

/** @brief Tests that an empty stack refuses to report an output shape. */
TEST(Unit_LayerStack, RefusesAnOutputShapeWhenEmpty)
{
    const LayerStack stack;
    SA_EXPECT_ERROR((void)stack.output_shape(), "no layers", "output shape");
}

/** @brief Tests that build_layer_stack hands each layer's output shape to the next factory. */
TEST(Unit_LayerStack, BuildLayerStackThreadsEachOutputShapeIntoTheNextLayer)
{
    const LayerStack stack = NN::build_layer_stack(make_layer_registry(), hand_built_chain());

    ASSERT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.layer(0).output_shape(), (Shape{8}));
    EXPECT_EQ(stack.layer(1).output_shape(), (Shape{8}));
    EXPECT_EQ(stack.output_shape(), (Shape{4}));

    const auto& last = dynamic_cast<const LayerAdapter<Linear>&>(stack.layer(2));
    EXPECT_EQ(last.layer().in_features, 8);
}

/** @brief Tests that a hand-built spec seeds the chain from its input shape. */
TEST(Unit_LayerStack, BuildLayerStackSeedsTheChainFromTheInputShape)
{
    Config::ModelSpec model = hand_built_chain();
    model.input_shape = Shape{5};

    const LayerStack stack = NN::build_layer_stack(make_layer_registry(), model);

    const auto& first = dynamic_cast<const LayerAdapter<Linear>&>(stack.layer(0));
    EXPECT_EQ(first.layer().in_features, 5);
}

/** @brief Tests that a hand-built row that names no source is refused with its index and type. */
TEST(Unit_LayerStack, BuildLayerStackRefusesAHandBuiltRowWithoutASource)
{
    Config::ModelSpec model = hand_built_chain();
    model.layers[1].from.clear();

    SA_EXPECT_ERROR((void)NN::build_layer_stack(make_layer_registry(), model),
                    "layer 1 'gelu'", "the row before it");
}

/** @brief Tests that output_shape over an input shape is the last layer's output shape. */
TEST(Unit_LayerStack, OutputShapeOfAnInputIsTheLastLayersOutputShape)
{
    const LayerStack stack = NN::build_layer_stack(make_layer_registry(), hand_built_chain());

    EXPECT_EQ(stack.output_shape(Shape{6}), (Shape{4}));
    EXPECT_EQ(stack.output_shape(Shape{6}), stack.output_shape());
}

/** @brief Tests that an empty stack refuses an output shape in the layer form too. */
TEST(Unit_LayerStack, RefusesAnOutputShapeOfAnInputWhenEmpty)
{
    const LayerStack stack;
    SA_EXPECT_ERROR((void)stack.output_shape(Shape{6}), "no layers", "output shape");
}

/** @brief Tests that a factory hands a rank-3 per-sample shape to its layer with no axis merged. */
TEST(Unit_LayerStack, AnActivationFactoryKeepsARankThreeInputShape)
{
    const Config::Value attributes = Config::parse_json("{}", "<test>");
    const std::unique_ptr<NN::Layer> layer =
        NN::make_gelu(Shape{4, 4, 3}, attributes, "layer 1 'gelu'");

    EXPECT_EQ(layer->output_shape(), (Shape{4, 4, 3}));
    EXPECT_EQ(layer->output_features(), 48);
}

/** @brief Tests that a linear layer after a rank-3 input throws and tells the reader to flatten. */
TEST(Unit_LayerStack, ALinearFactoryRefusesARankThreeInputAndNamesFlatten)
{
    const Config::Value attributes = Config::parse_json(R"({"units": 8})", "<test>");

    SA_EXPECT_ERROR((void)NN::make_linear(Shape{4, 4, 3}, attributes, "layer 1 'linear'"),
                    "layer 1 'linear'", "[4, 4, 3]", "flatten");
    EXPECT_NO_THROW((void)NN::make_linear(Shape{48}, attributes, "layer 1 'linear'"));
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

    using SushiAI::Config::ModelSpec;

} // namespace

TEST(Unit_LayerStack, ChainsLayerWidthsFromTheInputFeatures)
{
    SushiAI::NN::LayerStack stack =
        SushiAI::NN::build_layer_stack(make_layer_registry(), model_from(K_GOOD_MODEL));

    EXPECT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.output_features(), 4);

    // An activation keeps the width it was handed; see tests/layers/nn/README.md.
    EXPECT_EQ(stack.layer(0).output_features(), 8);
    EXPECT_EQ(stack.layer(1).output_features(), 8);
}

TEST(Unit_LayerStack, AConfiguredStackTracesToTheDeclaredOutputWidth)
{
    SushiAI::NN::LayerStack stack =
        SushiAI::NN::build_layer_stack(make_layer_registry(), model_from(K_GOOD_MODEL));

    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);
    const SushiAI::Graph::ValueID x = builder.input(SushiAI::Shape{4, 6}, "inputs");

    EXPECT_EQ(graph.value(stack.forward(builder, x)).shape, (SushiAI::Shape{4, 4}));
}

TEST(Unit_LayerStack, AnUnknownLayerTypeListsEveryTypeThisBuildHas)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "hologram", "units": 8}]
    })");

    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), model),
                    "hologram", "layer type", "'linear'", "'gelu'");
}

TEST(Unit_LayerStack, AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory)
{
    const SushiAI::Config::ModelSpec typo = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "unitss": 8}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), typo),
                    "layer 0 'linear'", "'unitss'");

    // An activation accepts no attribute; see tests/layers/nn/README.md.
    const SushiAI::Config::ModelSpec stray = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "gelu", "units": 8}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), stray),
                    "layer 0 'gelu'", "'units'");
}

TEST(Unit_LayerStack, ALayerFactoryRefusesANonPositiveWidth)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 0}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(make_layer_registry(), model),
                    "layer 0 'linear'", "'units'");
}

TEST(Unit_LayerStack, UseBiasDefaultsToTrueAndCanBeTurnedOff)
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
