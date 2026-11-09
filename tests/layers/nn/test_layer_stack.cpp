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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
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

namespace
{
    using Test::attributes_from;
    using Test::make_layer_registry;
    using Test::model_from;

    /** @brief Traces @p model and returns its parameters as "name:extents". */
    template<typename M>
    std::vector<std::string> traced_names(M& model, int64_t batch, int64_t features)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{batch, features}, "inputs");
        SA_EXPECT_OK(model.forward(builder, x));

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
        row.attributes = attributes_from(attributes);
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

    /** @brief Holds the model file of 6 -> linear 8 -> gelu -> linear 4. */
    constexpr std::string_view K_GOOD_MODEL = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 8},
                   {"type": "gelu"},
                   {"type": "linear", "units": 4}]
    })";
}

/** @brief Tests that the stack names its parameters exactly as a Sequential of the same layers. */
TEST(Small_LayerStack, NamesItsParametersExactlyAsSequentialDoes)
{
    Sequential compile_time{Linear{6, 8}, GELU{}, Linear{8, 4}};
    LayerStack run_time = make_stack();

    const std::vector<std::string> expected = traced_names(compile_time, 4, 6);
    EXPECT_EQ(traced_names(run_time, 4, 6), expected);

    // Literals keep both paths from going empty together; see tests/layers/nn/README.md.
    EXPECT_EQ(expected, (std::vector<std::string>{
        "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]", "2.bias:[4]"}));
}

/** @brief Tests that output_features() of a stack is the width of its last layer. */
TEST(Small_LayerStack, OutputFeaturesIsTheLastLayersWidth)
{
    const LayerStack stack = make_stack();

    EXPECT_EQ(stack.output_features(), 4);
}

namespace
{
    /** @brief Builds the 6 -> 8 -> gelu -> 4 stack from adapters, as a caller of push() does. */
    LayerStack stack_from_adapters()
    {
        return make_stack();
    }

    /** @brief Builds the 6 -> 8 -> gelu -> 4 stack from the model file K_GOOD_MODEL. */
    LayerStack stack_from_a_model_file()
    {
        return Test::require_value(
            NN::build_layer_stack(make_layer_registry(), model_from(K_GOOD_MODEL)));
    }

    /** @brief Names a way to make the 6 -> 8 -> gelu -> 4 stack. */
    struct StackSourceRow
    {
        std::string name;            /**< The row's name. */
        LayerStack (*build)();       /**< Builds the stack. */
    };
}

/** @brief Runs the trace case over a stack built by hand and one built from a model file. */
class Small_LayerStackTrace : public ::testing::TestWithParam<StackSourceRow>
{
};

/** @brief Tests that the stack of the row traces its layers in order to the last layer's shape. */
TEST_P(Small_LayerStackTrace, TracesTheLayersInOrder)
{
    LayerStack stack = GetParam().build();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, stack.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 4}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerStackTrace,
    ::testing::Values(StackSourceRow{"BuiltByHand", stack_from_adapters},
                      StackSourceRow{"BuiltFromAModelFile", stack_from_a_model_file}),
    [](const ::testing::TestParamInfo<StackSourceRow>& info) { return info.param.name; });

/** @brief Tests that an empty stack refuses to trace and names itself. */
TEST(Small_LayerStack, RefusesToTraceWhenEmpty)
{
    LayerStack stack;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");

    SA_EXPECT_FAILURE(stack.forward(builder, x), StatusCode::InvalidDescription, "no layers",
                      "identity");
}

/** @brief Tests that a stack of three layers reports the last one by index 2. */
TEST(Small_LayerStack, ALayerInRangeReportsItsWidth)
{
    const LayerStack stack = make_stack();

    EXPECT_EQ(stack.layer(2).output_features(), 4);
}

/** @brief Tests that an empty stack stops the process on an output width. */
TEST(Medium_LayerStackDeathTest, AnEmptyStackStopsTheProcessOnAnOutputWidth)
{
    const LayerStack stack;
    SA_EXPECT_FATAL((void)stack.output_features(), "no layers");
}

/** @brief Tests that pushing a null layer stops the process, naming the index. */
TEST(Medium_LayerStackDeathTest, ANullLayerStopsTheProcess)
{
    LayerStack stack;
    SA_EXPECT_FATAL(stack.push(nullptr), "null layer", "index 0");
}

/** @brief Tests that a layer index past the end stops the process. */
TEST(Medium_LayerStackDeathTest, AnOutOfRangeLayerIndexStopsTheProcess)
{
    const LayerStack stack = make_stack();
    SA_EXPECT_FATAL((void)stack.layer(3), "layer 3", "holds 3");
}

/** @brief Tests that tracing a stack twice into one graph does not duplicate its parameters. */
TEST(Small_LayerStack, TracingTwiceDoesNotDuplicateParameters)
{
    // Both traces go into one graph; see tests/layers/nn/README.md.
    LayerStack stack = make_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");

    SA_EXPECT_OK(stack.forward(builder, x));
    EXPECT_EQ(NN::parameter_count(stack), 4u);

    SA_EXPECT_OK(stack.forward(builder, x));
    EXPECT_EQ(NN::parameter_count(stack), 4u);
}

/** @brief Tests that an adapter built from a width reports the rank-1 shape of that width. */
TEST(Small_LayerStack, AnAdapterBuiltFromAWidthReportsARankOneShape)
{
    const LayerAdapter<Linear> adapter(Linear{6, 8}, 8);
    EXPECT_EQ(adapter.output_shape(), (Shape{8}));
    EXPECT_EQ(adapter.output_features(), 8);
}

namespace
{
    /** @brief Builds the stack of the model chain a caller wrote by hand. */
    LayerStack stack_from_a_hand_built_chain()
    {
        return Test::require_value(
            NN::build_layer_stack(make_layer_registry(), hand_built_chain()));
    }

    /** @brief Builds make_stack() with a rank-3 gelu layer on its end. */
    LayerStack stack_with_an_image_tail()
    {
        LayerStack stack = make_stack();
        stack.push(std::make_unique<LayerAdapter<GELU>>(GELU{}, Shape{2, 2, 1}));
        return stack;
    }

    /** @brief Names a stack, the way its output shape is asked and the shape it must give. */
    struct OutputShapeRow
    {
        std::string name;        /**< The row's name. */
        LayerStack (*build)();   /**< Builds the stack. */
        bool over_an_input;      /**< Whether the shape is asked over the input shape [6]. */
        Shape expected;          /**< The per-sample shape of the last layer. */
    };
}

/** @brief Runs the output shape case over each overload and each stack. */
class Small_LayerStackOutputShape : public ::testing::TestWithParam<OutputShapeRow>
{
};

/** @brief Tests that the stack reports the per-sample shape of its last layer. */
TEST_P(Small_LayerStackOutputShape, ReportsTheLastLayersOutputShape)
{
    const OutputShapeRow& row = GetParam();
    const LayerStack stack = row.build();

    if (row.over_an_input)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Shape of_input, stack.output_shape(Shape{6}));
        EXPECT_EQ(of_input, row.expected);
    }
    else
    {
        EXPECT_EQ(stack.output_shape(), row.expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerStackOutputShape,
    ::testing::Values(
        OutputShapeRow{"OfAFeatureVector", stack_from_a_hand_built_chain, false, Shape{4}},
        OutputShapeRow{"OfAnImageTail", stack_with_an_image_tail, false, Shape{2, 2, 1}},
        OutputShapeRow{"OverTheInputShape", stack_from_a_hand_built_chain, true, Shape{4}}),
    [](const ::testing::TestParamInfo<OutputShapeRow>& info) { return info.param.name; });

/** @brief Tests that an empty stack stops the process on an output shape. */
TEST(Medium_LayerStackDeathTest, AnEmptyStackStopsTheProcessOnAnOutputShape)
{
    const LayerStack stack;
    SA_EXPECT_FATAL((void)stack.output_shape(), "no layers", "output shape");
}

/** @brief Tests that an empty stack stops the process on an output shape in the layer form. */
TEST(Medium_LayerStackDeathTest, AnEmptyStackStopsTheProcessOnAnOutputShapeOfAnInput)
{
    const LayerStack stack;
    SA_EXPECT_FATAL((void)stack.output_shape(Shape{6}), "no layers", "output shape");
}

/** @brief Tests that build_layer_stack hands each layer's output shape to the next factory. */
TEST(Small_LayerStack, BuildLayerStackThreadsEachOutputShapeIntoTheNextLayer)
{
    SA_ASSERT_OK_AND_ASSIGN(const LayerStack stack,
                            NN::build_layer_stack(make_layer_registry(), hand_built_chain()));

    ASSERT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.layer(0).output_shape(), (Shape{8}));
    EXPECT_EQ(stack.layer(1).output_shape(), (Shape{8}));
    EXPECT_EQ(stack.output_shape(), (Shape{4}));

    const auto& last = dynamic_cast<const LayerAdapter<Linear>&>(stack.layer(2));
    EXPECT_EQ(last.layer().in_features, 8);
}

/** @brief Tests that a hand-built spec seeds the chain from its input shape. */
TEST(Small_LayerStack, BuildLayerStackSeedsTheChainFromTheInputShape)
{
    Config::ModelSpec model = hand_built_chain();
    model.input_shape = Shape{5};

    SA_ASSERT_OK_AND_ASSIGN(const LayerStack stack,
                            NN::build_layer_stack(make_layer_registry(), model));

    const auto& first = dynamic_cast<const LayerAdapter<Linear>&>(stack.layer(0));
    EXPECT_EQ(first.layer().in_features, 5);
}

/** @brief Tests that a hand-built row that names no source is refused with its index and type. */
TEST(Small_LayerStack, BuildLayerStackRefusesAHandBuiltRowWithoutASource)
{
    Config::ModelSpec model = hand_built_chain();
    model.layers[1].from.clear();

    SA_EXPECT_FAILURE(NN::build_layer_stack(make_layer_registry(), model),
                      StatusCode::InvalidDescription, "layer 1 'gelu'", "the row before it");
}

/** @brief Tests that a factory hands a rank-3 per-sample shape to its layer with no axis merged. */
TEST(Small_LayerStack, AnActivationFactoryKeepsARankThreeInputShape)
{
    const Config::Value attributes = attributes_from("{}");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Layer> layer,
                            NN::make_gelu(Shape{4, 4, 3}, attributes, "layer 1 'gelu'"));

    EXPECT_EQ(layer->output_shape(), (Shape{4, 4, 3}));
    EXPECT_EQ(layer->output_features(), 48);
}

/** @brief Tests that a linear layer after a rank-3 input is refused and names flatten. */
TEST(Small_LayerStack, ALinearFactoryRefusesARankThreeInputAndNamesFlatten)
{
    const Config::Value attributes = attributes_from(R"({"units": 8})");

    SA_EXPECT_FAILURE(NN::make_linear(Shape{4, 4, 3}, attributes, "layer 1 'linear'"),
                      StatusCode::InvalidDescription, "layer 1 'linear'", "[4, 4, 3]", "flatten");
}

/** @brief Tests that a linear layer after a rank-1 input is accepted. */
TEST(Small_LayerStack, ALinearFactoryAcceptsARankOneInput)
{
    const Config::Value attributes = attributes_from(R"({"units": 8})");

    SA_EXPECT_OK(NN::make_linear(Shape{48}, attributes, "layer 1 'linear'"));
}

/** @brief Tests that build_layer_stack chains each layer's width from the input features. */
TEST(Small_LayerStack, ChainsLayerWidthsFromTheInputFeatures)
{
    SA_ASSERT_OK_AND_ASSIGN(
        SushiAI::NN::LayerStack stack,
        SushiAI::NN::build_layer_stack(make_layer_registry(), model_from(K_GOOD_MODEL)));

    EXPECT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.output_features(), 4);

    // An activation keeps the width it was handed; see tests/layers/nn/README.md.
    EXPECT_EQ(stack.layer(0).output_features(), 8);
    EXPECT_EQ(stack.layer(1).output_features(), 8);
}

/** @brief Tests that an unknown layer type is refused and the message lists the known types. */
TEST(Small_LayerStack, AnUnknownLayerTypeListsEveryTypeThisBuildHas)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "hologram", "units": 8}]
    })");

    SA_EXPECT_FAILURE(SushiAI::NN::build_layer_stack(make_layer_registry(), model),
                      StatusCode::InvalidDescription, "hologram", "layer type", "'linear'",
                      "'gelu'");
}

/** @brief Tests that a misspelled key on linear is refused by the layer's own factory. */
TEST(Small_LayerStack, AMisspelledKeyOnLinearIsRejectedByItsFactory)
{
    const SushiAI::Config::ModelSpec typo = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "unitss": 8}]
    })");

    SA_EXPECT_FAILURE(SushiAI::NN::build_layer_stack(make_layer_registry(), typo),
                      StatusCode::InvalidDescription, "layer 0 'linear'", "'unitss'");
}

/** @brief Tests that a stray key on gelu is refused by the layer's own factory. */
TEST(Small_LayerStack, AStrayKeyOnGeluIsRejectedByItsFactory)
{
    // An activation accepts no attribute; see tests/layers/nn/README.md.
    const SushiAI::Config::ModelSpec stray = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "gelu", "units": 8}]
    })");

    SA_EXPECT_FAILURE(SushiAI::NN::build_layer_stack(make_layer_registry(), stray),
                      StatusCode::InvalidDescription, "layer 0 'gelu'", "'units'");
}

/** @brief Tests that a layer factory refuses a non-positive width and names the key. */
TEST(Small_LayerStack, ALayerFactoryRefusesANonPositiveWidth)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 0}]
    })");
    SA_EXPECT_FAILURE(SushiAI::NN::build_layer_stack(make_layer_registry(), model),
                      StatusCode::InvalidDescription, "layer 0 'linear'", "'units'");
}

/** @brief Tests that use_bias defaults to true and false removes the bias. */
TEST(Small_LayerStack, UseBiasDefaultsToTrueAndCanBeTurnedOff)
{
    const SushiAI::NN::NodeRegistry registry = make_layer_registry();
    SA_ASSERT_OK_AND_ASSIGN(SushiAI::NN::LayerStack with_bias,
                            SushiAI::NN::build_layer_stack(registry, model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "linear", "units": 4}]
    })")));
    SA_ASSERT_OK_AND_ASSIGN(SushiAI::NN::LayerStack without_bias,
                            SushiAI::NN::build_layer_stack(registry, model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "linear", "units": 4, "use_bias": false}]
    })")));

    SushiAI::Graph::Graph a;
    SushiAI::Graph::Graph b;
    const SushiAI::Graph::GraphBuilder builder_a(a);
    const SushiAI::Graph::GraphBuilder builder_b(b);
    SA_EXPECT_OK(with_bias.forward(builder_a, builder_a.input(SushiAI::Shape{2, 6}, "x")));
    SA_EXPECT_OK(without_bias.forward(builder_b, builder_b.input(SushiAI::Shape{2, 6}, "x")));

    EXPECT_EQ(SushiAI::NN::parameter_count(with_bias), 2u);
    EXPECT_EQ(SushiAI::NN::parameter_count(without_bias), 1u);
}
