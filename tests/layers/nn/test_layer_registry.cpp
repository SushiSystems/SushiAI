/**************************************************************************/
/* test_layer_registry.cpp                                                */
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
 * @file test_layer_registry.cpp
 * @brief Tests the layer registry: each row's factory, the keys it reads and the model it builds.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/nn/avg_pool2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/global_avg_pool.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/nn/unflatten.hpp>
#include <SushiAI/nn/upsample2d.hpp>

using namespace SushiAI;

namespace
{
    using Test::attributes_from;
    using Test::make_layer_registry;
    using Test::model_from;

    /** @brief Holds the unflatten, conv, pool, conv, pool, linear classifier as a model file. */
    constexpr std::string_view K_CONV_MODEL = R"({
        "sushiai_model": 1,
        "input": {"features": 784},
        "layers": [
            {"type": "unflatten", "height": 28, "width": 28, "channels": 1},
            {"type": "conv2d", "filters": 8, "kernel": 3, "padding": 1},
            {"type": "relu"},
            {"type": "maxpool2d", "kernel": 2},
            {"type": "conv2d", "filters": 16, "kernel": 3, "padding": 1},
            {"type": "relu"},
            {"type": "maxpool2d", "kernel": 2},
            {"type": "flatten"},
            {"type": "linear", "units": 10}
        ]
    })";

    /** @brief Lists the parameters K_CONV_MODEL declares, as "name:extents". */
    const std::vector<std::string> K_CONV_MODEL_PARAMETERS = {
        "1.weight:[8, 3, 3, 1]",  "1.bias:[8]",
        "4.weight:[16, 3, 3, 8]", "4.bias:[16]",
        "8.weight:[10, 784]",     "8.bias:[10]",
    };

    /** @brief Describes one registry row's minimal layer object and the shapes around it. */
    struct MinimalCase
    {
        std::string_view type;       /**< @brief The registry key. */
        std::string_view attributes; /**< @brief The layer object without its "type". */
        Shape input;                 /**< @brief The per-sample shape entering the layer. */
        Shape output;                /**< @brief The per-sample shape the layer must report. */
    };

    /** @brief Lists a minimal object for every row of the registry. */
    const std::vector<MinimalCase> K_MINIMAL_CASES = {
        {"linear", R"({"units": 5})", Shape{6}, Shape{5}},
        {"gelu", "{}", Shape{6}, Shape{6}},
        {"relu", "{}", Shape{6}, Shape{6}},
        {"tanh", "{}", Shape{6}, Shape{6}},
        {"sigmoid", "{}", Shape{6}, Shape{6}},
        {"leaky_relu", "{}", Shape{6}, Shape{6}},
        {"unflatten", R"({"height": 2, "width": 3, "channels": 4})", Shape{24}, Shape{2, 3, 4}},
        {"conv2d", R"({"filters": 5, "kernel": 3})", Shape{8, 8, 2}, Shape{6, 6, 5}},
        {"maxpool2d", R"({"kernel": 2})", Shape{8, 8, 2}, Shape{4, 4, 2}},
        {"avgpool2d", R"({"kernel": 2})", Shape{8, 8, 2}, Shape{4, 4, 2}},
        {"global_avgpool", "{}", Shape{8, 8, 2}, Shape{2}},
        {"flatten", "{}", Shape{8, 8, 2}, Shape{128}},
        {"upsample", "{}", Shape{8, 8, 2}, Shape{16, 16, 2}},
        {"silu", "{}", Shape{8, 8, 2}, Shape{8, 8, 2}},
        {"mish", "{}", Shape{8, 8, 2}, Shape{8, 8, 2}},
        {"slice_channels", R"({"offset": 0, "count": 1})", Shape{8, 8, 2}, Shape{8, 8, 1}},
    };

    /** @brief Holds the 8 by 8 by 2 image most refusal rows enter the layer with. */
    const Shape K_IMAGE{8, 8, 2};

    /** @brief Returns the context a layer built from the row @p type names, "layer 0 '<type>'". */
    std::string layer_context(std::string_view type)
    {
        return "layer 0 '" + std::string(type) + "'";
    }

    /** @brief Builds the layer @p type names from @p attributes, as "layer 0 '<type>'". */
    Result<std::unique_ptr<NN::Layer>> try_make_layer(std::string_view type,
                                                      std::string_view attributes,
                                                      const Shape& input)
    {
        const NN::NodeRegistry registry = make_layer_registry();
        const NN::LayerEntry* entry = registry.find_layer(type);
        SA_FATAL_IF(entry == nullptr, "no layer row holds '{}'", type);

        return entry->make(input, attributes_from(attributes), layer_context(type));
    }

    /** @brief Describes a layer object a factory refuses and the words its message names. */
    struct RefusedLayerRow
    {
        std::string name;       /**< The row's name. */
        std::string_view type;  /**< The registry key. */
        std::string_view attributes; /**< The layer object without its "type". */
        Shape input;            /**< The per-sample shape entering the layer. */
        std::string subject;    /**< The context the message names; empty means "layer 0 '<type>'". */
        std::string first;      /**< The first substring the message names after the subject. */
        std::string second;     /**< The second substring the message names. */
    };

    /**
     * @brief Builds the layer @p type names from @p attributes.
     * @pre The layer is accepted; a test of a refusal calls try_make_layer().
     */
    std::unique_ptr<NN::Layer> make_layer(std::string_view type, std::string_view attributes,
                                          const Shape& input)
    {
        Result<std::unique_ptr<NN::Layer>> made = try_make_layer(type, attributes, input);
        SA_FATAL_IF(!made.has_value(), "the test's layer '{}' is refused: {}", type,
                    made.status().message());

        // The layer is held, so value() succeeds.
        return std::move(made).value();
    }

    /** @brief Returns the layer of type @p L that @p layer wraps, or null when it wraps another. */
    template<typename L>
    const L* wrapped(const std::unique_ptr<NN::Layer>& layer)
    {
        const auto* adapter = dynamic_cast<const NN::LayerAdapter<L>*>(layer.get());
        if (adapter == nullptr)
            return nullptr;

        return &adapter->layer();
    }

    /** @brief Builds a square window of @p kernel taps with one stride and pad on both axes. */
    Graph::ConvGeometry square_window(int64_t kernel, int64_t stride, int64_t padding)
    {
        Graph::ConvGeometry geometry;
        geometry.kernel_h = kernel;
        geometry.kernel_w = kernel;
        geometry.stride_h = stride;
        geometry.stride_w = stride;
        geometry.pad_top = padding;
        geometry.pad_left = padding;
        return geometry;
    }

    /** @brief Builds a Conv2d whose parameters are still undeclared. */
    NN::Conv2d make_conv(int64_t in_channels, int64_t out_channels, Graph::ConvGeometry geometry)
    {
        NN::Conv2d layer{};
        layer.in_channels = in_channels;
        layer.out_channels = out_channels;
        layer.geometry = geometry;
        return layer;
    }

    /**
     * @brief Builds the layer stack the model file @p text describes.
     * @pre The file is accepted; a test of a refusal calls build_layer_stack() itself.
     */
    NN::LayerStack build_stack(std::string_view text)
    {
        Result<NN::LayerStack> built =
            NN::build_layer_stack(make_layer_registry(), model_from(text));
        SA_FATAL_IF(!built.has_value(), "the test's model file is refused: {}",
                    built.status().message());

        // The stack is held, so value() succeeds.
        return std::move(built).value();
    }

    /** @brief Builds the stack K_CONV_MODEL describes. */
    NN::LayerStack make_conv_stack()
    {
        return build_stack(K_CONV_MODEL);
    }

    /** @brief Traces @p model over a [4, 784] input and lists its parameters as "name:extents". */
    template<typename M>
    std::vector<std::string> traced_names(M& model)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{4, 784}, "inputs");
        SA_EXPECT_OK(model.forward(builder, x));

        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });
        return names;
    }
}

/** @brief Tests that the registry spells its sixteen type strings as model files do, in order. */
TEST(Small_LayerRegistry, TheTypeStringsAreTheOnesModelFilesSpell)
{
    const std::vector<std::string_view> expected = {
        "linear",     "gelu",      "relu",   "tanh",      "sigmoid",
        "leaky_relu", "unflatten", "conv2d", "maxpool2d", "avgpool2d",
        "global_avgpool", "flatten", "upsample", "silu", "mish",
        "slice_channels",
    };

    const NN::NodeRegistry registry = make_layer_registry();
    std::vector<std::string_view> spelled;
    for (const NN::LayerEntry& entry : registry.layers())
        spelled.push_back(entry.type);

    EXPECT_EQ(spelled, expected);
}

/** @brief Runs a case over each row of the registry's minimal objects. */
class Small_LayerRegistryMinimal : public ::testing::TestWithParam<MinimalCase>
{
};

/** @brief Tests that the factory of the row builds a layer from its minimal object. */
TEST_P(Small_LayerRegistryMinimal, TheFactoryBuildsFromAMinimalObject)
{
    const MinimalCase& minimal = GetParam();

    const std::unique_ptr<NN::Layer> layer =
        make_layer(minimal.type, minimal.attributes, minimal.input);

    ASSERT_NE(layer, nullptr);
    EXPECT_EQ(layer->output_shape(), minimal.output);
}

/** @brief Tests that the factory of the row refuses a key it does not know and names that key. */
TEST_P(Small_LayerRegistryMinimal, TheFactoryRefusesAnUnknownKey)
{
    const MinimalCase& minimal = GetParam();

    SA_EXPECT_FAILURE(try_make_layer(minimal.type, R"({"no_such_key": 1})", minimal.input),
                      StatusCode::InvalidDescription, layer_context(minimal.type),
                      "'no_such_key'");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerRegistryMinimal, ::testing::ValuesIn(K_MINIMAL_CASES),
    [](const ::testing::TestParamInfo<MinimalCase>& info) { return std::string(info.param.type); });

namespace
{
    /** @brief Describes a key whose factory builds one layer type, and how to tell it. */
    struct OwnTypeRow
    {
        std::string name;       /**< The row's name. */
        std::string_view type;  /**< The registry key. */
        std::string_view attributes; /**< The layer object without its "type". */
        Shape input;            /**< The per-sample shape entering the layer. */
        bool (*is_own_type)(const std::unique_ptr<NN::Layer>&); /**< Whether the layer is the row's. */
    };

    /** @brief Returns whether @p layer wraps a layer of type @p L. */
    template<typename L>
    bool holds_layer(const std::unique_ptr<NN::Layer>& layer)
    {
        return wrapped<L>(layer) != nullptr;
    }
}

/** @brief Runs a case over each key of the registry that has a layer type of its own. */
class Small_LayerRegistryOwnType : public ::testing::TestWithParam<OwnTypeRow>
{
};

/** @brief Tests that the key of the row builds the layer type it names. */
TEST_P(Small_LayerRegistryOwnType, TheKeyBuildsItsOwnLayerType)
{
    const OwnTypeRow& row = GetParam();

    const std::unique_ptr<NN::Layer> layer = make_layer(row.type, row.attributes, row.input);

    EXPECT_TRUE(row.is_own_type(layer));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerRegistryOwnType,
    ::testing::Values(
        OwnTypeRow{"Linear", "linear", R"({"units": 5})", Shape{6}, holds_layer<NN::Linear>},
        OwnTypeRow{"GELU", "gelu", "{}", Shape{6}, holds_layer<NN::GELU>},
        OwnTypeRow{"ReLU", "relu", "{}", Shape{6}, holds_layer<NN::ReLU>},
        OwnTypeRow{"Tanh", "tanh", "{}", Shape{6}, holds_layer<NN::Tanh>},
        OwnTypeRow{"Sigmoid", "sigmoid", "{}", Shape{6}, holds_layer<NN::Sigmoid>},
        OwnTypeRow{"LeakyReLU", "leaky_relu", "{}", Shape{6}, holds_layer<NN::LeakyReLU>},
        OwnTypeRow{"Unflatten", "unflatten", R"({"height": 1, "width": 2, "channels": 3})",
                   Shape{6}, holds_layer<NN::Unflatten>},
        OwnTypeRow{"Conv2d", "conv2d", R"({"filters": 5, "kernel": 3})", Shape{8, 8, 2},
                   holds_layer<NN::Conv2d>},
        OwnTypeRow{"MaxPool2d", "maxpool2d", R"({"kernel": 2})", Shape{8, 8, 2},
                   holds_layer<NN::MaxPool2d>},
        OwnTypeRow{"AvgPool2d", "avgpool2d", R"({"kernel": 2})", Shape{8, 8, 2},
                   holds_layer<NN::AvgPool2d>},
        OwnTypeRow{"GlobalAvgPool", "global_avgpool", "{}", Shape{8, 8, 2},
                   holds_layer<NN::GlobalAvgPool>},
        OwnTypeRow{"Flatten", "flatten", "{}", Shape{8, 8, 2}, holds_layer<NN::Flatten>},
        OwnTypeRow{"Upsample2d", "upsample", "{}", Shape{8, 8, 2}, holds_layer<NN::Upsample2d>},
        OwnTypeRow{"SiLU", "silu", "{}", Shape{8, 8, 2}, holds_layer<NN::SiLU>}),
    [](const ::testing::TestParamInfo<OwnTypeRow>& info) { return info.param.name; });

/** @brief Tests that a convolution takes its input channel count from the input shape. */
TEST(Small_LayerRegistry, Conv2dTakesItsInputChannelsFromTheInputShape)
{
    const std::unique_ptr<NN::Layer> layer =
        make_layer("conv2d", R"({"filters": 5, "kernel": 3})", Shape{8, 8, 7});
    const NN::Conv2d* conv = wrapped<NN::Conv2d>(layer);

    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->in_channels, 7);
    EXPECT_EQ(conv->out_channels, 5);
}

/** @brief Tests that a model file cannot set a convolution's input channel count. */
TEST(Small_LayerRegistry, Conv2dRefusesTheInChannelsKey)
{
    SA_EXPECT_FAILURE(try_make_layer("conv2d",
                                     R"({"filters": 5, "kernel": 3, "in_channels": 7})",
                                     Shape{8, 8, 7}),
                      StatusCode::InvalidDescription, "layer 0 'conv2d'", "'in_channels'");
}

/** @brief Tests that a convolution fed a feature vector is refused with the layer that fixes it. */
TEST(Small_LayerRegistry, Conv2dRefusesAnInputThatIsNotAnImage)
{
    SA_EXPECT_FAILURE(try_make_layer("conv2d", R"({"filters": 5, "kernel": 3})", Shape{784}),
                      StatusCode::InvalidDescription, "layer 0 'conv2d'", "[784]", "[H, W, C]",
                      "'unflatten'");
}

/** @brief Tests that a convolution defaults to stride 1, no padding, dilation 1 and a bias. */
TEST(Small_LayerRegistry, Conv2dDefaultsItsOptionalKeys)
{
    const std::unique_ptr<NN::Layer> layer =
        make_layer("conv2d", R"({"filters": 5, "kernel": 3})", Shape{8, 8, 2});
    const NN::Conv2d* conv = wrapped<NN::Conv2d>(layer);

    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->geometry.stride_h, 1);
    EXPECT_EQ(conv->geometry.stride_w, 1);
    EXPECT_EQ(conv->geometry.pad_top, 0);
    EXPECT_EQ(conv->geometry.pad_left, 0);
    EXPECT_EQ(conv->geometry.dilation_h, 1);
    EXPECT_EQ(conv->geometry.dilation_w, 1);
    EXPECT_TRUE(conv->use_bias);
}

/** @brief Tests that a convolution stores every key the model file sets. */
TEST(Small_LayerRegistry, Conv2dReadsEveryKeyItAccepts)
{
    const std::unique_ptr<NN::Layer> layer = make_layer(
        "conv2d",
        R"({"filters": 5, "kernel": [3, 5], "stride": [2, 1], "padding": [1, 2],
            "dilation": [1, 2], "use_bias": false})",
        Shape{16, 16, 2});
    const NN::Conv2d* conv = wrapped<NN::Conv2d>(layer);

    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->geometry.kernel_h, 3);
    EXPECT_EQ(conv->geometry.kernel_w, 5);
    EXPECT_EQ(conv->geometry.stride_h, 2);
    EXPECT_EQ(conv->geometry.stride_w, 1);
    EXPECT_EQ(conv->geometry.pad_top, 1);
    EXPECT_EQ(conv->geometry.pad_left, 2);
    EXPECT_EQ(conv->geometry.dilation_h, 1);
    EXPECT_EQ(conv->geometry.dilation_w, 2);
    EXPECT_FALSE(conv->use_bias);
    EXPECT_EQ(layer->output_shape(), (Shape{8, 12, 5}));
}

namespace
{
    /** @brief Describes one spelling of a 3 by 3 window and the layer it yields. */
    struct KernelSpellingRow
    {
        std::string name;            /**< The row's name. */
        std::string_view type;       /**< The registry key. */
        std::string_view attributes; /**< The layer object, with the kernel spelled. */
        int64_t stride;              /**< The stride on both axes the layer must hold. */
        Shape output;                /**< The shape the layer reports for an 8 by 8 by 2 image. */
        Graph::ConvGeometry (*geometry_of)(const std::unique_ptr<NN::Layer>&); /**< Reads the window. */
    };

    /** @brief Returns the window of the layer of type @p L that @p layer wraps. */
    template<typename L>
    Graph::ConvGeometry geometry_of(const std::unique_ptr<NN::Layer>& layer)
    {
        const L* inner = wrapped<L>(layer);
        SA_FATAL_IF(inner == nullptr, "the test's layer wraps another type");

        // The layer is held, so the dereference is safe.
        return inner->geometry;
    }
}

/** @brief Runs a case over each spelling of a 3 by 3 kernel, as an integer and as a pair. */
class Small_LayerRegistryKernelSpelling : public ::testing::TestWithParam<KernelSpellingRow>
{
};

/** @brief Tests that the spelling of the row builds a 3 by 3 window with the stride and shape. */
TEST_P(Small_LayerRegistryKernelSpelling, TheSpellingBuildsTheWindow)
{
    const KernelSpellingRow& row = GetParam();

    const std::unique_ptr<NN::Layer> layer = make_layer(row.type, row.attributes, Shape{8, 8, 2});
    const Graph::ConvGeometry window = row.geometry_of(layer);

    EXPECT_EQ(window.kernel_h, 3);
    EXPECT_EQ(window.kernel_w, 3);
    EXPECT_EQ(window.stride_h, row.stride);
    EXPECT_EQ(window.stride_w, row.stride);
    EXPECT_EQ(layer->output_shape(), row.output);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerRegistryKernelSpelling,
    ::testing::Values(
        KernelSpellingRow{"Conv2dFromAnInteger", "conv2d", R"({"filters": 5, "kernel": 3})", 1,
                          Shape{6, 6, 5}, geometry_of<NN::Conv2d>},
        KernelSpellingRow{"Conv2dFromAPair", "conv2d", R"({"filters": 5, "kernel": [3, 3]})", 1,
                          Shape{6, 6, 5}, geometry_of<NN::Conv2d>},
        KernelSpellingRow{"MaxPool2dFromAnInteger", "maxpool2d", R"({"kernel": 3})", 3,
                          Shape{2, 2, 2}, geometry_of<NN::MaxPool2d>},
        KernelSpellingRow{"MaxPool2dFromAPair", "maxpool2d", R"({"kernel": [3, 3]})", 3,
                          Shape{2, 2, 2}, geometry_of<NN::MaxPool2d>},
        KernelSpellingRow{"AvgPool2dFromAnInteger", "avgpool2d", R"({"kernel": 3})", 3,
                          Shape{2, 2, 2}, geometry_of<NN::AvgPool2d>},
        KernelSpellingRow{"AvgPool2dFromAPair", "avgpool2d", R"({"kernel": [3, 3]})", 3,
                          Shape{2, 2, 2}, geometry_of<NN::AvgPool2d>}),
    [](const ::testing::TestParamInfo<KernelSpellingRow>& info) { return info.param.name; });

/** @brief Runs a case over each layer object a factory refuses, with the words it names. */
class Small_LayerRegistryRefusal : public ::testing::TestWithParam<RefusedLayerRow>
{
};

/** @brief Tests that the factory of the row refuses its object, naming the layer and the reason. */
TEST_P(Small_LayerRegistryRefusal, TheLayerIsRefused)
{
    const RefusedLayerRow& row = GetParam();
    const std::string subject = row.subject.empty() ? layer_context(row.type) : row.subject;

    SA_EXPECT_FAILURE(try_make_layer(row.type, row.attributes, row.input),
                      StatusCode::InvalidDescription, subject, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    AWindowSettingOfTheWrongForm, Small_LayerRegistryRefusal,
    ::testing::Values(
        RefusedLayerRow{"AOneEntryPair", "maxpool2d", R"({"kernel": [2]})", K_IMAGE,
                        "layer 0 'maxpool2d' 'kernel'", "two-entry [h, w]", "1 entries"},
        RefusedLayerRow{"AThreeEntryPair", "maxpool2d", R"({"kernel": [2, 2, 2]})", K_IMAGE,
                        "layer 0 'maxpool2d' 'kernel'", "two-entry [h, w]", "3 entries"},
        RefusedLayerRow{"AStringKernel", "avgpool2d", R"({"kernel": "2"})", K_IMAGE,
                        "layer 0 'avgpool2d' 'kernel'", "two-entry [h, w]", ""},
        RefusedLayerRow{"AFractionalStride", "conv2d",
                        R"({"filters": 5, "kernel": 3, "stride": 1.5})", K_IMAGE,
                        "layer 0 'conv2d' 'stride'", "two-entry [h, w]", ""}),
    [](const ::testing::TestParamInfo<RefusedLayerRow>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    Conv2dNonPositiveExtent, Small_LayerRegistryRefusal,
    ::testing::Values(
        RefusedLayerRow{"ZeroFilters", "conv2d", R"({"filters": 0, "kernel": 3})", K_IMAGE, "",
                        "Conv2d", "positive"},
        RefusedLayerRow{"NegativeFilters", "conv2d", R"({"filters": -4, "kernel": 3})", K_IMAGE,
                        "", "Conv2d", "positive"},
        RefusedLayerRow{"ZeroKernel", "conv2d", R"({"filters": 5, "kernel": 0})", K_IMAGE, "",
                        "Conv2d", "positive"},
        RefusedLayerRow{"ZeroKernelOnOneAxis", "conv2d", R"({"filters": 5, "kernel": [3, 0]})",
                        K_IMAGE, "", "Conv2d", "positive"},
        RefusedLayerRow{"NegativeKernel", "conv2d", R"({"filters": 5, "kernel": -3})", K_IMAGE, "",
                        "Conv2d", "positive"},
        RefusedLayerRow{"ZeroStride", "conv2d", R"({"filters": 5, "kernel": 3, "stride": 0})",
                        K_IMAGE, "", "Conv2d", "positive"},
        RefusedLayerRow{"NegativeStrideOnOneAxis", "conv2d",
                        R"({"filters": 5, "kernel": 3, "stride": [1, -1]})", K_IMAGE, "", "Conv2d",
                        "positive"},
        RefusedLayerRow{"ZeroDilation", "conv2d", R"({"filters": 5, "kernel": 3, "dilation": 0})",
                        K_IMAGE, "", "Conv2d", "positive"}),
    [](const ::testing::TestParamInfo<RefusedLayerRow>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    PoolNonPositiveExtent, Small_LayerRegistryRefusal,
    ::testing::Values(
        RefusedLayerRow{"MaxPoolZeroKernel", "maxpool2d", R"({"kernel": 0})", K_IMAGE, "",
                        "positive", ""},
        RefusedLayerRow{"MaxPoolZeroKernelOnOneAxis", "maxpool2d", R"({"kernel": [2, 0]})",
                        K_IMAGE, "", "positive", ""},
        RefusedLayerRow{"MaxPoolNegativeKernel", "maxpool2d", R"({"kernel": -2})", K_IMAGE, "",
                        "positive", ""},
        RefusedLayerRow{"MaxPoolZeroStride", "maxpool2d", R"({"kernel": 2, "stride": 0})",
                        K_IMAGE, "", "positive", ""},
        RefusedLayerRow{"MaxPoolNegativeStrideOnOneAxis", "maxpool2d",
                        R"({"kernel": 2, "stride": [-1, 2]})", K_IMAGE, "", "positive", ""},
        RefusedLayerRow{"AvgPoolZeroKernel", "avgpool2d", R"({"kernel": 0})", K_IMAGE, "",
                        "positive", ""},
        RefusedLayerRow{"AvgPoolZeroKernelOnOneAxis", "avgpool2d", R"({"kernel": [2, 0]})",
                        K_IMAGE, "", "positive", ""},
        RefusedLayerRow{"AvgPoolNegativeKernel", "avgpool2d", R"({"kernel": -2})", K_IMAGE, "",
                        "positive", ""},
        RefusedLayerRow{"AvgPoolZeroStride", "avgpool2d", R"({"kernel": 2, "stride": 0})",
                        K_IMAGE, "", "positive", ""},
        RefusedLayerRow{"AvgPoolNegativeStrideOnOneAxis", "avgpool2d",
                        R"({"kernel": 2, "stride": [-1, 2]})", K_IMAGE, "", "positive", ""}),
    [](const ::testing::TestParamInfo<RefusedLayerRow>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    NegativePadding, Small_LayerRegistryRefusal,
    ::testing::Values(
        RefusedLayerRow{"Conv2dOnBothAxes", "conv2d",
                        R"({"filters": 5, "kernel": 3, "padding": -1})", K_IMAGE, "", "padding",
                        ""},
        RefusedLayerRow{"Conv2dOnTheWidth", "conv2d",
                        R"({"filters": 5, "kernel": 3, "padding": [0, -1]})", K_IMAGE, "",
                        "padding", ""},
        RefusedLayerRow{"Conv2dOnTheHeight", "conv2d",
                        R"({"filters": 5, "kernel": 3, "padding": [-1, 0]})", K_IMAGE, "",
                        "padding", ""},
        RefusedLayerRow{"MaxPoolOnBothAxes", "maxpool2d", R"({"kernel": 3, "padding": -1})",
                        K_IMAGE, "", "padding", ""},
        RefusedLayerRow{"MaxPoolOnTheWidth", "maxpool2d", R"({"kernel": 3, "padding": [0, -1]})",
                        K_IMAGE, "", "padding", ""},
        RefusedLayerRow{"MaxPoolOnTheHeight", "maxpool2d", R"({"kernel": 3, "padding": [-1, 0]})",
                        K_IMAGE, "", "padding", ""},
        RefusedLayerRow{"AvgPoolOnBothAxes", "avgpool2d", R"({"kernel": 3, "padding": -1})",
                        K_IMAGE, "", "padding", ""},
        RefusedLayerRow{"AvgPoolOnTheWidth", "avgpool2d", R"({"kernel": 3, "padding": [0, -1]})",
                        K_IMAGE, "", "padding", ""},
        RefusedLayerRow{"AvgPoolOnTheHeight", "avgpool2d", R"({"kernel": 3, "padding": [-1, 0]})",
                        K_IMAGE, "", "padding", ""}),
    [](const ::testing::TestParamInfo<RefusedLayerRow>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    AMissingRequiredKey, Small_LayerRegistryRefusal,
    ::testing::Values(
        RefusedLayerRow{"Conv2dWithoutFilters", "conv2d", R"({"kernel": 3})", K_IMAGE, "",
                        "has no 'filters'", ""},
        RefusedLayerRow{"Conv2dWithoutKernel", "conv2d", R"({"filters": 5})", K_IMAGE, "",
                        "has no 'kernel'", ""},
        RefusedLayerRow{"MaxPoolWithoutKernel", "maxpool2d", R"({"stride": 2})", K_IMAGE, "",
                        "has no 'kernel'", ""},
        RefusedLayerRow{"AvgPoolWithoutKernel", "avgpool2d", R"({"stride": 2})", K_IMAGE, "",
                        "has no 'kernel'", ""},
        RefusedLayerRow{"LinearWithoutUnits", "linear", R"({"use_bias": false})", Shape{6}, "",
                        "has no 'units'", ""}),
    [](const ::testing::TestParamInfo<RefusedLayerRow>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    ALeakyReLUSlopeOutsideItsRange, Small_LayerRegistryRefusal,
    ::testing::Values(
        RefusedLayerRow{"Zero", "leaky_relu", R"({"slope": 0})", Shape{6}, "", "slope", ""},
        RefusedLayerRow{"One", "leaky_relu", R"({"slope": 1})", Shape{6}, "", "slope", ""},
        RefusedLayerRow{"Negative", "leaky_relu", R"({"slope": -0.1})", Shape{6}, "", "slope", ""}),
    [](const ::testing::TestParamInfo<RefusedLayerRow>& info) { return info.param.name; });

/** @brief Tests that a pool's stride follows its kernel unless the model file sets one. */
TEST(Small_LayerRegistry, APoolStrideDefaultsToItsKernel)
{
    const std::unique_ptr<NN::Layer> defaulted =
        make_layer("maxpool2d", R"({"kernel": [2, 3]})", Shape{12, 12, 2});
    const NN::MaxPool2d* max_pool = wrapped<NN::MaxPool2d>(defaulted);

    ASSERT_NE(max_pool, nullptr);
    EXPECT_EQ(max_pool->geometry.stride_h, 2);
    EXPECT_EQ(max_pool->geometry.stride_w, 3);
    EXPECT_EQ(max_pool->geometry.pad_top, 0);
    EXPECT_EQ(max_pool->geometry.pad_left, 0);
    EXPECT_EQ(defaulted->output_shape(), (Shape{6, 4, 2}));
}

/** @brief Tests that a pool reads the stride and the padding the model file sets. */
TEST(Small_LayerRegistry, APoolReadsAnExplicitStrideAndPadding)
{
    const std::unique_ptr<NN::Layer> spelled =
        make_layer("avgpool2d", R"({"kernel": 3, "stride": 1, "padding": 1})", Shape{12, 12, 2});
    const NN::AvgPool2d* avg_pool = wrapped<NN::AvgPool2d>(spelled);

    ASSERT_NE(avg_pool, nullptr);
    EXPECT_EQ(avg_pool->geometry.stride_h, 1);
    EXPECT_EQ(avg_pool->geometry.stride_w, 1);
    EXPECT_EQ(avg_pool->geometry.pad_top, 1);
    EXPECT_EQ(avg_pool->geometry.pad_left, 1);
    EXPECT_EQ(spelled->output_shape(), (Shape{12, 12, 2}));
}

/** @brief Tests that upsample refuses a scale below one when the layer is built. */
TEST(Small_LayerRegistry, UpsampleRejectsAScaleBelowOne)
{
    SA_EXPECT_FAILURE(try_make_layer("upsample", R"({"scale": 0})", Shape{4, 4, 2}),
                      StatusCode::InvalidDescription, "layer 0 'upsample'", "Upsample2d", "scale");
}

/** @brief Tests that leaky_relu uses 0.01 until the model file sets a slope. */
TEST(Small_LayerRegistry, LeakyReLUReadsItsOptionalSlope)
{
    const std::unique_ptr<NN::Layer> default_leaky = make_layer("leaky_relu", "{}", Shape{6});
    const std::unique_ptr<NN::Layer> steep_leaky =
        make_layer("leaky_relu", R"({"slope": 0.25})", Shape{6});

    ASSERT_NE(wrapped<NN::LeakyReLU>(default_leaky), nullptr);
    ASSERT_NE(wrapped<NN::LeakyReLU>(steep_leaky), nullptr);
    EXPECT_DOUBLE_EQ(wrapped<NN::LeakyReLU>(default_leaky)->slope, 0.01);
    EXPECT_DOUBLE_EQ(wrapped<NN::LeakyReLU>(steep_leaky)->slope, 0.25);
}

/** @brief Tests that upsample uses a scale of 2 until the model file sets one. */
TEST(Small_LayerRegistry, UpsampleReadsItsOptionalScale)
{
    const std::unique_ptr<NN::Layer> default_upsample =
        make_layer("upsample", "{}", Shape{4, 4, 2});
    const std::unique_ptr<NN::Layer> triple_upsample =
        make_layer("upsample", R"({"scale": 3})", Shape{4, 4, 2});

    EXPECT_EQ(default_upsample->output_shape(), (Shape{8, 8, 2}));
    EXPECT_EQ(triple_upsample->output_shape(), (Shape{12, 12, 2}));
}

/** @brief Tests that unflatten refuses an input whose element count its image does not hold. */
TEST(Small_LayerRegistry, UnflattenRefusesAnInputOfAnotherSize)
{
    SA_EXPECT_FAILURE(try_make_layer("unflatten",
                                     R"({"height": 28, "width": 28, "channels": 1})",
                                     Shape{783}),
                      StatusCode::InvalidDescription, "layer 0 'unflatten'", "[783]", "[784]");
}

/** @brief Tests that unflatten refuses an object that names no channel count. */
TEST(Small_LayerRegistry, UnflattenRefusesAMissingChannelCount)
{
    SA_EXPECT_FAILURE(try_make_layer("unflatten", R"({"height": 28, "width": 28})", Shape{784}),
                      StatusCode::InvalidDescription, "layer 0 'unflatten'", "channels");
}

/** @brief Tests that the convolutional model file builds a stack that ends in ten features. */
TEST(Small_LayerRegistry, TheConvModelBuildsAndReportsItsOutputShape)
{
    const NN::LayerStack stack = make_conv_stack();

    EXPECT_EQ(stack.size(), 9u);
    EXPECT_EQ(stack.output_shape(), (Shape{10}));
    EXPECT_EQ(stack.layer(1).output_shape(), (Shape{28, 28, 8}));
    EXPECT_EQ(stack.layer(3).output_shape(), (Shape{14, 14, 8}));
    EXPECT_EQ(stack.layer(6).output_shape(), (Shape{7, 7, 16}));
    EXPECT_EQ(stack.layer(7).output_shape(), (Shape{784}));
}

/** @brief Tests that the convolutional model traces a [4, 784] batch to [4, 10]. */
TEST(Small_LayerRegistry, TheConvModelTracesToTheShapeItReports)
{
    NN::LayerStack stack = make_conv_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 784}, "inputs");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, stack.forward(builder, x));
    EXPECT_EQ(graph.value(y).shape, (Shape{4, 10}));
}

/** @brief Tests that a Sequential of the same layer types yields the model file's names. */
TEST(Small_LayerRegistry, ASequentialOfTheSameLayersYieldsTheSameNames)
{
    NN::Sequential compile_time{
        NN::Unflatten{28, 28, 1},
        make_conv(1, 8, square_window(3, 1, 1)),
        NN::ReLU{},
        NN::MaxPool2d{square_window(2, 2, 0)},
        make_conv(8, 16, square_window(3, 1, 1)),
        NN::ReLU{},
        NN::MaxPool2d{square_window(2, 2, 0)},
        NN::Flatten{},
        NN::Linear{784, 10},
    };
    NN::LayerStack run_time = make_conv_stack();

    const std::vector<std::string> expected = traced_names(compile_time);
    EXPECT_EQ(expected, K_CONV_MODEL_PARAMETERS);
    EXPECT_EQ(traced_names(run_time), expected);
}

namespace
{
    /** @brief Names one parameter of the convolutional model by its position in the walk. */
    struct ParameterRow
    {
        std::string name;    /**< The row's name. */
        std::size_t index;   /**< The position of the parameter in the walk. */
        Shape shape;         /**< The shape its gradient must have. */
    };
}

/** @brief Runs the gradient case over each of the six parameters of the convolutional model. */
class Small_LayerRegistryGradient : public ::testing::TestWithParam<ParameterRow>
{
};

/** @brief Tests that differentiating cross-entropy gives the parameter of the row a gradient. */
TEST_P(Small_LayerRegistryGradient, CrossEntropyDifferentiatesToTheParameter)
{
    const ParameterRow& row = GetParam();
    NN::LayerStack stack = make_conv_stack();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 784}, "inputs");
    const Graph::ValueID targets = builder.input(Shape{4, 10}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID logits, stack.forward(builder, x));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            NN::CrossEntropyLoss{}.forward(builder, logits, targets));
    std::vector<Graph::ValueID> wrt;
    NN::for_each_parameter(stack, [&](const std::string&, NN::Parameter& parameter)
    {
        wrt.push_back(parameter.id_in(graph));
    });
    ASSERT_EQ(wrt.size(), 6u);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Graph::ValueID> gradients, Autograd::differentiate(
        graph, loss, SushiRuntime::span<const Graph::ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(gradients.size(), wrt.size());
    ASSERT_NE(gradients[row.index], Graph::K_NO_VALUE);
    EXPECT_EQ(graph.value(gradients[row.index]).shape, row.shape);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerRegistryGradient,
    ::testing::Values(ParameterRow{"TheFirstConvWeight", 0, Shape{8, 3, 3, 1}},
                      ParameterRow{"TheFirstConvBias", 1, Shape{8}},
                      ParameterRow{"TheSecondConvWeight", 2, Shape{16, 3, 3, 8}},
                      ParameterRow{"TheSecondConvBias", 3, Shape{16}},
                      ParameterRow{"TheLinearWeight", 4, Shape{10, 784}},
                      ParameterRow{"TheLinearBias", 5, Shape{10}}),
    [](const ::testing::TestParamInfo<ParameterRow>& info) { return info.param.name; });

/** @brief Tests that a silu row traces one SILU node that keeps the input's shape. */
TEST(Small_LayerRegistry, ASiLURowTracesOneSiLUNode)
{
    const std::unique_ptr<NN::Layer> layer = make_layer("silu", "{}", Shape{8, 8, 2});

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 8, 8, 2}, "inputs");
    const std::size_t leaves = graph.node_count();
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer->forward(builder, x));

    ASSERT_EQ(graph.node_count(), leaves + 1u);
    EXPECT_EQ(graph.nodes()[leaves].op, Ops::K_SILU.id);
    EXPECT_EQ(graph.value(y).shape, (Shape{4, 8, 8, 2}));
}

/** @brief Tests that a version 2 chain is seeded with input.shape, not with its element count. */
TEST(Small_LayerRegistry, AVersionTwoChainStartsFromTheInputShape)
{
    constexpr std::string_view K_IMAGE_CHAIN = R"({
        "format_version": 2,
        "input": {"shape": [8, 8, 2]},
        "layers": [
            {"type": "conv2d", "filters": 5, "kernel": 3, "from": -1},
            {"type": "silu"},
            {"type": "flatten", "from": 1}
        ]
    })";

    const NN::LayerStack stack = build_stack(K_IMAGE_CHAIN);

    ASSERT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.layer(0).output_shape(), (Shape{6, 6, 5}));
    EXPECT_EQ(stack.output_shape(), (Shape{180}));
}

namespace
{
    /** @brief Describes a model file build_layer_stack refuses and the words it names. */
    struct ChainRefusalRow
    {
        std::string name;        /**< The row's name. */
        std::string_view model;  /**< The model file text. */
        std::string_view first;  /**< The first substring the message names. */
        std::string_view second; /**< The second substring the message names. */
        std::string_view third;  /**< The third substring the message names; empty names none. */
    };
}

/** @brief Runs the chain refusal over each model file a layer stack cannot hold. */
class Small_LayerRegistryChainRefusal : public ::testing::TestWithParam<ChainRefusalRow>
{
};

/** @brief Tests that build_layer_stack refuses the model file of the row. */
TEST_P(Small_LayerRegistryChainRefusal, TheChainIsRefused)
{
    const ChainRefusalRow& row = GetParam();
    const Config::ModelSpec model = model_from(row.model);

    SA_EXPECT_FAILURE(NN::build_layer_stack(make_layer_registry(), model),
                      StatusCode::InvalidDescription, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerRegistryChainRefusal,
    ::testing::Values(
        ChainRefusalRow{"ARowThatReadsTwoRows", R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh"},
            {"type": "relu", "from": [-1, 0]}
        ]
    })", "layer 2 'relu'", "the row before it", "build_model"},
        ChainRefusalRow{"ARowThatSkipsBack", R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh"},
            {"type": "relu", "from": 0}
        ]
    })", "layer 2 'relu'", "the row before it", ""},
        ChainRefusalRow{"ALaterRowThatReadsTheInput", R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh", "from": "input"}
        ]
    })", "layer 1 'tanh'", "the row before it", ""},
        ChainRefusalRow{"OutputsThatPublishAnEarlierRow", R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh"}
        ],
        "outputs": [{"node": 0}]
    })", "'outputs'", "last row", "build_model"},
        ChainRefusalRow{"ARowThatReadsANamedPort", R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh", "from": {"node": 0, "port": "output"}}
        ]
    })", "layer 1 'tanh'", "the row before it", ""}),
    [](const ::testing::TestParamInfo<ChainRefusalRow>& info) { return info.param.name; });

/** @brief Tests that build_layer_stack builds a file whose outputs name the last row whole. */
TEST(Small_LayerRegistry, AChainAcceptsOutputsThatPublishTheLastRow)
{
    constexpr std::string_view K_LAST_OUTPUT_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh"}
        ],
        "outputs": [{"node": 1}]
    })";

    const NN::LayerStack stack = build_stack(K_LAST_OUTPUT_MODEL);

    EXPECT_EQ(stack.size(), 2u);
}
