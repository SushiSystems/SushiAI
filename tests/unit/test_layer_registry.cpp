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
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
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
#include <common/model_text.hpp>
#include <common/test_status.hpp>
#include <common/test_throws.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

namespace
{
    using Test::attributes_from;
    using Test::make_layer_registry;

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

    /** @brief Builds the layer @p type names from @p attributes, as "layer 0 '<type>'". */
    std::unique_ptr<NN::Layer> make_layer(std::string_view type, std::string_view attributes,
                                           const Shape& input)
    {
        const NN::NodeRegistry registry = make_layer_registry();
        const NN::LayerEntry* entry = registry.find_layer(type);
        if (entry == nullptr)
        {
            ADD_FAILURE() << "no layer row holds '" << type << "'";
            return nullptr;
        }
        const std::string context = "layer 0 '" + std::string(type) + "'";
        return entry->make(input, attributes_from(attributes), context);
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

    /** @brief Expects the pool @p type to store one window for kernel 3 and kernel [3, 3]. */
    template<typename Pool>
    void expect_pool_window_spellings_agree(std::string_view type, const Shape& image)
    {
        SCOPED_TRACE(std::string(type));

        const std::unique_ptr<NN::Layer> from_integer =
            make_layer(type, R"({"kernel": 3})", image);
        const std::unique_ptr<NN::Layer> from_pair =
            make_layer(type, R"({"kernel": [3, 3]})", image);
        const Pool* integer_pool = wrapped<Pool>(from_integer);
        const Pool* pair_pool = wrapped<Pool>(from_pair);

        ASSERT_NE(integer_pool, nullptr);
        ASSERT_NE(pair_pool, nullptr);
        EXPECT_EQ(integer_pool->geometry.kernel_h, 3);
        EXPECT_EQ(integer_pool->geometry.kernel_w, 3);
        EXPECT_EQ(integer_pool->geometry.kernel_h, pair_pool->geometry.kernel_h);
        EXPECT_EQ(integer_pool->geometry.kernel_w, pair_pool->geometry.kernel_w);
        EXPECT_EQ(integer_pool->geometry.stride_h, pair_pool->geometry.stride_h);
        EXPECT_EQ(integer_pool->geometry.stride_w, pair_pool->geometry.stride_w);
    }

    /** @brief Builds the stack K_CONV_MODEL describes. */
    NN::LayerStack make_conv_stack()
    {
        return NN::build_layer_stack(
            make_layer_registry(),
            Config::parse_model(Config::parse_json(K_CONV_MODEL, "<test>"), "<test>"));
    }

    /** @brief Traces @p model over a [4, 784] input and lists its parameters as "name:extents". */
    template<typename M>
    std::vector<std::string> traced_names(M& model)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{4, 784}, "inputs");
        model.forward(builder, x);

        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });
        return names;
    }
}

/** @brief Tests that every registry row has a minimal case here, so no factory goes unbuilt. */
TEST(Unit_LayerRegistry, EveryRowHasAMinimalCase)
{
    const NN::NodeRegistry registry = make_layer_registry();
    const SushiRuntime::span<const NN::LayerEntry> layers = registry.layers();

    ASSERT_EQ(K_MINIMAL_CASES.size(), layers.size());

    for (std::size_t i = 0; i < layers.size(); ++i)
        EXPECT_EQ(K_MINIMAL_CASES[i].type, layers[i].type);
}

/** @brief Tests that the registry spells its sixteen type strings as model files do, in order. */
TEST(Unit_LayerRegistry, TheTypeStringsAreTheOnesModelFilesSpell)
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

/** @brief Tests that each factory builds a layer from its minimal object and reports its shape. */
TEST(Unit_LayerRegistry, EachFactoryBuildsFromAMinimalObject)
{
    for (const MinimalCase& minimal : K_MINIMAL_CASES)
    {
        SCOPED_TRACE(std::string(minimal.type));

        const std::unique_ptr<NN::Layer> layer =
            make_layer(minimal.type, minimal.attributes, minimal.input);

        ASSERT_NE(layer, nullptr);
        EXPECT_EQ(layer->output_shape(), minimal.output);
    }
}

/** @brief Tests that each registry key builds the layer type it names and no other. */
TEST(Unit_LayerRegistry, EachKeyBuildsItsOwnLayerType)
{
    const Shape vector{6};
    const Shape image{8, 8, 2};

    EXPECT_NE(wrapped<NN::Linear>(make_layer("linear", R"({"units": 5})", vector)), nullptr);
    EXPECT_NE(wrapped<NN::GELU>(make_layer("gelu", "{}", vector)), nullptr);
    EXPECT_NE(wrapped<NN::ReLU>(make_layer("relu", "{}", vector)), nullptr);
    EXPECT_NE(wrapped<NN::Tanh>(make_layer("tanh", "{}", vector)), nullptr);
    EXPECT_NE(wrapped<NN::Sigmoid>(make_layer("sigmoid", "{}", vector)), nullptr);
    EXPECT_NE(wrapped<NN::LeakyReLU>(make_layer("leaky_relu", "{}", vector)), nullptr);
    EXPECT_NE(wrapped<NN::Unflatten>(make_layer(
                  "unflatten", R"({"height": 1, "width": 2, "channels": 3})", vector)),
              nullptr);
    EXPECT_NE(wrapped<NN::Conv2d>(make_layer("conv2d", R"({"filters": 5, "kernel": 3})", image)),
              nullptr);
    EXPECT_NE(wrapped<NN::MaxPool2d>(make_layer("maxpool2d", R"({"kernel": 2})", image)), nullptr);
    EXPECT_NE(wrapped<NN::AvgPool2d>(make_layer("avgpool2d", R"({"kernel": 2})", image)), nullptr);
    EXPECT_NE(wrapped<NN::GlobalAvgPool>(make_layer("global_avgpool", "{}", image)), nullptr);
    EXPECT_NE(wrapped<NN::Flatten>(make_layer("flatten", "{}", image)), nullptr);
    EXPECT_NE(wrapped<NN::Upsample2d>(make_layer("upsample", "{}", image)), nullptr);
    EXPECT_NE(wrapped<NN::SiLU>(make_layer("silu", "{}", image)), nullptr);
}

/** @brief Tests that each factory refuses a key it does not know and names that key. */
TEST(Unit_LayerRegistry, EachFactoryRejectsAnUnknownKey)
{
    for (const MinimalCase& minimal : K_MINIMAL_CASES)
    {
        const std::string context = "layer 0 '" + std::string(minimal.type) + "'";

        SA_EXPECT_ERROR((void)make_layer(minimal.type, R"({"no_such_key": 1})", minimal.input),
                        context, "'no_such_key'");
    }
}

/** @brief Tests that a model file cannot set a convolution's input channel count. */
TEST(Unit_LayerRegistry, Conv2dTakesItsInputChannelsFromTheInputShape)
{
    const std::unique_ptr<NN::Layer> layer =
        make_layer("conv2d", R"({"filters": 5, "kernel": 3})", Shape{8, 8, 7});
    const NN::Conv2d* conv = wrapped<NN::Conv2d>(layer);

    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->in_channels, 7);
    EXPECT_EQ(conv->out_channels, 5);

    SA_EXPECT_ERROR((void)make_layer("conv2d",
                                     R"({"filters": 5, "kernel": 3, "in_channels": 7})",
                                     Shape{8, 8, 7}),
                    "layer 0 'conv2d'", "'in_channels'");
}

/** @brief Tests that a convolution fed a feature vector is refused with the layer that fixes it. */
TEST(Unit_LayerRegistry, Conv2dRefusesAnInputThatIsNotAnImage)
{
    SA_EXPECT_ERROR((void)make_layer("conv2d", R"({"filters": 5, "kernel": 3})", Shape{784}),
                    "layer 0 'conv2d'", "[784]", "[H, W, C]", "'unflatten'");
}

/** @brief Tests that a convolution defaults to stride 1, no padding, dilation 1 and a bias. */
TEST(Unit_LayerRegistry, Conv2dDefaultsItsOptionalKeys)
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
TEST(Unit_LayerRegistry, Conv2dReadsEveryKeyItAccepts)
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

/** @brief Tests that a kernel spelled as an integer and as [h, w] builds the same window. */
TEST(Unit_LayerRegistry, AKernelAsAnIntegerAndAsAPairAgree)
{
    const Shape image{8, 8, 2};

    for (std::string_view type : {"conv2d", "maxpool2d", "avgpool2d"})
    {
        SCOPED_TRACE(std::string(type));
        const std::string filters = (type == "conv2d") ? R"("filters": 5, )" : "";

        const std::unique_ptr<NN::Layer> from_integer =
            make_layer(type, "{" + filters + R"("kernel": 3})", image);
        const std::unique_ptr<NN::Layer> from_pair =
            make_layer(type, "{" + filters + R"("kernel": [3, 3]})", image);

        EXPECT_EQ(from_integer->output_shape(), from_pair->output_shape());
    }

    const std::unique_ptr<NN::Layer> from_integer =
        make_layer("conv2d", R"({"filters": 5, "kernel": 3})", image);
    const std::unique_ptr<NN::Layer> from_pair =
        make_layer("conv2d", R"({"filters": 5, "kernel": [3, 3]})", image);
    const NN::Conv2d* integer_conv = wrapped<NN::Conv2d>(from_integer);
    const NN::Conv2d* pair_conv = wrapped<NN::Conv2d>(from_pair);

    ASSERT_NE(integer_conv, nullptr);
    ASSERT_NE(pair_conv, nullptr);
    EXPECT_EQ(integer_conv->geometry.kernel_h, 3);
    EXPECT_EQ(integer_conv->geometry.kernel_w, 3);
    EXPECT_EQ(pair_conv->geometry.kernel_h, 3);
    EXPECT_EQ(pair_conv->geometry.kernel_w, 3);

    expect_pool_window_spellings_agree<NN::MaxPool2d>("maxpool2d", image);
    expect_pool_window_spellings_agree<NN::AvgPool2d>("avgpool2d", image);
}

/** @brief Tests that a window setting is refused unless it is an integer or a two-entry array. */
TEST(Unit_LayerRegistry, AWindowSettingOfTheWrongFormIsRefused)
{
    const Shape image{8, 8, 2};

    SA_EXPECT_ERROR((void)make_layer("maxpool2d", R"({"kernel": [2]})", image),
                    "layer 0 'maxpool2d' 'kernel'", "two-entry [h, w]", "1 entries");
    SA_EXPECT_ERROR((void)make_layer("maxpool2d", R"({"kernel": [2, 2, 2]})", image),
                    "layer 0 'maxpool2d' 'kernel'", "two-entry [h, w]", "3 entries");
    SA_EXPECT_ERROR((void)make_layer("avgpool2d", R"({"kernel": "2"})", image),
                    "layer 0 'avgpool2d' 'kernel'", "two-entry [h, w]");
    SA_EXPECT_ERROR((void)make_layer("conv2d", R"({"filters": 5, "kernel": 3, "stride": 1.5})",
                                     image),
                    "layer 0 'conv2d' 'stride'", "two-entry [h, w]");
}

/** @brief Tests that a pool's stride follows its kernel unless the model file sets one. */
TEST(Unit_LayerRegistry, APoolStrideDefaultsToItsKernel)
{
    const Shape image{12, 12, 2};

    const std::unique_ptr<NN::Layer> defaulted =
        make_layer("maxpool2d", R"({"kernel": [2, 3]})", image);
    const NN::MaxPool2d* max_pool = wrapped<NN::MaxPool2d>(defaulted);
    ASSERT_NE(max_pool, nullptr);
    EXPECT_EQ(max_pool->geometry.stride_h, 2);
    EXPECT_EQ(max_pool->geometry.stride_w, 3);
    EXPECT_EQ(max_pool->geometry.pad_top, 0);
    EXPECT_EQ(max_pool->geometry.pad_left, 0);
    EXPECT_EQ(defaulted->output_shape(), (Shape{6, 4, 2}));

    const std::unique_ptr<NN::Layer> spelled =
        make_layer("avgpool2d", R"({"kernel": 3, "stride": 1, "padding": 1})", image);
    const NN::AvgPool2d* avg_pool = wrapped<NN::AvgPool2d>(spelled);
    ASSERT_NE(avg_pool, nullptr);
    EXPECT_EQ(avg_pool->geometry.stride_h, 1);
    EXPECT_EQ(avg_pool->geometry.stride_w, 1);
    EXPECT_EQ(avg_pool->geometry.pad_top, 1);
    EXPECT_EQ(avg_pool->geometry.pad_left, 1);
    EXPECT_EQ(spelled->output_shape(), (Shape{12, 12, 2}));
}

/** @brief Tests that conv2d refuses a non-positive filter count, kernel, stride or dilation. */
TEST(Unit_LayerRegistry, Conv2dRejectsANonPositiveExtent)
{
    const Shape image{8, 8, 2};

    for (std::string_view attributes : {
             R"({"filters": 0, "kernel": 3})",
             R"({"filters": -4, "kernel": 3})",
             R"({"filters": 5, "kernel": 0})",
             R"({"filters": 5, "kernel": [3, 0]})",
             R"({"filters": 5, "kernel": -3})",
             R"({"filters": 5, "kernel": 3, "stride": 0})",
             R"({"filters": 5, "kernel": 3, "stride": [1, -1]})",
             R"({"filters": 5, "kernel": 3, "dilation": 0})"})
    {
        SCOPED_TRACE(std::string(attributes));
        SA_EXPECT_ERROR((void)make_layer("conv2d", attributes, image),
                        "layer 0 'conv2d'", "Conv2d", "positive");
    }
}

/** @brief Tests that both pools refuse a non-positive kernel or stride. */
TEST(Unit_LayerRegistry, ThePoolsRejectANonPositiveExtent)
{
    const Shape image{8, 8, 2};

    for (std::string_view type : {"maxpool2d", "avgpool2d"})
    {
        const std::string context = "layer 0 '" + std::string(type) + "'";

        for (std::string_view attributes : {
                 R"({"kernel": 0})",
                 R"({"kernel": [2, 0]})",
                 R"({"kernel": -2})",
                 R"({"kernel": 2, "stride": 0})",
                 R"({"kernel": 2, "stride": [-1, 2]})"})
        {
            SCOPED_TRACE(std::string(type) + " " + std::string(attributes));
            SA_EXPECT_ERROR((void)make_layer(type, attributes, image), context, "positive");
        }
    }
}

/** @brief Tests that conv2d and both pools refuse a negative padding on either axis. */
TEST(Unit_LayerRegistry, AWindowedLayerRejectsANegativePadding)
{
    const Shape image{8, 8, 2};

    for (std::string_view type : {"conv2d", "maxpool2d", "avgpool2d"})
    {
        const std::string context = "layer 0 '" + std::string(type) + "'";
        const std::string filters = (type == "conv2d") ? R"("filters": 5, )" : "";

        for (std::string_view padding : {"-1", "[0, -1]", "[-1, 0]"})
        {
            const std::string attributes =
                "{" + filters + R"("kernel": 3, "padding": )" + std::string(padding) + "}";

            SCOPED_TRACE(std::string(type) + " " + attributes);
            SA_EXPECT_ERROR((void)make_layer(type, attributes, image), context, "padding");
        }
    }
}

/** @brief Tests that a factory refuses an object missing a key it requires and names the key. */
TEST(Unit_LayerRegistry, AFactoryRejectsAMissingRequiredKey)
{
    const Shape image{8, 8, 2};

    SA_EXPECT_ERROR((void)make_layer("conv2d", R"({"kernel": 3})", image),
                    "layer 0 'conv2d'", "has no 'filters'");
    SA_EXPECT_ERROR((void)make_layer("conv2d", R"({"filters": 5})", image),
                    "layer 0 'conv2d'", "has no 'kernel'");
    SA_EXPECT_ERROR((void)make_layer("maxpool2d", R"({"stride": 2})", image),
                    "layer 0 'maxpool2d'", "has no 'kernel'");
    SA_EXPECT_ERROR((void)make_layer("avgpool2d", R"({"stride": 2})", image),
                    "layer 0 'avgpool2d'", "has no 'kernel'");
    SA_EXPECT_ERROR((void)make_layer("linear", R"({"use_bias": false})", Shape{6}),
                    "layer 0 'linear'", "has no 'units'");
}

/** @brief Tests that leaky_relu refuses a slope outside (0, 1) when the layer is built. */
TEST(Unit_LayerRegistry, LeakyReLURejectsASlopeOutsideItsRange)
{
    for (std::string_view attributes : {R"({"slope": 0})", R"({"slope": 1})",
                                        R"({"slope": -0.1})"})
    {
        SCOPED_TRACE(std::string(attributes));
        SA_EXPECT_ERROR((void)make_layer("leaky_relu", attributes, Shape{6}),
                        "layer 0 'leaky_relu'", "slope");
    }
}

/** @brief Tests that upsample refuses a scale below one when the layer is built. */
TEST(Unit_LayerRegistry, UpsampleRejectsAScaleBelowOne)
{
    SA_EXPECT_ERROR((void)make_layer("upsample", R"({"scale": 0})", Shape{4, 4, 2}),
                    "layer 0 'upsample'", "Upsample2d", "scale");
}

/** @brief Tests that leaky_relu and upsample use 0.01 and 2 until the model file sets a value. */
TEST(Unit_LayerRegistry, LeakyReLUAndUpsampleReadTheirOptionalKey)
{
    const std::unique_ptr<NN::Layer> default_leaky = make_layer("leaky_relu", "{}", Shape{6});
    const std::unique_ptr<NN::Layer> steep_leaky =
        make_layer("leaky_relu", R"({"slope": 0.25})", Shape{6});
    ASSERT_NE(wrapped<NN::LeakyReLU>(default_leaky), nullptr);
    ASSERT_NE(wrapped<NN::LeakyReLU>(steep_leaky), nullptr);
    EXPECT_DOUBLE_EQ(wrapped<NN::LeakyReLU>(default_leaky)->slope, 0.01);
    EXPECT_DOUBLE_EQ(wrapped<NN::LeakyReLU>(steep_leaky)->slope, 0.25);

    const std::unique_ptr<NN::Layer> default_upsample =
        make_layer("upsample", "{}", Shape{4, 4, 2});
    const std::unique_ptr<NN::Layer> triple_upsample =
        make_layer("upsample", R"({"scale": 3})", Shape{4, 4, 2});
    EXPECT_EQ(default_upsample->output_shape(), (Shape{8, 8, 2}));
    EXPECT_EQ(triple_upsample->output_shape(), (Shape{12, 12, 2}));
}

/** @brief Tests that unflatten refuses an input whose element count its image does not hold. */
TEST(Unit_LayerRegistry, UnflattenRefusesAnInputOfAnotherSize)
{
    SA_EXPECT_ERROR((void)make_layer("unflatten",
                                     R"({"height": 28, "width": 28, "channels": 1})",
                                     Shape{783}),
                    "layer 0 'unflatten'", "[783]", "[784]");
    SA_EXPECT_ERROR((void)make_layer("unflatten", R"({"height": 28, "width": 28})", Shape{784}),
                    "layer 0 'unflatten'", "channels");
}

/** @brief Tests that the convolutional model file builds a stack that ends in ten features. */
TEST(Unit_LayerRegistry, TheConvModelBuildsAndReportsItsOutputShape)
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
TEST(Unit_LayerRegistry, TheConvModelTracesToTheShapeItReports)
{
    NN::LayerStack stack = make_conv_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 784}, "inputs");

    EXPECT_EQ(graph.value(stack.forward(builder, x)).shape, (Shape{4, 10}));
}

/** @brief Tests that the convolutional model names its six parameters by layer position. */
TEST(Unit_LayerRegistry, TheConvModelNamesItsParametersByLayerPosition)
{
    NN::LayerStack stack = make_conv_stack();

    EXPECT_EQ(traced_names(stack), K_CONV_MODEL_PARAMETERS);
}

/** @brief Tests that a Sequential of the same layer types yields the model file's names. */
TEST(Unit_LayerRegistry, ASequentialOfTheSameLayersYieldsTheSameNames)
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

/** @brief Tests that differentiating cross-entropy reaches all six parameters of the model. */
TEST(Unit_LayerRegistry, CrossEntropyDifferentiatesToEveryParameterOfTheConvModel)
{
    NN::LayerStack stack = make_conv_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 784}, "inputs");
    const Graph::ValueID targets = builder.input(Shape{4, 10}, "targets");
    const Graph::ValueID logits = stack.forward(builder, x);
    const Graph::ValueID loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    std::vector<std::string> names;
    std::vector<Shape> shapes;
    std::vector<Graph::ValueID> wrt;
    NN::for_each_parameter(stack, [&](const std::string& name, NN::Parameter& parameter)
    {
        names.push_back(name);
        shapes.push_back(parameter.shape());
        wrt.push_back(parameter.id_in(graph));
    });
    ASSERT_EQ(wrt.size(), 6u);

    const std::vector<Graph::ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const Graph::ValueID>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), wrt.size());
    for (std::size_t i = 0; i < gradients.size(); ++i)
    {
        SCOPED_TRACE(names[i]);
        ASSERT_NE(gradients[i], Graph::K_NO_VALUE);
        EXPECT_EQ(graph.value(gradients[i]).shape, shapes[i]);
    }
}

/** @brief Tests that a silu row traces one SILU node that keeps the input's shape. */
TEST(Unit_LayerRegistry, ASiLURowTracesOneSiLUNode)
{
    const std::unique_ptr<NN::Layer> layer = make_layer("silu", "{}", Shape{8, 8, 2});

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 8, 8, 2}, "inputs");
    const std::size_t leaves = graph.node_count();
    const Graph::ValueID y = layer->forward(builder, x);

    ASSERT_EQ(graph.node_count(), leaves + 1u);
    EXPECT_EQ(graph.nodes()[leaves].op, Ops::K_SILU.id);
    EXPECT_EQ(graph.value(y).shape, (Shape{4, 8, 8, 2}));
}

/** @brief Tests that a version 2 chain is seeded with input.shape, not with its element count. */
TEST(Unit_LayerRegistry, AVersionTwoChainStartsFromTheInputShape)
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

    const NN::LayerStack stack = NN::build_layer_stack(
        make_layer_registry(),
        Config::parse_model(Config::parse_json(K_IMAGE_CHAIN, "<test>"), "<test>"));

    ASSERT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.layer(0).output_shape(), (Shape{6, 6, 5}));
    EXPECT_EQ(stack.output_shape(), (Shape{180}));
}

/** @brief Tests that build_layer_stack refuses a row that reads two rows and names build_model. */
TEST(Unit_LayerRegistry, AChainRefusesARowThatReadsTwoRows)
{
    constexpr std::string_view K_MERGING_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh"},
            {"type": "relu", "from": [-1, 0]}
        ]
    })";
    const Config::ModelSpec model =
        Config::parse_model(Config::parse_json(K_MERGING_MODEL, "<test>"), "<test>");

    SA_EXPECT_ERROR((void)NN::build_layer_stack(make_layer_registry(), model),
                    "layer 2 'relu'", "the row before it", "build_model");
}

/** @brief Tests that build_layer_stack refuses a row that reads one row other than the last. */
TEST(Unit_LayerRegistry, AChainRefusesARowThatSkipsBack)
{
    constexpr std::string_view K_SKIPPING_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh"},
            {"type": "relu", "from": 0}
        ]
    })";
    const Config::ModelSpec model =
        Config::parse_model(Config::parse_json(K_SKIPPING_MODEL, "<test>"), "<test>");

    SA_EXPECT_ERROR((void)NN::build_layer_stack(make_layer_registry(), model),
                    "layer 2 'relu'", "the row before it");
}

/** @brief Tests that build_layer_stack refuses a row that reads the model input below row 0. */
TEST(Unit_LayerRegistry, AChainRefusesALaterRowThatReadsTheInput)
{
    constexpr std::string_view K_RESTARTING_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh", "from": "input"}
        ]
    })";
    const Config::ModelSpec model =
        Config::parse_model(Config::parse_json(K_RESTARTING_MODEL, "<test>"), "<test>");

    SA_EXPECT_ERROR((void)NN::build_layer_stack(make_layer_registry(), model),
                    "layer 1 'tanh'", "the row before it");
}

/** @brief Tests that build_layer_stack refuses a file whose outputs publish an earlier row. */
TEST(Unit_LayerRegistry, AChainRefusesOutputsThatPublishAnEarlierRow)
{
    constexpr std::string_view K_EARLY_OUTPUT_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh"}
        ],
        "outputs": [{"node": 0}]
    })";
    const Config::ModelSpec model =
        Config::parse_model(Config::parse_json(K_EARLY_OUTPUT_MODEL, "<test>"), "<test>");

    SA_EXPECT_ERROR((void)NN::build_layer_stack(make_layer_registry(), model),
                    "'outputs'", "last row", "build_model");
}

/** @brief Tests that build_layer_stack builds a file whose outputs name the last row whole. */
TEST(Unit_LayerRegistry, AChainAcceptsOutputsThatPublishTheLastRow)
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

    const NN::LayerStack stack = NN::build_layer_stack(
        make_layer_registry(),
        Config::parse_model(Config::parse_json(K_LAST_OUTPUT_MODEL, "<test>"), "<test>"));

    EXPECT_EQ(stack.size(), 2u);
}

/** @brief Tests that build_layer_stack refuses a row that reads a named port of the row above. */
TEST(Unit_LayerRegistry, AChainRefusesARowThatReadsANamedPort)
{
    constexpr std::string_view K_PORT_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8]},
        "layers": [
            {"type": "relu"},
            {"type": "tanh", "from": {"node": 0, "port": "output"}}
        ]
    })";
    const Config::ModelSpec model =
        Config::parse_model(Config::parse_json(K_PORT_MODEL, "<test>"), "<test>");

    SA_EXPECT_ERROR((void)NN::build_layer_stack(make_layer_registry(), model),
                    "layer 1 'tanh'", "the row before it");
}
