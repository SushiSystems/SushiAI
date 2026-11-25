/**************************************************************************/
/* test_caffe_import.cpp                                                  */
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
 * @file test_caffe_import.cpp
 * @brief Tests the Caffe importer: its rows, its weights, its refusals and the LeNet fixture.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>
#include <SushiAI/vision/importers/caffe_import.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds an input blob of 2 channels over 4 by 4 pixels, as a net-level input. */
    constexpr const char* K_IMAGE_INPUT =
        "input: \"data\"\ninput_shape { dim: 1 dim: 2 dim: 4 dim: 4 }\n";

    /** @brief Holds the file name the test nets state as their origin. */
    constexpr const char* K_ORIGIN = "test.prototxt";

    /** @brief Names the fixture folder of the LeNet files. */
    std::filesystem::path fixture_folder()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "importers" /
               "fixtures" / "caffe";
    }

    /** @brief Returns the text of one layer message. */
    std::string layer_text(std::string_view name, std::string_view type,
                           std::initializer_list<std::string_view> bottoms,
                           std::initializer_list<std::string_view> tops,
                           std::string_view parameters = "")
    {
        std::string text = "layer { name: \"" + std::string(name) + "\" type: \"" +
                           std::string(type) + "\" ";
        for (const std::string_view bottom : bottoms)
            text += "bottom: \"" + std::string(bottom) + "\" ";
        for (const std::string_view top : tops)
            text += "top: \"" + std::string(top) + "\" ";
        return text + std::string(parameters) + " }\n";
    }

    /** @brief Holds the floats of the blobs a layer of a test net carries. */
    struct LayerBlobs
    {
        std::string layer;                    /**< The layer's name. */
        std::vector<std::vector<float>> blobs; /**< The blobs, in order. */
    };

    /** @brief Returns a blob that owns @p values as little-endian float32. */
    Interchange::CaffeBlob make_blob(const std::vector<float>& values)
    {
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        if (!values.empty())
            std::memcpy(bytes.data(), values.data(), bytes.size());
        Interchange::CaffeBlob blob;
        blob.data = Interchange::TensorBytes::owning(std::move(bytes));
        return blob;
    }

    /** @brief Reads @p prototxt, gives each layer of @p weights its blobs and imports the net. */
    Result<IO::ImportedModel> import_net(const std::string& prototxt,
                                         const std::vector<LayerBlobs>& weights = {})
    {
        SA_TRY_ASSIGN(Interchange::CaffeNet net,
                      Interchange::read_caffe_prototxt(prototxt, K_ORIGIN));
        for (const LayerBlobs& held : weights)
        {
            for (Interchange::CaffeLayer& layer : net.layers)
            {
                if (layer.name != held.layer)
                    continue;
                for (const std::vector<float>& values : held.blobs)
                    layer.blobs.push_back(make_blob(values));
            }
        }
        return Vision::Importers::import_caffe_net(net, K_ORIGIN);
    }

    /** @brief Returns the types of the rows of @p imported, in order. */
    std::vector<std::string> row_types(const IO::ImportedModel& imported)
    {
        std::vector<std::string> types;
        for (const Config::LayerSpec& row : imported.model.layers)
            types.push_back(row.type);
        return types;
    }

    /** @brief Returns the whole numbers @p key of @p row holds: one, or the entries of an array. */
    std::vector<std::int64_t> integers_of(const Config::LayerSpec& row, std::string_view key)
    {
        std::vector<std::int64_t> values;
        if (!row.attributes.has(key))
        {
            ADD_FAILURE() << "the row '" << row.type << "' has no '" << key << "'";
            return values;
        }
        const Config::Value& stated = row.attributes.at(key, "row").value();
        if (stated.kind() == Config::Kind::Int)
        {
            values.push_back(stated.as_int("row").value());
            return values;
        }
        for (const Config::Value& entry : stated.as_array("row").value())
            values.push_back(entry.as_int("row").value());
        return values;
    }

    /** @brief Returns the number @p key of @p row holds. */
    double real_of(const Config::LayerSpec& row, std::string_view key)
    {
        if (!row.attributes.has(key))
        {
            ADD_FAILURE() << "the row '" << row.type << "' has no '" << key << "'";
            return 0.0;
        }
        return row.attributes.at(key, "row").value().as_real("row").value();
    }

    /** @brief Returns the tensor named @p name of @p imported, or null when none is. */
    const IO::ImportedTensor* find_tensor(const IO::ImportedModel& imported,
                                          std::string_view name)
    {
        for (const IO::ImportedTensor& tensor : imported.weights)
        {
            if (tensor.record.name == name)
                return &tensor;
        }
        ADD_FAILURE() << "no tensor named '" << name << "'";
        return nullptr;
    }

    /** @brief Returns the floats of the tensor named @p name; empty when there is none. */
    std::vector<float> floats_of(const IO::ImportedModel& imported, std::string_view name)
    {
        const IO::ImportedTensor* tensor = find_tensor(imported, name);
        std::vector<float> values;
        if (tensor == nullptr)
            return values;
        values.resize(tensor->payload.size() / sizeof(float));
        std::memcpy(values.data(), tensor->payload.data(), tensor->payload.size());
        return values;
    }

    /** @brief Returns the net of a BatchNorm layer and the Scale layer after it. */
    std::string batch_norm_and_scale_net()
    {
        return std::string(K_IMAGE_INPUT) +
               layer_text("bn", "BatchNorm", {"data"}, {"bn_out"},
                          "batch_norm_param { eps: 0.25 }") +
               layer_text("sc", "Scale", {"bn_out"}, {"sc_out"},
                          "scale_param { bias_term: true }");
    }

    /** @brief Returns the blobs of the BatchNorm and Scale layers of batch_norm_and_scale_net. */
    std::vector<LayerBlobs> batch_norm_and_scale_blobs()
    {
        // mean {2, 4} and variance {4, 8} are stored times the scale factor 2.
        return {{"bn", {{2.0f, 4.0f}, {4.0f, 8.0f}, {2.0f}}}, {"sc", {{3.0f, 5.0f}, {0.5f, -1.0f}}}};
    }
} // namespace

/** @brief Tests that a BatchNorm layer and the Scale layer after it become one row. */
TEST(Small_CaffeImport, BatchNormAndScaleBecomeOneBatchnorm2dRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            import_net(batch_norm_and_scale_net(), batch_norm_and_scale_blobs()));

    ASSERT_EQ(row_types(imported), std::vector<std::string>{"batchnorm2d"});
    // The epsilon is copied from the text 0.25, which a double holds exactly.
    EXPECT_EQ(real_of(imported.model.layers[0], "bn_epsilon"), 0.25);
    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].node, 0);
    EXPECT_EQ(imported.model.outputs[0].name, "sc_out");
}

/**
 * @brief Tests that the folded row stores the scale and shift of the Scale layer and the mean and
 *        variance of the BatchNorm layer divided by its scale factor.
 *
 * Every value is a dyadic rational, so each result is exact: the mean {2, 4} over the factor 2 is
 * {1, 2}, and the variance {4, 8} over 2 is {2, 4}.
 */
TEST(Small_CaffeImport, BatchNormAndScaleStoreTheFoldedValuesExactly)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            import_net(batch_norm_and_scale_net(), batch_norm_and_scale_blobs()));

    EXPECT_EQ(floats_of(imported, "0.weight"), (std::vector<float>{3.0f, 5.0f}));
    EXPECT_EQ(floats_of(imported, "0.bias"), (std::vector<float>{0.5f, -1.0f}));
    EXPECT_EQ(floats_of(imported, "0.running_mean"), (std::vector<float>{1.0f, 2.0f}));
    EXPECT_EQ(floats_of(imported, "0.running_var"), (std::vector<float>{2.0f, 4.0f}));
    EXPECT_EQ(imported.weights.size(), 4u);
}

/** @brief Tests that a BatchNorm layer with no Scale stores a scale of 1 and a shift of 0. */
TEST(Small_CaffeImport, ABatchNormWithoutAScaleStoresUnitScaleAndZeroShift)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("bn", "BatchNorm", {"data"}, {"bn"});
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel imported,
        import_net(net, {{"bn", {{2.0f, 4.0f}, {4.0f, 8.0f}, {2.0f}}}}));

    EXPECT_EQ(floats_of(imported, "0.weight"), (std::vector<float>{1.0f, 1.0f}));
    EXPECT_EQ(floats_of(imported, "0.bias"), (std::vector<float>{0.0f, 0.0f}));
}

/** @brief Tests that a BatchNorm layer whose scale factor is 0 stores a mean and variance of 0. */
TEST(Small_CaffeImport, AZeroScaleFactorStoresZeroStatistics)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("bn", "BatchNorm", {"data"}, {"bn"});
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel imported,
        import_net(net, {{"bn", {{2.0f, 4.0f}, {4.0f, 8.0f}, {0.0f}}}}));

    EXPECT_EQ(floats_of(imported, "0.running_mean"), (std::vector<float>{0.0f, 0.0f}));
    EXPECT_EQ(floats_of(imported, "0.running_var"), (std::vector<float>{0.0f, 0.0f}));
}

/** @brief Tests that a Scale layer that no BatchNorm layer precedes is refused. */
TEST(Small_CaffeImport, AScaleThatNoBatchNormPrecedesIsRefused)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("lonely", "Scale", {"data"}, {"scaled"});

    SA_EXPECT_FAILURE(import_net(net), StatusCode::Unsupported, "layer 'lonely' (Scale)");
}

/** @brief Tests that every layer without a row is named in one refusal. */
TEST(Small_CaffeImport, UnsupportedLayersAreListedTogether)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("up", "Deconvolution", {"data"}, {"up"}) +
                            layer_text("norm", "LRN", {"up"}, {"norm"}) +
                            layer_text("fine", "ReLU", {"norm"}, {"fine"});

    SA_EXPECT_FAILURE(import_net(net), StatusCode::Unsupported, "2 layers",
                      "layer 'up' (Deconvolution)", "layer 'norm' (LRN)");
}

/** @brief Tests that a layer of the net that only the training net holds is left out. */
TEST(Small_CaffeImport, ALayerOfTheTrainingNetAloneIsLeftOut)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("train_data", "Data", {}, {"labels"},
                                       "include { phase: TRAIN }") +
                            layer_text("act", "ReLU", {"data"}, {"act"});

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported), std::vector<std::string>{"relu"});
}

/** @brief Tests that a layer that reads a blob no layer writes is refused, naming the blob. */
TEST(Small_CaffeImport, ALayerThatReadsAnUnwrittenBlobIsRefused)
{
    SA_EXPECT_FAILURE(import_net(layer_text("act", "ReLU", {"data"}, {"act"})),
                      StatusCode::InvalidDescription, "layer 'act' (ReLU)", "'data'");
}

/** @brief Tests that a net with no layer is refused for stating no input. */
TEST(Small_CaffeImport, ANetWithoutLayersIsRefusedForItsMissingInput)
{
    SA_EXPECT_FAILURE(import_net("name: \"empty\"\n"), StatusCode::InvalidDescription,
                      "declares no input");
}

/** @brief Tests that a Convolution layer becomes a conv2d row of the stated window. */
TEST(Small_CaffeImport, ConvolutionBecomesAConv2dRowOfTheStatedWindow)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 3 kernel_size: 3 stride: 2 pad: 1 "
                   "bias_term: false }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    ASSERT_EQ(row_types(imported), std::vector<std::string>{"conv2d"});
    const Config::LayerSpec& row = imported.model.layers[0];
    EXPECT_EQ(integers_of(row, "filters"), std::vector<std::int64_t>{3});
    EXPECT_EQ(integers_of(row, "kernel"), std::vector<std::int64_t>{3});
    EXPECT_EQ(integers_of(row, "stride"), std::vector<std::int64_t>{2});
    EXPECT_EQ(integers_of(row, "padding"), std::vector<std::int64_t>{1});
    EXPECT_FALSE(row.attributes.at("use_bias", "row").value().as_bool("row").value());
    EXPECT_EQ(imported.model.input_shape, (Shape{4, 4, 2}));
}

/**
 * @brief Tests that a convolution weight [out, in, kh, kw] is stored as [out, kh, kw, in].
 *
 * One filter over two channels and a 1 by 2 kernel holds 0, 1 | 2, 3: channel 0 has the taps 0
 * and 1, channel 1 the taps 2 and 3. Stored by tap, each tap lists its channels, so the record
 * reads (tap 0: 0, 2) then (tap 1: 1, 3).
 */
TEST(Small_CaffeImport, ConvolutionWeightsMoveToTheRecordLayout)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 1 kernel_h: 1 kernel_w: 2 bias_term: false }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            import_net(net, {{"conv", {{0.0f, 1.0f, 2.0f, 3.0f}}}}));

    const IO::ImportedTensor* weight = find_tensor(imported, "0.weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->record.shape, (Shape{1, 1, 2, 2}));
    EXPECT_EQ(floats_of(imported, "0.weight"), (std::vector<float>{0.0f, 2.0f, 1.0f, 3.0f}));
}

/** @brief Tests that a convolution that states groups becomes a conv2d row with the key groups. */
TEST(Small_CaffeImport, ConvolutionGroupsBecomeTheGroupsKey)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 4 kernel_size: 1 group: 2 bias_term: false }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    ASSERT_EQ(row_types(imported), std::vector<std::string>{"conv2d"});
    EXPECT_EQ(integers_of(imported.model.layers[0], "groups"), std::vector<std::int64_t>{2});
}

/** @brief Tests that a convolution of one group states no groups key. */
TEST(Small_CaffeImport, ConvolutionOfOneGroupStatesNoGroupsKey)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 4 kernel_size: 1 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_FALSE(imported.model.layers[0].attributes.has("groups"));
}

/**
 * @brief Tests that a grouped weight [out, in / group, kh, kw] is stored [out, kh, kw, in / group].
 *
 * Four filters over two channels in two groups read one channel each, so the blob is [4, 1, 1, 1]
 * and the stored weight [4, 1, 1, 1] holds the four values in order.
 */
TEST(Small_CaffeImport, GroupedConvolutionWeightsKeepOneChannelPerFilter)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 4 kernel_size: 1 group: 2 bias_term: false }");
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel imported,
        import_net(net, {{"conv", {{1.0f, 2.0f, 3.0f, 4.0f}}}}));

    const IO::ImportedTensor* weight = find_tensor(imported, "0.weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->record.shape, (Shape{4, 1, 1, 1}));
    EXPECT_EQ(floats_of(imported, "0.weight"), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

/**
 * @brief Tests that a depthwise weight [C, 1, kh, kw] is stored [kh, kw, C].
 *
 * Two filters over two channels in two groups, with a 1 by 2 kernel: filter 0 holds the taps 10,
 * 11 and filter 1 the taps 20, 21, so tap 0 lists (10, 20) and tap 1 lists (11, 21).
 */
TEST(Small_CaffeImport, DepthwiseConvolutionWeightsAreKernelHeightWidthChannel)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 2 kernel_h: 1 kernel_w: 2 group: 2 "
                   "bias_term: false }");
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel imported,
        import_net(net, {{"conv", {{10.0f, 11.0f, 20.0f, 21.0f}}}}));

    const IO::ImportedTensor* weight = find_tensor(imported, "0.weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->record.shape, (Shape{1, 2, 2}));
    EXPECT_EQ(floats_of(imported, "0.weight"), (std::vector<float>{10.0f, 20.0f, 11.0f, 21.0f}));
}

/** @brief Tests that groups that divide neither the channels nor the filters are refused. */
TEST(Small_CaffeImport, ConvolutionGroupsThatDoNotDivideTheChannelsAreRefused)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 3 kernel_size: 1 group: 3 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::InvalidDescription,
                      "layer 'conv' (Convolution)", "group 3", "2 channels");
}

/** @brief Tests that a group of 0 is refused. */
TEST(Small_CaffeImport, ConvolutionOfZeroGroupsIsRefused)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 2 kernel_size: 1 group: 0 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::InvalidDescription,
                      "layer 'conv' (Convolution)", "group of 1 or more");
}

/** @brief Tests that Caffe's rounded-up window count becomes a trailing pad of the pool row. */
TEST(Small_CaffeImport, AMaxPoolOverAnOddExtentGetsTrailingPadding)
{
    const std::string net = "input: \"data\"\ninput_shape { dim: 1 dim: 1 dim: 5 dim: 5 }\n" +
                            layer_text("pool", "Pooling", {"data"}, {"pool"},
                                       "pooling_param { pool: MAX kernel_size: 2 stride: 2 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    // Caffe yields ceil((5 - 2) / 2) + 1 = 3 windows, the last needing one pad cell after column 4.
    ASSERT_EQ(row_types(imported), std::vector<std::string>{"maxpool2d"});
    EXPECT_EQ(integers_of(imported.model.layers[0], "trailing_padding"),
              (std::vector<std::int64_t>{1, 1}));
}

/** @brief Tests that an average pool whose last window leaves the input is refused. */
TEST(Small_CaffeImport, AnAveragePoolPastThePaddedEdgeIsRefused)
{
    const std::string net = "input: \"data\"\ninput_shape { dim: 1 dim: 1 dim: 5 dim: 5 }\n" +
                            layer_text("pool", "Pooling", {"data"}, {"pool"},
                                       "pooling_param { pool: AVE kernel_size: 2 stride: 2 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::Unsupported, "layer 'pool' (Pooling)",
                      "average window");
}

/** @brief Tests that an average pool that fits becomes an avgpool2d row without a trailing pad. */
TEST(Small_CaffeImport, AnAveragePoolThatFitsBecomesAnAvgpool2dRow)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("pool", "Pooling", {"data"}, {"pool"},
                                       "pooling_param { pool: AVE kernel_size: 2 stride: 2 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    ASSERT_EQ(row_types(imported), std::vector<std::string>{"avgpool2d"});
    EXPECT_FALSE(imported.model.layers[0].attributes.has("trailing_padding"));
}

/** @brief Tests that global average pooling becomes a global_avgpool row. */
TEST(Small_CaffeImport, GlobalAveragePoolingBecomesAGlobalAvgpoolRow)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("pool", "Pooling", {"data"}, {"pool"},
                                       "pooling_param { pool: AVE global_pooling: true }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported), std::vector<std::string>{"global_avgpool"});
}

/** @brief Tests that a ReLU with no slope becomes a relu row. */
TEST(Small_CaffeImport, ReluBecomesAReluRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel imported,
        import_net(std::string(K_IMAGE_INPUT) + layer_text("act", "ReLU", {"data"}, {"act"})));

    EXPECT_EQ(row_types(imported), std::vector<std::string>{"relu"});
}

/** @brief Tests that a ReLU with a negative slope becomes a leaky_relu row of that slope. */
TEST(Small_CaffeImport, AReluWithASlopeBecomesALeakyReluRow)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("act", "ReLU", {"data"}, {"act"},
                                       "relu_param { negative_slope: 0.125 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    ASSERT_EQ(row_types(imported), std::vector<std::string>{"leaky_relu"});
    // 0.125 is a dyadic rational, so the text and the double agree exactly.
    EXPECT_EQ(real_of(imported.model.layers[0], "slope"), 0.125);
}

/** @brief Tests that an in-place ReLU publishes the blob it rewrote, from its own row. */
TEST(Small_CaffeImport, AnInPlaceReluPublishesTheBlobFromItsOwnRow)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 1 kernel_size: 1 }") +
        layer_text("act", "ReLU", {"conv"}, {"conv"});
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].node, 1);
    EXPECT_EQ(imported.model.outputs[0].name, "conv");
}

/** @brief Tests that an InnerProduct layer over an image flattens it first. */
TEST(Small_CaffeImport, AnInnerProductAfterAnImageFlattensFirst)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("fc", "InnerProduct", {"data"}, {"fc"},
                                       "inner_product_param { num_output: 3 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported), (std::vector<std::string>{"flatten", "linear"}));
    EXPECT_EQ(integers_of(imported.model.layers[1], "units"), std::vector<std::int64_t>{3});
}

/**
 * @brief Tests that the inputs of an inner product follow the order of a flattened [H, W, C] image.
 *
 * Caffe flattens [C, H, W]: the weight index of (c, h, w) is (c * 2 + h) * 2 + w, with 2 by 2
 * pixels over 2 channels, and holds that index as its value. A flattened [H, W, C] image lists
 * (h, w, c) in the order (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) (1,0,1) (1,1,0) (1,1,1), whose
 * Caffe indices are 0 4 1 5 2 6 3 7.
 */
TEST(Small_CaffeImport, InnerProductWeightsFollowTheChannelsLastFlatten)
{
    const std::string net = "input: \"data\"\ninput_shape { dim: 1 dim: 2 dim: 2 dim: 2 }\n" +
                            layer_text("fc", "InnerProduct", {"data"}, {"fc"},
                                       "inner_product_param { num_output: 1 bias_term: false }");
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel imported,
        import_net(net, {{"fc", {{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}}}}));

    EXPECT_EQ(floats_of(imported, "1.weight"),
              (std::vector<float>{0.0f, 4.0f, 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f}));
}

/** @brief Tests that a Concat along the channels becomes one concat row over its bottoms' rows. */
TEST(Small_CaffeImport, ConcatJoinsTheChannelsOfItsBottoms)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("left", "ReLU", {"data"}, {"left"}) +
                            layer_text("right", "ReLU", {"data"}, {"right"}) +
                            layer_text("joined", "Concat", {"left", "right"}, {"joined"});
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported), (std::vector<std::string>{"relu", "relu", "concat"}));
    ASSERT_EQ(imported.model.layers[2].from.size(), 2u);
    EXPECT_EQ(imported.model.layers[2].from[0].node, 0);
    EXPECT_EQ(imported.model.layers[2].from[1].node, 1);
}

/** @brief Tests that a Concat along another axis than the channels is refused. */
TEST(Small_CaffeImport, ConcatAlongAnotherAxisIsRefused)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("left", "ReLU", {"data"}, {"left"}) +
                            layer_text("right", "ReLU", {"data"}, {"right"}) +
                            layer_text("joined", "Concat", {"left", "right"}, {"joined"},
                                       "concat_param { axis: 2 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::Unsupported, "layer 'joined' (Concat)",
                      "axis 2");
}

/** @brief Tests that a sum of three bottoms becomes two add rows, the second over the first. */
TEST(Small_CaffeImport, EltwiseSumOfThreeBottomsChainsTwoAddRows)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("a", "ReLU", {"data"}, {"a"}) +
                            layer_text("b", "ReLU", {"data"}, {"b"}) +
                            layer_text("c", "ReLU", {"data"}, {"c"}) +
                            layer_text("sum", "Eltwise", {"a", "b", "c"}, {"sum"});
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported),
              (std::vector<std::string>{"relu", "relu", "relu", "add", "add"}));
    ASSERT_EQ(imported.model.layers[4].from.size(), 2u);
    EXPECT_EQ(imported.model.layers[4].from[0].node, 3);
    EXPECT_EQ(imported.model.layers[4].from[1].node, 2);
}

/** @brief Tests that a sum weighted by a coefficient other than 1 is refused. */
TEST(Small_CaffeImport, AnEltwiseCoefficientOtherThanOneIsRefused)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("a", "ReLU", {"data"}, {"a"}) +
                            layer_text("b", "ReLU", {"data"}, {"b"}) +
                            layer_text("sum", "Eltwise", {"a", "b"}, {"sum"},
                                       "eltwise_param { coeff: 1 coeff: -1 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::Unsupported, "layer 'sum' (Eltwise)",
                      "weight 1");
}

/** @brief Holds one Eltwise operation with the row it becomes. */
struct EltwiseRow
{
    std::string operation; /**< The operation the prototxt spells. */
    std::string row;       /**< The type of the rows of the chain. */
};

/** @brief Runs each Eltwise operation that is not a sum. */
class Small_CaffeEltwiseOperation : public ::testing::TestWithParam<EltwiseRow>
{
};

/** @brief Tests that PROD and MAX become a chain of mul or maximum rows, one per extra bottom. */
TEST_P(Small_CaffeEltwiseOperation, TheOperationBecomesAChainOfItsRow)
{
    const EltwiseRow& row = GetParam();
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("a", "ReLU", {"data"}, {"a"}) +
                            layer_text("b", "ReLU", {"data"}, {"b"}) +
                            layer_text("c", "ReLU", {"data"}, {"c"}) +
                            layer_text("both", "Eltwise", {"a", "b", "c"}, {"both"},
                                       "eltwise_param { operation: " + row.operation + " }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported),
              (std::vector<std::string>{"relu", "relu", "relu", row.row, row.row}));
    ASSERT_EQ(imported.model.layers[4].from.size(), 2u);
    EXPECT_EQ(imported.model.layers[4].from[0].node, 3);
    EXPECT_EQ(imported.model.layers[4].from[1].node, 2);
}

/** @brief Tests that a coefficient on a PROD or a MAX is refused as Caffe refuses it. */
TEST_P(Small_CaffeEltwiseOperation, ACoefficientOnTheOperationIsRefused)
{
    const EltwiseRow& row = GetParam();
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("a", "ReLU", {"data"}, {"a"}) +
                            layer_text("b", "ReLU", {"data"}, {"b"}) +
                            layer_text("both", "Eltwise", {"a", "b"}, {"both"},
                                       "eltwise_param { operation: " + row.operation +
                                           " coeff: 1 coeff: 1 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::InvalidDescription, "layer 'both' (Eltwise)",
                      "coefficients", row.operation, "SUM only");
}

INSTANTIATE_TEST_SUITE_P(Operations, Small_CaffeEltwiseOperation,
                         ::testing::Values(EltwiseRow{"PROD", "mul"},
                                           EltwiseRow{"MAX", "maximum"}),
                         [](const ::testing::TestParamInfo<EltwiseRow>& info)
                         { return info.param.operation; });

/** @brief Tests that an operation Caffe does not know is refused and named. */
TEST(Small_CaffeImport, AnUnknownEltwiseOperationIsRefused)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("a", "ReLU", {"data"}, {"a"}) +
                            layer_text("b", "ReLU", {"data"}, {"b"}) +
                            layer_text("both", "Eltwise", {"a", "b"}, {"both"},
                                       "eltwise_param { operation: MEAN }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::InvalidDescription, "layer 'both' (Eltwise)",
                      "MEAN", "SUM, PROD and MAX");
}

/** @brief Tests that a Softmax over the features of a vector becomes a softmax row the net publishes. */
TEST(Small_CaffeImport, ASoftmaxOverAVectorBecomesASoftmaxRowAndIsPublished)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("fc", "InnerProduct", {"data"}, {"scores"},
                                       "inner_product_param { num_output: 3 }") +
                            layer_text("prob", "Softmax", {"scores"}, {"prob"});
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported), (std::vector<std::string>{"flatten", "linear", "softmax"}));
    EXPECT_FALSE(imported.model.layers[2].attributes.has("axis"));
    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].name, "prob");
    EXPECT_EQ(imported.model.outputs[0].node, 2);
    EXPECT_TRUE(imported.notes.empty());
}

/** @brief Tests that a Softmax over the channels of an image states no axis, the channels being last. */
TEST(Small_CaffeImport, ASoftmaxOverTheChannelsOfAnImageStatesNoAxis)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("prob", "Softmax", {"data"}, {"prob"});
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported), std::vector<std::string>{"softmax"});
    EXPECT_FALSE(imported.model.layers[0].attributes.has("axis"));
}

/**
 * @brief Tests that the axis of a Softmax over an image moves to the channel-last order.
 *
 * Caffe's axes 2 and 3 of [N, C, H, W] are the rows and the columns, which sit at the positions 0
 * and 1 of a sample [H, W, C].
 */
TEST(Small_CaffeImport, ASoftmaxOverTheRowsOfAnImageStatesTheRowAxis)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("prob", "Softmax", {"data"}, {"prob"},
                                       "softmax_param { axis: 2 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(integers_of(imported.model.layers[0], "axis"), std::vector<std::int64_t>{0});
}

/** @brief Tests that a Softmax that another layer reads is an interior row, and only the last is published. */
TEST(Small_CaffeImport, ASoftmaxThatAnotherLayerReadsIsAnInteriorRow)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("fc", "InnerProduct", {"data"}, {"scores"},
                                       "inner_product_param { num_output: 3 }") +
                            layer_text("prob", "Softmax", {"scores"}, {"prob"}) +
                            layer_text("fc2", "InnerProduct", {"prob"}, {"out"},
                                       "inner_product_param { num_output: 2 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported),
              (std::vector<std::string>{"flatten", "linear", "softmax", "linear"}));
    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].name, "out");
}

/** @brief Tests that a Softmax over the batch axis is refused. */
TEST(Small_CaffeImport, ASoftmaxOverTheBatchAxisIsRefused)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("prob", "Softmax", {"data"}, {"prob"},
                                       "softmax_param { axis: 0 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::Unsupported, "layer 'prob' (Softmax)",
                      "batch axis");
}

/** @brief Tests that a Softmax over an axis the blob does not have is refused. */
TEST(Small_CaffeImport, ASoftmaxOverAnAxisBeyondTheBlobIsRefused)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("prob", "Softmax", {"data"}, {"prob"},
                                       "softmax_param { axis: 4 }");

    SA_EXPECT_FAILURE(import_net(net), StatusCode::InvalidDescription, "layer 'prob' (Softmax)",
                      "axis 4");
}

/** @brief Tests that a Dropout layer becomes a dropout row that keeps the stated ratio. */
TEST(Small_CaffeImport, DropoutKeepsItsRatioAsTheRate)
{
    const std::string net = std::string(K_IMAGE_INPUT) +
                            layer_text("drop", "Dropout", {"data"}, {"drop"},
                                       "dropout_param { dropout_ratio: 0.75 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    ASSERT_EQ(row_types(imported), std::vector<std::string>{"dropout"});
    // 0.75 is a dyadic rational, so the text and the double agree exactly.
    EXPECT_EQ(real_of(imported.model.layers[0], "rate"), 0.75);
}

/** @brief Tests that a net with no caffemodel imports its rows and no tensor. */
TEST(Small_CaffeImport, AWeightlessNetImportsWithoutTensors)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 1 kernel_size: 1 }");
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_net(net));

    EXPECT_EQ(row_types(imported), std::vector<std::string>{"conv2d"});
    EXPECT_TRUE(imported.weights.empty());
}

/** @brief Tests that a weight blob that does not fill the layer's extents is refused. */
TEST(Small_CaffeImport, ABlobThatDoesNotFillItsLayerIsIncompatible)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("conv", "Convolution", {"data"}, {"conv"},
                   "convolution_param { num_output: 1 kernel_size: 1 bias_term: false }");

    // One filter over two channels and a 1 by 1 kernel holds two floats, not three.
    SA_EXPECT_FAILURE(import_net(net, {{"conv", {{1.0f, 2.0f, 3.0f}}}}), StatusCode::Incompatible,
                      "layer 'conv' (Convolution)", "3 floats", "gives 2");
}

/** @brief Tests that a layer without blobs in a net that carries weights is refused. */
TEST(Small_CaffeImport, ALayerWithoutBlobsInAWeightedNetIsIncompatible)
{
    const std::string net =
        std::string(K_IMAGE_INPUT) +
        layer_text("first", "Convolution", {"data"}, {"first"},
                   "convolution_param { num_output: 1 kernel_size: 1 bias_term: false }") +
        layer_text("second", "Convolution", {"first"}, {"second"},
                   "convolution_param { num_output: 1 kernel_size: 1 bias_term: false }");

    SA_EXPECT_FAILURE(import_net(net, {{"first", {{1.0f, 2.0f}}}}), StatusCode::Incompatible,
                      "layer 'second' (Convolution)", "need 1");
}

/** @brief Tests that three files are refused before any is opened. */
TEST(Small_CaffeImport, ThreeFilesAreRefusedBeforeAnyIsOpened)
{
    const std::filesystem::path files[] = {"a.prototxt", "b.caffemodel", "c.caffemodel"};

    SA_EXPECT_FAILURE(Vision::Importers::import_caffe_files(
                          SushiRuntime::span<const std::filesystem::path>(files, 3)),
                      StatusCode::InvalidDescription, "'caffe'", "3 files");
}

namespace
{
    /** @brief Holds the taps along the longest path of the fixture's LeNet: 1 * 5 * 5 of the
     *         first convolution, 4 * 5 * 5 of the second, 8 * 4 * 4 and 32 of the two products. */
    constexpr std::size_t K_LENET_PATH_TAPS = 25 + 100 + 128 + 32;

    /** @brief Holds the roundings the softmax adds to a probability: the shift, the exponential, the
     *         sum of ten, the logarithm, the difference and the last exponential. */
    constexpr std::size_t K_SOFTMAX_ROUNDINGS = 16;

    /** @brief Names the LeNet files and skips the test when the exporter has not written them. */
    class Medium_CaffeLenetImport : public SushiAI::Test::EngineTest
    {
        protected:
            /** @brief Skips the test when a file of the fixture folder is missing. */
            void SetUp() override
            {
                SushiAI::Test::EngineTest::SetUp();
                for (const char* name : {"lenet.prototxt", "lenet.caffemodel", "lenet.expected.json"})
                {
                    if (!std::filesystem::exists(fixture_folder() / name))
                        GTEST_SKIP() << "run tools/reference/make_caffe_lenet_fixture.py and "
                                        "export_caffe_reference.py first";
                }
            }

            /** @brief Returns the import of the two LeNet files. */
            static Result<IO::ImportedModel> import_lenet()
            {
                const std::filesystem::path files[] = {fixture_folder() / "lenet.prototxt",
                                                       fixture_folder() / "lenet.caffemodel"};
                return Vision::Importers::import_caffe_files(
                    SushiRuntime::span<const std::filesystem::path>(files, 2));
            }

            /** @brief Returns the values of the entry @p name of the expected file's outputs. */
            static Result<std::vector<float>> read_expected(std::string_view name)
            {
                SA_TRY_ASSIGN(const Config::Value root,
                              Config::parse_json_file(fixture_folder() / "lenet.expected.json"));
                SA_TRY_ASSIGN(const Config::Value& outputs, root.at("outputs", "expected"));
                SA_TRY_ASSIGN(const std::vector<Config::Value>& entries,
                              outputs.as_array("outputs"));
                for (const Config::Value& entry : entries)
                {
                    SA_TRY_ASSIGN(const Config::Value& stated, entry.at("name", "output"));
                    SA_TRY_ASSIGN(const std::string& stated_name, stated.as_string("name"));
                    if (stated_name == name)
                        return read_values(entry);
                }
                SA_FAIL_IF(true, StatusCode::Incompatible, "the expected file has no '{}'", name);
                return std::vector<float>();
            }

            /** @brief Returns the "values" of @p entry as floats. */
            static Result<std::vector<float>> read_values(const Config::Value& entry)
            {
                SA_TRY_ASSIGN(const Config::Value& values, entry.at("values", "output"));
                SA_TRY_ASSIGN(const std::vector<Config::Value>& numbers, values.as_array("values"));
                std::vector<float> floats;
                for (const Config::Value& number : numbers)
                {
                    SA_TRY_ASSIGN(const double value, number.as_real("value"));
                    floats.push_back(static_cast<float>(value));
                }
                return floats;
            }

            /** @brief Runs @p imported over @p input and returns the published tensor @p name. */
            Result<std::vector<float>> run_model(const IO::ImportedModel& imported,
                                                 const std::vector<float>& input,
                                                 std::string_view name)
            {
                SushiBLAS::Engine& engine = backend().engine();
                const Shape& sample = imported.model.input_shape;
                const Shape batch_shape{1, sample[0], sample[1], sample[2]};

                NN::NodeRegistry registry;
                SA_TRY(NN::register_layers(registry));
                SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, imported.model));
                SA_TRY_ASSIGN(const Train::TracedStep traced,
                              Train::build_evaluation_graph(graph, batch_shape));
                SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, graph));
                SA_TRY(SushiAI::Test::fill_imported_state(
                    SushiRuntime::span<const IO::ImportedTensor>(imported.weights.data(),
                                                                 imported.weights.size()),
                    state));

                Graph::TensorPool pool(engine);
                SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                              Train::EvaluationStep::make(backend(), traced, state, pool));
                SA_TRY_ASSIGN(SushiBLAS::Tensor image, allocate_tensor(engine, batch_shape));
                SA_FAIL_IF(static_cast<std::size_t>(image.num_elements) != input.size(),
                           StatusCode::Incompatible, "the input holds {} values, the model needs {}",
                           input.size(), image.num_elements);
                SA_TRY(::SushiAI::write_elements(image, input));
                SA_TRY(step->run(image));

                SA_TRY_ASSIGN(const SushiBLAS::Tensor* const output,
                              step->output(std::string(name)));
                return ::SushiAI::read_elements<float>(*output);
            }
    };
} // namespace

/** @brief Tests that the LeNet files import to the nine rows of the prototxt and eight tensors. */
TEST_F(Medium_CaffeLenetImport, TheImportedLenetHoldsTheRowsAndTheTensorsOfItsFiles)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_lenet());

    EXPECT_EQ(row_types(imported),
              (std::vector<std::string>{"conv2d", "maxpool2d", "conv2d", "maxpool2d", "flatten",
                                        "linear", "relu", "linear", "softmax"}));
    EXPECT_EQ(imported.weights.size(), 8u);
    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].name, "prob");
}

/**
 * @brief Tests that the imported LeNet publishes the softmax of the scores OpenCV computes.
 *
 * A probability moves by at most twice the change of a score, K_LENET_PATH_TAPS unit roundoffs
 * times the largest score, and the softmax adds K_SOFTMAX_ROUNDINGS unit roundoffs of its own.
 */
TEST_F(Medium_CaffeLenetImport, TheProbabilitiesAreTheSoftmaxOfOpenCvsScores)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_lenet());
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> scores, read_expected("ip2"));
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value root,
                            Config::parse_json_file(fixture_folder() / "lenet.expected.json"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> input,
                            read_values(root.at("input", "expected").value()));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> probabilities,
                            run_model(imported, input, "prob"));

    ASSERT_EQ(probabilities.size(), scores.size());
    double magnitude = 0.0;
    double total = 0.0;
    for (const float score : scores)
    {
        magnitude = std::max(magnitude, std::fabs(static_cast<double>(score)));
        total += std::exp(static_cast<double>(score));
    }
    const double bound = (2.0 * static_cast<double>(K_LENET_PATH_TAPS) * magnitude +
                          static_cast<double>(K_SOFTMAX_ROUNDINGS)) *
                         TestSupport::unit_roundoff<float>();
    for (std::size_t i = 0; i < scores.size(); ++i)
    {
        const double expected = std::exp(static_cast<double>(scores[i])) / total;
        EXPECT_TRUE(TestSupport::is_within(
            TestSupport::measure_error(probabilities[i], expected, bound)))
            << "probability " << i;
    }
}
