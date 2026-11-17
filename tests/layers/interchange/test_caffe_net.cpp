/**************************************************************************/
/* test_caffe_net.cpp                                                     */
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
 * @file test_caffe_net.cpp
 * @brief Tests the Caffe net readers: a prototxt, a caffemodel and the join of the two.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include "protobuf_bytes.hpp"

using namespace SushiAI;
using namespace SushiAI::Test::Wire;

namespace
{
    /** @brief Names the prototxt in every message. */
    constexpr std::string_view K_PROTOTXT = "tiny.prototxt";

    /** @brief Names the caffemodel in every message. */
    constexpr std::string_view K_MODEL = "tiny.caffemodel";

    /** @brief Holds the type number V1LayerParameter gives INNER_PRODUCT. */
    constexpr std::int64_t K_V1_INNER_PRODUCT = 14;

    using Integers = std::vector<std::int64_t>;

    /** @brief Holds a net of a convolution and a V1 inner product, with a net-level input. */
    constexpr std::string_view K_TINY_PROTOTXT = R"(name: "Tiny"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 4
input_dim: 4
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param { num_output: 2 kernel_size: 3 }
}
layers {
  name: "ip1"
  type: INNER_PRODUCT
  bottom: "conv1"
  top: "ip1"
  inner_product_param { num_output: 3 }
}
)";

    /** @brief Returns the 18 weights of conv1: 0.5 times the element index. */
    std::vector<float> conv_weights()
    {
        std::vector<float> values;
        for (int i = 0; i < 18; ++i)
            values.push_back(0.5f * static_cast<float>(i));
        return values;
    }

    /** @brief Returns a BlobProto with a shape message and packed floats. */
    std::string shaped_blob(const Integers& dimensions, const std::vector<float>& values)
    {
        return bytes_field(7, packed_integers(1, dimensions)) + packed_floats(5, values);
    }

    /** @brief Returns the blobs field of a V2 layer holding @p blob. */
    std::string blob_of_v2(const std::string& blob)
    {
        return bytes_field(7, blob);
    }

    /** @brief Returns a LayerParameter field of a NetParameter. */
    std::string v2_layer(std::string_view name, std::string_view type, const std::string& blobs)
    {
        return bytes_field(100, bytes_field(1, name) + bytes_field(2, type) +
                                    bytes_field(3, "in") + bytes_field(4, "out") + blobs);
    }

    /** @brief Returns the V1 layer ip1 with a blob stated by the four legacy extents. */
    std::string v1_inner_product()
    {
        const std::string blob = integer_field(1, 1) + integer_field(2, 1) + integer_field(3, 3) +
                                 integer_field(4, 2) +
                                 packed_floats(5, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        return bytes_field(2, bytes_field(2, "conv1") + bytes_field(3, "ip1") +
                                  bytes_field(4, "ip1") + integer_field(5, K_V1_INNER_PRODUCT) +
                                  bytes_field(6, blob));
    }

    /** @brief Returns the caffemodel of the tiny net: conv1 weights and bias, and ip1. */
    std::string tiny_model_text()
    {
        const std::string conv_blobs =
            blob_of_v2(shaped_blob({2, 1, 3, 3}, conv_weights())) +
            blob_of_v2(shaped_blob({2}, {10.0f, 20.0f}));
        return bytes_field(1, "Tiny") + v2_layer("conv1", "Convolution", conv_blobs) +
               v2_layer("relu1", "ReLU", "") + v1_inner_product();
    }

    /** @brief Reads the tiny prototxt. */
    Result<Interchange::CaffeNet> tiny_net()
    {
        return Interchange::read_caffe_prototxt(K_TINY_PROTOTXT, K_PROTOTXT);
    }

    /** @brief Reads @p file as a caffemodel; @p file must outlive the result. */
    Result<Interchange::CaffeNet> read_model(const std::vector<std::byte>& file)
    {
        return Interchange::read_caffe_model(view_of(file), K_MODEL);
    }

    /** @brief Returns the float whose four little-endian bytes start at @p at. */
    float float_at(const std::byte* at)
    {
        std::uint32_t bits = 0;
        for (unsigned i = 0; i < 4; ++i)
            bits |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(at[i])) << (8u * i);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
}

/** @brief Tests that a prototxt gives its layers with types, wiring and parameter messages. */
TEST(Small_CaffeNet, ReadsTheLayersOfAPrototxt)
{
    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet net, tiny_net());

    EXPECT_EQ(net.name, "Tiny");
    EXPECT_EQ(net.inputs, (std::vector<std::string>{"data"}));
    ASSERT_EQ(net.input_shapes.size(), 1u);
    EXPECT_EQ(net.input_shapes[0], (Integers{1, 1, 4, 4}));
    ASSERT_EQ(net.layers.size(), 2u);
    const Interchange::CaffeLayer& conv = net.layers[0];
    EXPECT_EQ(conv.name, "conv1");
    EXPECT_EQ(conv.type, "Convolution");
    EXPECT_EQ(conv.bottoms, (std::vector<std::string>{"data"}));
    EXPECT_EQ(conv.tops, (std::vector<std::string>{"conv1"}));
    EXPECT_FALSE(conv.legacy);
    const IO::TextField* parameters = IO::find_field(conv.parameters, "convolution_param");
    ASSERT_NE(parameters, nullptr);
    ASSERT_NE(IO::find_field(*parameters, "num_output"), nullptr);
    EXPECT_EQ(IO::find_field(*parameters, "num_output")->text, "2");
}

/** @brief Tests that a layer of the V1 spelling reads as a layer, its type as the V2 string. */
TEST(Small_CaffeNet, ReadsAV1LayerAsALayer)
{
    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet net, tiny_net());

    const Interchange::CaffeLayer& ip = net.layers[1];
    EXPECT_EQ(ip.name, "ip1");
    EXPECT_EQ(ip.type, "InnerProduct");
    EXPECT_TRUE(ip.legacy);
    EXPECT_EQ(ip.bottoms, (std::vector<std::string>{"conv1"}));
    EXPECT_NE(IO::find_field(ip.parameters, "inner_product_param"), nullptr);
}

/** @brief Tests that an input_shape message states the shape of an input. */
TEST(Small_CaffeNet, ReadsInputShapesFromInputShapeMessages)
{
    constexpr std::string_view text =
        "input: \"a\"\ninput: \"b\"\ninput_shape { dim: 1 dim: 3 }\ninput_shape { dim: 2 }\n";

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet net,
                            Interchange::read_caffe_prototxt(text, K_PROTOTXT));

    ASSERT_EQ(net.input_shapes.size(), 2u);
    EXPECT_EQ(net.input_shapes[0], (Integers{1, 3}));
    EXPECT_EQ(net.input_shapes[1], (Integers{2}));
}

/** @brief Tests that a caffemodel's blobs come back by layer, a packed blob viewed in place. */
TEST(Small_CaffeNet, ReadsTheBlobsOfACaffemodel)
{
    const std::string text = tiny_model_text();
    const std::vector<std::byte> file = to_bytes(text);
    // The weights are the first packed float field of the file; find where its bytes start.
    std::string payload;
    for (float value : conv_weights())
        payload += float_bytes(value);
    const std::size_t offset = text.find(payload);
    ASSERT_NE(offset, std::string::npos);

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet net, read_model(file));

    EXPECT_EQ(net.name, "Tiny");
    ASSERT_EQ(net.layers.size(), 3u);
    const Interchange::CaffeLayer& conv = net.layers[0];
    EXPECT_EQ(conv.name, "conv1");
    EXPECT_EQ(conv.type, "Convolution");
    ASSERT_EQ(conv.blobs.size(), 2u);
    EXPECT_EQ(conv.blobs[0].dimensions, (Integers{2, 1, 3, 3}));
    EXPECT_EQ(conv.blobs[0].element_count(), 18u);
    EXPECT_TRUE(conv.blobs[0].data.is_view());
    EXPECT_EQ(conv.blobs[0].data.bytes().data(), file.data() + offset);
    EXPECT_EQ(float_at(conv.blobs[0].data.bytes().data() + 17 * 4), 8.5f);
    EXPECT_EQ(conv.blobs[1].dimensions, (Integers{2}));
    EXPECT_EQ(float_at(conv.blobs[1].data.bytes().data() + 4), 20.0f);
    EXPECT_TRUE(net.layers[1].blobs.empty());
    const Interchange::CaffeLayer& ip = net.layers[2];
    EXPECT_TRUE(ip.legacy);
    EXPECT_EQ(ip.type, "InnerProduct");
    ASSERT_EQ(ip.blobs.size(), 1u);
    EXPECT_EQ(ip.blobs[0].dimensions, (Integers{1, 1, 3, 2}));
    EXPECT_EQ(ip.blobs[0].element_count(), 6u);
}

/** @brief Tests that floats written one field each read into an owned blob. */
TEST(Small_CaffeNet, ReadsFloatsThatAreNotPacked)
{
    const std::string blob = bytes_field(7, packed_integers(1, {3})) + float_field(5, 1.0f) +
                             float_field(5, 2.0f) + float_field(5, 3.0f);
    const std::vector<std::byte> file =
        to_bytes(v2_layer("ip", "InnerProduct", blob_of_v2(blob)));

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet net, read_model(file));

    const Interchange::CaffeBlob& read = net.layers[0].blobs[0];
    EXPECT_FALSE(read.data.is_view());
    ASSERT_EQ(read.element_count(), 3u);
    EXPECT_EQ(float_at(read.data.bytes().data()), 1.0f);
    EXPECT_EQ(float_at(read.data.bytes().data() + 8), 3.0f);
}

/** @brief Tests that a prototxt and a caffemodel join by layer name, V1 and V2 alike. */
TEST(Small_CaffeNet, JoinsWeightsToLayersByName)
{
    const std::vector<std::byte> file = to_bytes(tiny_model_text());
    SA_ASSERT_OK_AND_ASSIGN(Interchange::CaffeNet net, tiny_net());
    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet weights, read_model(file));

    SA_EXPECT_OK(Interchange::attach_caffe_weights(net, weights));

    ASSERT_EQ(net.layers[0].blobs.size(), 2u);
    EXPECT_EQ(net.layers[0].blobs[0].dimensions, (Integers{2, 1, 3, 3}));
    EXPECT_EQ(net.layers[0].blobs[0].data.bytes().data(),
              weights.layers[0].blobs[0].data.bytes().data());
    ASSERT_EQ(net.layers[1].blobs.size(), 1u);
    EXPECT_EQ(net.layers[1].blobs[0].dimensions, (Integers{1, 1, 3, 2}));
    // The structure the prototxt gave is kept beside the weights.
    EXPECT_NE(IO::find_field(net.layers[0].parameters, "convolution_param"), nullptr);
}

/** @brief Tests that weights of layers the net lacks are named, and the net is left as it was. */
TEST(Small_CaffeNet, NamesTheWeightsLayersAbsentFromTheNet)
{
    const std::string blob = blob_of_v2(shaped_blob({1}, {1.0f}));
    const std::vector<std::byte> file =
        to_bytes(tiny_model_text() + v2_layer("fc9", "InnerProduct", blob) +
                 v2_layer("fc10", "InnerProduct", blob));
    SA_ASSERT_OK_AND_ASSIGN(Interchange::CaffeNet net, tiny_net());
    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet weights, read_model(file));

    SA_EXPECT_FAILURE(Interchange::attach_caffe_weights(net, weights), StatusCode::Incompatible,
                      "'fc9'", "'fc10'", "does not define");
    EXPECT_TRUE(net.layers[0].blobs.empty());
    EXPECT_TRUE(net.layers[1].blobs.empty());
}

/** @brief Tests that a weights layer with no blobs and no place in the net is no error. */
TEST(Small_CaffeNet, IgnoresAWeightslessLayerTheNetLacks)
{
    const std::vector<std::byte> file = to_bytes(v2_layer("loss", "SoftmaxWithLoss", ""));
    SA_ASSERT_OK_AND_ASSIGN(Interchange::CaffeNet net, tiny_net());
    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet weights, read_model(file));

    SA_EXPECT_OK(Interchange::attach_caffe_weights(net, weights));
}

/** @brief Tests that a layer of one name and two types, or two layers of one name, is refused. */
TEST(Small_CaffeNet, RefusesWeightsThatDoNotFitTheLayer)
{
    const std::string blob = blob_of_v2(shaped_blob({1}, {1.0f}));
    const std::vector<std::byte> other_type =
        to_bytes(v2_layer("conv1", "InnerProduct", blob));
    SA_ASSERT_OK_AND_ASSIGN(Interchange::CaffeNet net, tiny_net());
    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet weights, read_model(other_type));
    Interchange::CaffeNet twice = net;
    twice.layers.push_back(net.layers[0]);

    SA_EXPECT_FAILURE(Interchange::attach_caffe_weights(net, weights), StatusCode::Incompatible,
                      "'conv1'", "Convolution", "InnerProduct");
    SA_EXPECT_FAILURE(Interchange::attach_caffe_weights(twice, weights),
                      StatusCode::Incompatible, "2 layers 'conv1'");
}

/** @brief Tests that a blob whose floats do not fill its extents is refused. */
TEST(Small_CaffeNet, RefusesABlobWhoseFloatsDoNotFillItsExtents)
{
    const std::vector<std::byte> file = to_bytes(
        v2_layer("conv1", "Convolution", blob_of_v2(shaped_blob({2, 2}, {1.0f, 2.0f, 3.0f}))));

    SA_EXPECT_FAILURE(read_model(file), StatusCode::InvalidFile, "'tiny.caffemodel'", "'conv1'",
                      "3 floats", "give 4");
}

/** @brief Tests that a blob of doubles, or of a partial float, is refused. */
TEST(Small_CaffeNet, RefusesBlobsItCannotReadAsFloats)
{
    const std::vector<std::byte> doubles = to_bytes(
        v2_layer("conv1", "Convolution", blob_of_v2(bytes_field(8, std::string(8, '\0')))));
    const std::vector<std::byte> ragged = to_bytes(
        v2_layer("conv1", "Convolution", blob_of_v2(bytes_field(5, std::string(6, '\0')))));

    SA_EXPECT_FAILURE(read_model(doubles), StatusCode::Unsupported, "doubles");
    SA_EXPECT_FAILURE(read_model(ragged), StatusCode::InvalidFile, "6 bytes",
                      "whole number of floats");
}

/** @brief Tests that a V1 layer of a type number the table lacks is refused with the number. */
TEST(Small_CaffeNet, RefusesAV1TypeNumberTheTableLacks)
{
    const std::vector<std::byte> file = to_bytes(
        bytes_field(2, bytes_field(4, "odd") + integer_field(5, 99)));

    SA_EXPECT_FAILURE(read_model(file), StatusCode::InvalidFile, "'odd'", "type number 99");
}

/** @brief Tests that the V1 type number 39, DECONVOLUTION in caffe.proto, reads as Deconvolution. */
TEST(Small_CaffeNet, ReadsTheV1TypeNumberOfADeconvolution)
{
    const std::vector<std::byte> file =
        to_bytes(bytes_field(2, bytes_field(4, "up") + integer_field(5, 39)));

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::CaffeNet net, read_model(file));

    ASSERT_EQ(net.layers.size(), 1u);
    EXPECT_EQ(net.layers[0].type, "Deconvolution");
}

/** @brief Tests that prototxt text that is no net, or that states a layer badly, names the line. */
TEST(Small_CaffeNet, RefusesAPrototxtThatStatesALayerBadly)
{
    SA_EXPECT_FAILURE(Interchange::read_caffe_prototxt("layer { name: \"a\" }\n", K_PROTOTXT),
                      StatusCode::InvalidFile, "'tiny.prototxt'", "line 1", "name and a type");
    SA_EXPECT_FAILURE(
        Interchange::read_caffe_prototxt("layers { name: \"a\" type: FANCY }\n", K_PROTOTXT),
        StatusCode::InvalidFile, "line 1", "'FANCY'");
    SA_EXPECT_FAILURE(Interchange::read_caffe_prototxt("input: \"a\"\ninput_dim: 1\n", K_PROTOTXT),
                      StatusCode::InvalidFile, "input_dim holds 1 numbers for 1 inputs");
    SA_EXPECT_FAILURE(Interchange::read_caffe_prototxt("input_shape { dim: x }\n", K_PROTOTXT),
                      StatusCode::InvalidFile, "line 1", "'dim' needs an integer");
    SA_EXPECT_FAILURE(Interchange::read_caffe_prototxt("layer {\n", K_PROTOTXT),
                      StatusCode::InvalidFile, "'tiny.prototxt'");
}
