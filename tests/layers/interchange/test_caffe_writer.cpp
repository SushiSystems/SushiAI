/**************************************************************************/
/* test_caffe_writer.cpp                                                  */
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
 * @file test_caffe_writer.cpp
 * @brief Tests the Caffe writers against the readers on the LeNet fixture.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/interchange/caffe_writer.hpp>
#include <SushiAI/io/protobuf_text.hpp>

using namespace SushiAI;
using namespace SushiAI::Interchange;

namespace
{
    /** @brief Returns the path of the LeNet fixture file @p name. */
    std::filesystem::path fixture(const char* name)
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "importers" /
               "fixtures" / "caffe" / name;
    }

    /** @brief Returns the bytes of the file at @p path; empty when it cannot be read. */
    std::string read_file(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>());
    }

    /** @brief Returns @p text as a span of bytes. */
    SushiRuntime::span<const std::byte> as_bytes(const std::string& text)
    {
        return SushiRuntime::span<const std::byte>(
            reinterpret_cast<const std::byte*>(text.data()), text.size());
    }

    /** @brief Expects @p read to hold the fields of @p written, by name, kind and text. */
    void expect_same_fields(const IO::TextField& read, const IO::TextField& written)
    {
        EXPECT_EQ(read.name, written.name);
        EXPECT_EQ(read.kind, written.kind);
        EXPECT_EQ(read.text, written.text);
        ASSERT_EQ(read.fields.size(), written.fields.size()) << written.name;
        for (std::size_t i = 0; i < written.fields.size(); ++i)
            expect_same_fields(read.fields[i], written.fields[i]);
    }
} // namespace

/** @brief Tests that the readers read back LeNet's layers, parameters and blobs. */
TEST(Small_CaffeWriter, TheReadersReadBackLenet)
{
    const std::string prototxt = read_file(fixture("lenet.prototxt"));
    const std::string caffemodel = read_file(fixture("lenet.caffemodel"));
    if (prototxt.empty() || caffemodel.empty())
        GTEST_SKIP() << "run tools/reference/make_caffe_lenet_fixture.py first";

    SA_ASSERT_OK_AND_ASSIGN(CaffeNet net, read_caffe_prototxt(prototxt, "lenet.prototxt"));
    SA_ASSERT_OK_AND_ASSIGN(const CaffeNet weights,
                            read_caffe_model(as_bytes(caffemodel), "lenet.caffemodel"));
    SA_EXPECT_OK(attach_caffe_weights(net, weights));

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, write_caffe_prototxt(net));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::byte> file, write_caffe_model(net));
    SA_ASSERT_OK_AND_ASSIGN(CaffeNet back, read_caffe_prototxt(text, "written.prototxt"));
    SA_ASSERT_OK_AND_ASSIGN(const CaffeNet back_weights,
                            read_caffe_model(SushiRuntime::span<const std::byte>(file.data(),
                                                                                 file.size()),
                                             "written.caffemodel"));
    SA_EXPECT_OK(attach_caffe_weights(back, back_weights));

    EXPECT_EQ(back.name, net.name);
    EXPECT_EQ(back.inputs, net.inputs);
    EXPECT_EQ(back.input_shapes, net.input_shapes);
    ASSERT_EQ(back.layers.size(), net.layers.size());
    for (std::size_t i = 0; i < net.layers.size(); ++i)
    {
        const CaffeLayer& a = back.layers[i];
        const CaffeLayer& b = net.layers[i];
        EXPECT_EQ(a.name, b.name);
        EXPECT_EQ(a.type, b.type);
        EXPECT_EQ(a.bottoms, b.bottoms);
        EXPECT_EQ(a.tops, b.tops);
        for (const char* parameter : {"convolution_param", "pooling_param",
                                      "inner_product_param"})
        {
            const IO::TextField* written = IO::find_field(b.parameters, parameter);
            const IO::TextField* read = IO::find_field(a.parameters, parameter);
            ASSERT_EQ(read == nullptr, written == nullptr) << b.name << " " << parameter;
            if (written != nullptr)
                expect_same_fields(*read, *written);
        }
        ASSERT_EQ(a.blobs.size(), b.blobs.size()) << b.name;
        for (std::size_t k = 0; k < b.blobs.size(); ++k)
        {
            EXPECT_EQ(a.blobs[k].dimensions, b.blobs[k].dimensions);
            const SushiRuntime::span<const std::byte> x = a.blobs[k].data.bytes();
            const SushiRuntime::span<const std::byte> y = b.blobs[k].data.bytes();
            EXPECT_EQ(std::vector<std::byte>(x.begin(), x.end()),
                      std::vector<std::byte>(y.begin(), y.end()));
        }
    }
}

/** @brief Tests that a layer with no type is refused by name. */
TEST(Small_CaffeWriter, ALayerWithNoTypeIsRefused)
{
    CaffeNet net;
    CaffeLayer layer;
    layer.name = "conv1";
    net.layers.push_back(layer);

    SA_EXPECT_FAILURE(write_caffe_prototxt(net), StatusCode::InvalidDescription, "'conv1'");
}
