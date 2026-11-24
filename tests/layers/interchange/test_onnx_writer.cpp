/**************************************************************************/
/* test_onnx_writer.cpp                                                   */
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
 * @file test_onnx_writer.cpp
 * @brief Tests the ONNX writer against read_onnx_graph: every attribute kind, initializers and
 *        a symbolic batch extent.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/interchange/onnx_writer.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>

using namespace SushiAI;
using namespace SushiAI::Interchange;

namespace
{
    /** @brief Returns a float tensor named @p name of @p dimensions holding @p values. */
    OnnxTensor float_tensor(const std::string& name, std::vector<std::int64_t> dimensions,
                            const std::vector<float>& values)
    {
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        OnnxTensor tensor;
        tensor.name = name;
        tensor.dimensions = std::move(dimensions);
        tensor.data_type = OnnxDataType::Float;
        tensor.data = TensorBytes::owning(std::move(bytes));
        return tensor;
    }

    /** @brief Returns an attribute named @p name of @p type. */
    OnnxAttribute attribute(const std::string& name, OnnxAttributeType type)
    {
        OnnxAttribute out;
        out.name = name;
        out.type = type;
        return out;
    }

    /** @brief Returns a value info named @p name of a float tensor with a batch symbol. */
    OnnxValueInfo value(const std::string& name)
    {
        OnnxValueInfo info;
        info.name = name;
        info.element_type = OnnxDataType::Float;
        info.has_shape = true;
        OnnxDimension batch;
        batch.parameter = "batch";
        OnnxDimension width;
        width.value = 4;
        info.dimensions = {batch, width};
        return info;
    }

    /** @brief Expects @p read to hold the name, type, extents and bytes of @p written. */
    void expect_same_tensor(const OnnxTensor& read, const OnnxTensor& written)
    {
        EXPECT_EQ(read.name, written.name);
        EXPECT_EQ(read.data_type, written.data_type);
        EXPECT_EQ(read.dimensions, written.dimensions);
        const SushiRuntime::span<const std::byte> a = read.data.bytes();
        const SushiRuntime::span<const std::byte> b = written.data.bytes();
        EXPECT_EQ(std::vector<std::byte>(a.begin(), a.end()),
                  std::vector<std::byte>(b.begin(), b.end()));
    }
} // namespace

/** @brief Tests that the reader reads back a graph of every attribute kind. */
TEST(Small_OnnxWriter, TheReaderReadsBackTheGraph)
{
    OnnxGraph body;
    body.name = "body";
    OnnxNode identity;
    identity.op_type = "Identity";
    identity.inputs = {"x"};
    identity.outputs = {"y"};
    body.nodes = {identity};

    OnnxNode node;
    node.name = "everything";
    node.op_type = "Custom";
    node.domain = "sushiai.test";
    node.inputs = {"input", "", "w"};
    node.outputs = {"output"};
    OnnxAttribute real = attribute("alpha", OnnxAttributeType::Float);
    real.real = 0.125f;
    OnnxAttribute whole = attribute("axis", OnnxAttributeType::Int);
    whole.integer = -1;
    OnnxAttribute text = attribute("mode", OnnxAttributeType::String);
    text.text = "nearest";
    OnnxAttribute tensor = attribute("value", OnnxAttributeType::Tensor);
    tensor.tensor = float_tensor("", {2}, {1.5f, -2.0f});
    OnnxAttribute graph = attribute("then_branch", OnnxAttributeType::Graph);
    graph.graphs = {body};
    OnnxAttribute reals = attribute("scales", OnnxAttributeType::Floats);
    reals.reals = {1.0f, 2.0f};
    OnnxAttribute integers = attribute("pads", OnnxAttributeType::Ints);
    integers.integers = {0, 1, -3};
    OnnxAttribute texts = attribute("names", OnnxAttributeType::Strings);
    texts.texts = {"a", "b"};
    OnnxAttribute tensors = attribute("bank", OnnxAttributeType::Tensors);
    tensors.tensors = {float_tensor("t0", {1}, {3.0f})};
    OnnxAttribute graphs = attribute("branches", OnnxAttributeType::Graphs);
    graphs.graphs = {body, body};
    node.attributes = {real, whole, text, tensor, graph, reals, integers, texts, tensors, graphs};

    OnnxGraph written;
    written.name = "model";
    written.ir_version = 8;
    written.producer_name = "SushiAI";
    written.producer_version = "1";
    written.opsets = {OnnxOpset{"", 17}};
    written.nodes = {node};
    written.initializers = {float_tensor("w", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f})};
    written.inputs = {value("input")};
    written.outputs = {value("output")};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::byte> file, write_onnx_graph(written));
    SA_ASSERT_OK_AND_ASSIGN(const OnnxGraph read,
                            read_onnx_graph(SushiRuntime::span<const std::byte>(file.data(),
                                                                                file.size()),
                                            "written"));

    EXPECT_EQ(read.name, "model");
    EXPECT_EQ(read.ir_version, 8);
    EXPECT_EQ(read.producer_name, "SushiAI");
    EXPECT_EQ(read.opset_version(""), 17);
    ASSERT_EQ(read.initializers.size(), 1u);
    expect_same_tensor(read.initializers[0], written.initializers[0]);
    ASSERT_EQ(read.inputs.size(), 1u);
    ASSERT_EQ(read.inputs[0].dimensions.size(), 2u);
    EXPECT_EQ(read.inputs[0].dimensions[0].parameter, "batch");
    EXPECT_EQ(read.inputs[0].dimensions[1].value, 4);

    ASSERT_EQ(read.nodes.size(), 1u);
    const OnnxNode& back = read.nodes[0];
    EXPECT_EQ(back.name, "everything");
    EXPECT_EQ(back.domain, "sushiai.test");
    EXPECT_EQ(back.inputs, node.inputs);
    EXPECT_EQ(back.outputs, node.outputs);
    ASSERT_EQ(back.attributes.size(), node.attributes.size());
    for (std::size_t i = 0; i < node.attributes.size(); ++i)
    {
        EXPECT_EQ(back.attributes[i].name, node.attributes[i].name);
        EXPECT_EQ(back.attributes[i].type, node.attributes[i].type);
    }
    EXPECT_EQ(back.attributes[0].real, 0.125f);
    EXPECT_EQ(back.attributes[1].integer, -1);
    EXPECT_EQ(back.attributes[2].text, "nearest");
    expect_same_tensor(back.attributes[3].tensor, tensor.tensor);
    ASSERT_EQ(back.attributes[4].graphs.size(), 1u);
    EXPECT_EQ(back.attributes[4].graphs[0].nodes.size(), 1u);
    EXPECT_EQ(back.attributes[5].reals, reals.reals);
    EXPECT_EQ(back.attributes[6].integers, integers.integers);
    EXPECT_EQ(back.attributes[7].texts, texts.texts);
    ASSERT_EQ(back.attributes[8].tensors.size(), 1u);
    expect_same_tensor(back.attributes[8].tensors[0], tensors.tensors[0]);
    EXPECT_EQ(back.attributes[9].graphs.size(), 2u);
}

/** @brief Tests that a tensor whose bytes do not fill its extents is refused by name. */
TEST(Small_OnnxWriter, ATensorThatDoesNotFillItsExtentsIsRefused)
{
    OnnxGraph graph;
    graph.initializers = {float_tensor("short", {3}, {1.0f})};

    SA_EXPECT_FAILURE(write_onnx_graph(graph), StatusCode::InvalidDescription, "'short'");
}
