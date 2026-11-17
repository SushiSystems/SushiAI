/**************************************************************************/
/* test_onnx_graph.cpp                                                    */
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
 * @file test_onnx_graph.cpp
 * @brief Tests the ONNX graph reader on a model assembled byte by byte in the test.
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
#include <SushiAI/interchange/onnx_graph.hpp>
#include "protobuf_bytes.hpp"

using namespace SushiAI;
using namespace SushiAI::Test::Wire;

namespace
{
    /** @brief Names the file in every message. */
    constexpr std::string_view K_ORIGIN = "fixture.onnx";

    /** @brief Holds the number ONNX gives the float element type. */
    constexpr std::int64_t K_FLOAT = 1;

    /** @brief Holds the number ONNX gives the int64 element type. */
    constexpr std::int64_t K_INT64 = 7;

    /** @brief Holds the number ONNX gives the int32 element type. */
    constexpr std::int64_t K_INT32 = 6;

    /** @brief Holds the eight raw bytes of the initializer "w": 1.5f then -2.0f. */
    const std::string K_W_RAW = float_bytes(1.5f) + float_bytes(-2.0f);

    /** @brief Returns a TensorProto holding the fields in @p body, named @p name. */
    std::string tensor(std::string_view name, const std::vector<std::int64_t>& dims,
                       std::int64_t data_type, const std::string& body)
    {
        return packed_integers(1, dims) + integer_field(2, data_type) + body +
               bytes_field(8, name);
    }

    /** @brief Returns an AttributeProto with the name and the fields in @p body. */
    std::string attribute(std::string_view name, const std::string& body)
    {
        return bytes_field(1, name) + body;
    }

    /** @brief Returns a NodeProto. */
    std::string node(std::string_view op_type, std::string_view name,
                     const std::vector<std::string_view>& inputs,
                     const std::vector<std::string_view>& outputs, const std::string& attributes)
    {
        std::string out;
        for (std::string_view input : inputs)
            out += bytes_field(1, input);
        for (std::string_view output : outputs)
            out += bytes_field(2, output);
        return out + bytes_field(3, name) + bytes_field(4, op_type) + attributes;
    }

    /** @brief Returns a ValueInfoProto of the float type with the dimensions given as a body. */
    std::string value_info(std::string_view name, const std::string& dimensions)
    {
        const std::string shape = bytes_field(2, dimensions);
        const std::string tensor_type = integer_field(1, K_FLOAT) + shape;
        return bytes_field(1, name) + bytes_field(2, bytes_field(1, tensor_type));
    }

    /** @brief Returns a Dimension field holding a stated extent. */
    std::string extent(std::int64_t value)
    {
        return bytes_field(1, integer_field(1, value));
    }

    /** @brief Returns a Dimension field holding a symbol. */
    std::string symbol(std::string_view name)
    {
        return bytes_field(1, bytes_field(2, name));
    }

    /** @brief Returns a GraphProto whose body is @p body. */
    std::string graph(const std::string& body)
    {
        return bytes_field(2, "tiny") + body;
    }

    /** @brief Returns a ModelProto around the GraphProto @p graph_body. */
    std::string model(const std::string& graph_body)
    {
        return integer_field(1, 8) + bytes_field(2, "unit-test") +
               bytes_field(8, bytes_field(1, "") + integer_field(2, 13)) +
               bytes_field(8, bytes_field(1, "com.vendor") + integer_field(2, 2)) +
               bytes_field(7, graph_body);
    }

    /** @brief Returns the nodes of the tiny graph: a Conv, a Relu and a Probe with no attribute. */
    std::string tiny_nodes()
    {
        const std::string conv_attributes =
            bytes_field(5, attribute("kernel_shape", packed_integers(8, {1, 1}) +
                                                         integer_field(20, 7))) +
            bytes_field(5, attribute("group", integer_field(3, 1) + integer_field(20, 2))) +
            bytes_field(5, attribute("mode", bytes_field(4, "constant") + integer_field(20, 3))) +
            // The Float attribute names no type: the reader implies it from field 2.
            bytes_field(5, attribute("epsilon", float_field(2, 0.5f)));
        return bytes_field(1, node("Conv", "conv0", {"x", "w"}, {"y"}, conv_attributes) +
                                  integer_field(99, 5)) +
               bytes_field(1, node("Relu", "relu0", {"y"}, {"z"}, ""));
    }

    /** @brief Returns the initializers "w" (raw), "b" (typed floats) and "ids" (int64 data). */
    std::string tiny_initializers()
    {
        return bytes_field(5, tensor("w", {2, 1, 1, 1}, K_FLOAT, bytes_field(9, K_W_RAW))) +
               bytes_field(5, tensor("b", {2}, K_FLOAT, packed_floats(4, {0.25f, 0.5f}))) +
               bytes_field(5, tensor("ids", {3}, K_INT64, packed_integers(7, {7, -1, 300})));
    }

    /** @brief Returns the inputs and outputs of the tiny graph. */
    std::string tiny_signature()
    {
        return bytes_field(11, value_info("x", symbol("N") + extent(1) + extent(4) + extent(4))) +
               bytes_field(12, value_info("z", symbol("N") + extent(2) + extent(4) + extent(4)));
    }

    /** @brief Returns the whole tiny model. */
    std::string tiny_model()
    {
        return model(graph(tiny_nodes() + tiny_initializers() + tiny_signature()));
    }

    /** @brief Reads @p file as an ONNX model; @p file must outlive the result. */
    Result<Interchange::OnnxGraph> read(const std::vector<std::byte>& file)
    {
        return Interchange::read_onnx_graph(view_of(file), K_ORIGIN);
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

/** @brief Tests that nodes come back with operator, name and wiring, unknown fields skipped. */
TEST(Small_OnnxGraph, ReadsNodesWithTheirNamesAndWires)
{
    const std::vector<std::byte> file = to_bytes(tiny_model());

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::OnnxGraph parsed, read(file));

    ASSERT_EQ(parsed.nodes.size(), 2u);
    EXPECT_EQ(parsed.name, "tiny");
    EXPECT_EQ(parsed.ir_version, 8);
    EXPECT_EQ(parsed.producer_name, "unit-test");
    EXPECT_EQ(parsed.nodes[0].op_type, "Conv");
    EXPECT_EQ(parsed.nodes[0].name, "conv0");
    EXPECT_EQ(parsed.nodes[0].inputs, (std::vector<std::string>{"x", "w"}));
    EXPECT_EQ(parsed.nodes[0].outputs, (std::vector<std::string>{"y"}));
    EXPECT_EQ(parsed.nodes[1].op_type, "Relu");
    EXPECT_EQ(parsed.nodes[1].inputs, (std::vector<std::string>{"y"}));
    EXPECT_EQ(parsed.nodes[1].outputs, (std::vector<std::string>{"z"}));
}

/** @brief Tests that attributes read by their stated type, or the type their payload implies. */
TEST(Small_OnnxGraph, ReadsTheAttributesOfANode)
{
    const std::vector<std::byte> file = to_bytes(tiny_model());

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::OnnxGraph parsed, read(file));

    const Interchange::OnnxNode& conv = parsed.nodes[0];
    ASSERT_EQ(conv.attributes.size(), 4u);
    const Interchange::OnnxAttribute* kernel = conv.find_attribute("kernel_shape");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->type, Interchange::OnnxAttributeType::Ints);
    EXPECT_EQ(kernel->integers, (std::vector<std::int64_t>{1, 1}));
    EXPECT_EQ(conv.find_attribute("group")->type, Interchange::OnnxAttributeType::Int);
    EXPECT_EQ(conv.find_attribute("group")->integer, 1);
    EXPECT_EQ(conv.find_attribute("mode")->type, Interchange::OnnxAttributeType::String);
    EXPECT_EQ(conv.find_attribute("mode")->text, "constant");
    EXPECT_EQ(conv.find_attribute("epsilon")->type, Interchange::OnnxAttributeType::Float);
    EXPECT_EQ(conv.find_attribute("epsilon")->real, 0.5f);
    EXPECT_EQ(conv.find_attribute("absent"), nullptr);
}

/** @brief Tests every attribute type: scalars, lists, a tensor, a list of tensors and subgraphs. */
TEST(Small_OnnxGraph, ReadsAnAttributeOfEveryType)
{
    const std::string inner_node = bytes_field(1, node("Identity", "id0", {"a"}, {"b"}, ""));
    const std::string subgraph = bytes_field(2, "body") + inner_node;
    const std::string one_tensor = tensor("t", {2}, K_FLOAT, bytes_field(9, float_bytes(1.0f) +
                                                                                float_bytes(2.0f)));
    const std::string int32_tensor = tensor("u", {1}, K_INT32, packed_integers(5, {9}));
    const std::string attributes =
        bytes_field(5, attribute("f", float_field(2, 0.5f) + integer_field(20, 1))) +
        bytes_field(5, attribute("i", integer_field(3, -3) + integer_field(20, 2))) +
        bytes_field(5, attribute("s", bytes_field(4, "same") + integer_field(20, 3))) +
        bytes_field(5, attribute("t", bytes_field(5, one_tensor) + integer_field(20, 4))) +
        bytes_field(5, attribute("g", bytes_field(6, subgraph) + integer_field(20, 5))) +
        bytes_field(5, attribute("fs", packed_floats(7, {1.5f, 2.5f}) + integer_field(20, 6))) +
        // Three unpacked varints, as an older exporter writes a repeated field.
        bytes_field(5, attribute("is", integer_field(8, 4) + integer_field(8, 5) +
                                           integer_field(8, 6) + integer_field(20, 7))) +
        bytes_field(5, attribute("ss", bytes_field(9, "a") + bytes_field(9, "bc") +
                                           integer_field(20, 8))) +
        bytes_field(5, attribute("ts", bytes_field(10, int32_tensor) + integer_field(20, 9))) +
        bytes_field(5, attribute("gs", bytes_field(11, subgraph) + bytes_field(11, subgraph) +
                                           integer_field(20, 10)));
    const std::string probe = bytes_field(1, node("Probe", "p0", {}, {"out"}, attributes));
    const std::vector<std::byte> file = to_bytes(model(graph(probe)));

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::OnnxGraph parsed, read(file));

    const Interchange::OnnxNode& probe_node = parsed.nodes[0];
    EXPECT_EQ(probe_node.find_attribute("f")->real, 0.5f);
    EXPECT_EQ(probe_node.find_attribute("i")->integer, -3);
    EXPECT_EQ(probe_node.find_attribute("s")->text, "same");
    const Interchange::OnnxTensor& tensor_value = probe_node.find_attribute("t")->tensor;
    EXPECT_EQ(tensor_value.name, "t");
    EXPECT_EQ(tensor_value.dimensions, (std::vector<std::int64_t>{2}));
    ASSERT_EQ(tensor_value.data.size(), 8u);
    EXPECT_EQ(float_at(tensor_value.data.bytes().data() + 4), 2.0f);
    ASSERT_EQ(probe_node.find_attribute("g")->graphs.size(), 1u);
    EXPECT_EQ(probe_node.find_attribute("g")->graphs[0].name, "body");
    EXPECT_EQ(probe_node.find_attribute("g")->graphs[0].nodes[0].op_type, "Identity");
    EXPECT_EQ(probe_node.find_attribute("fs")->reals, (std::vector<float>{1.5f, 2.5f}));
    EXPECT_EQ(probe_node.find_attribute("is")->integers, (std::vector<std::int64_t>{4, 5, 6}));
    EXPECT_EQ(probe_node.find_attribute("ss")->texts, (std::vector<std::string>{"a", "bc"}));
    ASSERT_EQ(probe_node.find_attribute("ts")->tensors.size(), 1u);
    EXPECT_EQ(probe_node.find_attribute("ts")->tensors[0].data.size(), 4u);
    EXPECT_EQ(probe_node.find_attribute("gs")->graphs.size(), 2u);
    EXPECT_EQ(probe_node.find_attribute("gs")->type, Interchange::OnnxAttributeType::Graphs);
}

/** @brief Tests that a raw_data initializer is a view into the file, at the offset of its bytes. */
TEST(Small_OnnxGraph, ViewsRawInitializerBytesWithoutCopyingThem)
{
    const std::string text = tiny_model();
    const std::vector<std::byte> file = to_bytes(text);
    // The offset of the payload is where the same eight bytes first appear in the file text.
    const std::size_t expected_offset = text.find(K_W_RAW);
    ASSERT_NE(expected_offset, std::string::npos);
    ASSERT_EQ(text.rfind(K_W_RAW), expected_offset);

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::OnnxGraph parsed, read(file));

    const Interchange::OnnxTensor* w = parsed.find_initializer("w");
    ASSERT_NE(w, nullptr);
    EXPECT_TRUE(w->data.is_view());
    EXPECT_EQ(w->data.offset(), expected_offset);
    EXPECT_EQ(w->data.size(), 8u);
    EXPECT_EQ(w->data.bytes().data(), file.data() + expected_offset);
    EXPECT_EQ(w->dimensions, (std::vector<std::int64_t>{2, 1, 1, 1}));
    EXPECT_EQ(w->data_type, Interchange::OnnxDataType::Float);
    EXPECT_EQ(w->element_count(), 2u);
    EXPECT_EQ(float_at(w->data.bytes().data()), 1.5f);
    EXPECT_EQ(float_at(w->data.bytes().data() + 4), -2.0f);
}

/** @brief Tests that typed fields are assembled once into the little-endian layout of the type. */
TEST(Small_OnnxGraph, AssemblesTypedInitializersLittleEndian)
{
    const std::vector<std::byte> file = to_bytes(tiny_model());

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::OnnxGraph parsed, read(file));

    const Interchange::OnnxTensor* b = parsed.find_initializer("b");
    ASSERT_NE(b, nullptr);
    EXPECT_FALSE(b->data.is_view());
    ASSERT_EQ(b->data.size(), 8u);
    EXPECT_EQ(float_at(b->data.bytes().data()), 0.25f);
    EXPECT_EQ(float_at(b->data.bytes().data() + 4), 0.5f);

    const Interchange::OnnxTensor* ids = parsed.find_initializer("ids");
    ASSERT_NE(ids, nullptr);
    // 7, -1 and 300 as three little-endian int64.
    const std::vector<unsigned> expected = {7, 0,    0,    0,    0,    0,    0, 0,
                                            255, 255, 255, 255, 255, 255, 255, 255,
                                            0x2C, 0x01, 0,  0,    0,    0,    0, 0};
    ASSERT_EQ(ids->data.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(std::to_integer<unsigned>(ids->data.bytes()[i]), expected[i]) << "byte " << i;
}

/** @brief Tests that inputs and outputs carry their element type, stated extents and symbols. */
TEST(Small_OnnxGraph, ReadsGraphInputsAndOutputsWithTheirShapes)
{
    const std::vector<std::byte> file = to_bytes(tiny_model());

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::OnnxGraph parsed, read(file));

    ASSERT_EQ(parsed.inputs.size(), 1u);
    const Interchange::OnnxValueInfo& input = parsed.inputs[0];
    EXPECT_EQ(input.name, "x");
    EXPECT_EQ(input.element_type, Interchange::OnnxDataType::Float);
    ASSERT_TRUE(input.has_shape);
    ASSERT_EQ(input.dimensions.size(), 4u);
    EXPECT_EQ(input.dimensions[0].value, -1);
    EXPECT_EQ(input.dimensions[0].parameter, "N");
    EXPECT_EQ(input.dimensions[1].value, 1);
    EXPECT_EQ(input.dimensions[3].value, 4);
    ASSERT_EQ(parsed.outputs.size(), 1u);
    EXPECT_EQ(parsed.outputs[0].name, "z");
    EXPECT_EQ(parsed.outputs[0].dimensions[1].value, 2);
}

/** @brief Tests that operator sets come back by domain, the default under both spellings. */
TEST(Small_OnnxGraph, ReadsTheOpsetImports)
{
    const std::vector<std::byte> file = to_bytes(tiny_model());

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::OnnxGraph parsed, read(file));

    ASSERT_EQ(parsed.opsets.size(), 2u);
    EXPECT_EQ(parsed.opset_version(""), 13);
    EXPECT_EQ(parsed.opset_version("ai.onnx"), 13);
    EXPECT_EQ(parsed.opset_version("com.vendor"), 2);
    EXPECT_EQ(parsed.opset_version("com.other"), 0);
}

/** @brief Tests that a tensor in an external file is refused and named. */
TEST(Small_OnnxGraph, RefusesExternalDataByTheNameOfTheTensor)
{
    const std::string by_location =
        bytes_field(5, tensor("w_loc", {1}, K_FLOAT, integer_field(14, 1)));
    const std::string by_entry = bytes_field(
        5, tensor("w_entry", {1}, K_FLOAT,
                  bytes_field(13, bytes_field(1, "location") + bytes_field(2, "w.bin"))));
    const std::vector<std::byte> located = to_bytes(model(graph(by_location)));
    const std::vector<std::byte> listed = to_bytes(model(graph(by_entry)));

    SA_EXPECT_FAILURE(read(located), StatusCode::Unsupported, "fixture.onnx", "w_loc",
                      "external file");
    SA_EXPECT_FAILURE(read(listed), StatusCode::Unsupported, "w_entry", "external file");
}

/** @brief Tests that a model with no graph field is refused. */
TEST(Small_OnnxGraph, RefusesAModelWithNoGraph)
{
    const std::vector<std::byte> file = to_bytes(integer_field(1, 8));

    SA_EXPECT_FAILURE(read(file), StatusCode::InvalidFile, "fixture.onnx", "no graph");
}

/** @brief Tests that a message cut short fails with InvalidFile naming the file. */
TEST(Small_OnnxGraph, RefusesATruncatedMessage)
{
    std::string text = tiny_model();
    text.resize(text.size() - 3);
    const std::vector<std::byte> file = to_bytes(text);

    SA_EXPECT_FAILURE(read(file), StatusCode::InvalidFile, "fixture.onnx");
}

/** @brief Tests that raw bytes that do not fill the extents are refused with the tensor's name. */
TEST(Small_OnnxGraph, RefusesRawDataThatDoesNotFillTheExtents)
{
    // Three floats need 12 bytes; the field holds 8.
    const std::string bad = bytes_field(5, tensor("w_bad", {3}, K_FLOAT, bytes_field(9, K_W_RAW)));
    const std::vector<std::byte> file = to_bytes(model(graph(bad)));

    SA_EXPECT_FAILURE(read(file), StatusCode::InvalidFile, "w_bad", "8 raw bytes", "need 12");
}

/** @brief Tests that a field of the wrong wire type fails and names the message and the field. */
TEST(Small_OnnxGraph, RefusesAFieldOfTheWrongWireType)
{
    // op_type, field 4 of NodeProto, written as a varint.
    const std::string bad = bytes_field(1, integer_field(4, 3));
    const std::vector<std::byte> file = to_bytes(model(graph(bad)));

    SA_EXPECT_FAILURE(read(file), StatusCode::InvalidFile, "NodeProto", "field 4", "varint",
                      "length-delimited");
}

/** @brief Tests that string tensors and sparse values are refused by name. */
TEST(Small_OnnxGraph, RefusesWhatTheTablesDoNotRead)
{
    const std::string strings = bytes_field(5, tensor("names", {1}, 8, ""));
    const std::string sparse = bytes_field(15, "");
    const std::string sparse_attribute = bytes_field(
        1, node("Probe", "p0", {}, {"o"},
                bytes_field(5, attribute("table", bytes_field(22, "")))));
    const std::vector<std::byte> strings_file = to_bytes(model(graph(strings)));
    const std::vector<std::byte> sparse_file = to_bytes(model(graph(sparse)));
    const std::vector<std::byte> attribute_file = to_bytes(model(graph(sparse_attribute)));

    SA_EXPECT_FAILURE(read(strings_file), StatusCode::Unsupported, "names", "strings");
    SA_EXPECT_FAILURE(read(sparse_file), StatusCode::Unsupported, "sparse initializer");
    SA_EXPECT_FAILURE(read(attribute_file), StatusCode::Unsupported, "table");
}
