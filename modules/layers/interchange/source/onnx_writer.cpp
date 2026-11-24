/**************************************************************************/
/* onnx_writer.cpp                                                        */
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
 * @file onnx_writer.cpp
 * @brief Implements the writer of an ONNX ModelProto.
 * @author Mustafa Garip
 */

#include <SushiAI/interchange/onnx_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/io/protobuf_wire_writer.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        namespace
        {
            using IO::ProtobufWireWriter;

            /** @brief Holds the field numbers of ModelProto the writer writes. */
            enum ModelField : std::uint32_t
            {
                K_MODEL_IR_VERSION = 1,
                K_MODEL_PRODUCER_NAME = 2,
                K_MODEL_PRODUCER_VERSION = 3,
                K_MODEL_GRAPH = 7,
                K_MODEL_OPSET = 8,
            };

            /** @brief Holds the field numbers of GraphProto the writer writes. */
            enum GraphField : std::uint32_t
            {
                K_GRAPH_NODE = 1,
                K_GRAPH_NAME = 2,
                K_GRAPH_INITIALIZER = 5,
                K_GRAPH_INPUT = 11,
                K_GRAPH_OUTPUT = 12,
            };

            /** @brief Holds the field numbers of NodeProto. */
            enum NodeField : std::uint32_t
            {
                K_NODE_INPUT = 1,
                K_NODE_OUTPUT = 2,
                K_NODE_NAME = 3,
                K_NODE_OP_TYPE = 4,
                K_NODE_ATTRIBUTE = 5,
                K_NODE_DOMAIN = 7,
            };

            /** @brief Holds the field numbers of TensorProto the writer writes. */
            enum TensorField : std::uint32_t
            {
                K_TENSOR_DIMS = 1,
                K_TENSOR_DATA_TYPE = 2,
                K_TENSOR_NAME = 8,
                K_TENSOR_RAW_DATA = 9,
            };

            /** @brief Holds the field numbers of AttributeProto the writer writes. */
            enum AttributeField : std::uint32_t
            {
                K_ATTRIBUTE_NAME = 1,
                K_ATTRIBUTE_REAL = 2,
                K_ATTRIBUTE_INTEGER = 3,
                K_ATTRIBUTE_TEXT = 4,
                K_ATTRIBUTE_TENSOR = 5,
                K_ATTRIBUTE_GRAPH = 6,
                K_ATTRIBUTE_REALS = 7,
                K_ATTRIBUTE_INTEGERS = 8,
                K_ATTRIBUTE_TEXTS = 9,
                K_ATTRIBUTE_TENSORS = 10,
                K_ATTRIBUTE_GRAPHS = 11,
                K_ATTRIBUTE_TYPE = 20,
            };

            /** @brief Holds the field numbers of ValueInfoProto, TypeProto and its shape. */
            enum ValueField : std::uint32_t
            {
                K_VALUE_NAME = 1,
                K_VALUE_TYPE = 2,
                K_TYPE_TENSOR = 1,
                K_TENSOR_TYPE_ELEMENT = 1,
                K_TENSOR_TYPE_SHAPE = 2,
                K_SHAPE_DIMENSION = 1,
                K_DIMENSION_VALUE = 1,
                K_DIMENSION_PARAMETER = 2,
                K_OPSET_DOMAIN = 1,
                K_OPSET_VERSION = 2,
            };

            Status write_graph(const OnnxGraph& graph, ProtobufWireWriter& out);

            /** @brief Writes @p tensor as a TensorProto into @p out. */
            Status write_tensor(const OnnxTensor& tensor, ProtobufWireWriter& out)
            {
                const std::size_t width = onnx_element_bytes(tensor.data_type);
                SA_FAIL_IF(width == 0, StatusCode::InvalidDescription,
                           "the tensor '{}' has no element type of fixed width", tensor.name);
                SA_FAIL_IF(tensor.data.size() != tensor.element_count() * width,
                           StatusCode::InvalidDescription,
                           "the tensor '{}' holds {} bytes, but its extents need {}", tensor.name,
                           tensor.data.size(), tensor.element_count() * width);
                if (!tensor.dimensions.empty())
                    out.write_packed_varints(K_TENSOR_DIMS,
                                             SushiRuntime::span<const std::int64_t>(
                                                 tensor.dimensions.data(),
                                                 tensor.dimensions.size()));
                out.write_varint(K_TENSOR_DATA_TYPE,
                                 static_cast<std::uint64_t>(tensor.data_type));
                if (!tensor.name.empty())
                    out.write_string(K_TENSOR_NAME, tensor.name);
                out.write_bytes(K_TENSOR_RAW_DATA, tensor.data.bytes());
                return Status();
            }

            /** @brief Writes @p attribute as an AttributeProto into @p out. */
            Status write_attribute(const OnnxAttribute& attribute, ProtobufWireWriter& out)
            {
                out.write_string(K_ATTRIBUTE_NAME, attribute.name);
                switch (attribute.type)
                {
                    case OnnxAttributeType::Float:
                        out.write_float(K_ATTRIBUTE_REAL, attribute.real);
                        break;
                    case OnnxAttributeType::Int:
                        out.write_varint(K_ATTRIBUTE_INTEGER,
                                         static_cast<std::uint64_t>(attribute.integer));
                        break;
                    case OnnxAttributeType::String:
                        out.write_string(K_ATTRIBUTE_TEXT, attribute.text);
                        break;
                    case OnnxAttributeType::Tensor:
                    {
                        ProtobufWireWriter tensor;
                        SA_TRY(write_tensor(attribute.tensor, tensor));
                        out.write_message(K_ATTRIBUTE_TENSOR, tensor);
                        break;
                    }
                    case OnnxAttributeType::Graph:
                    case OnnxAttributeType::Graphs:
                        for (const OnnxGraph& graph : attribute.graphs)
                        {
                            ProtobufWireWriter nested;
                            SA_TRY(write_graph(graph, nested));
                            out.write_message(attribute.type == OnnxAttributeType::Graph
                                                  ? K_ATTRIBUTE_GRAPH
                                                  : K_ATTRIBUTE_GRAPHS,
                                              nested);
                        }
                        break;
                    case OnnxAttributeType::Floats:
                        out.write_packed_floats(K_ATTRIBUTE_REALS,
                                                SushiRuntime::span<const float>(
                                                    attribute.reals.data(),
                                                    attribute.reals.size()));
                        break;
                    case OnnxAttributeType::Ints:
                        out.write_packed_varints(K_ATTRIBUTE_INTEGERS,
                                                 SushiRuntime::span<const std::int64_t>(
                                                     attribute.integers.data(),
                                                     attribute.integers.size()));
                        break;
                    case OnnxAttributeType::Strings:
                        for (const std::string& text : attribute.texts)
                            out.write_string(K_ATTRIBUTE_TEXTS, text);
                        break;
                    case OnnxAttributeType::Tensors:
                        for (const OnnxTensor& tensor : attribute.tensors)
                        {
                            ProtobufWireWriter nested;
                            SA_TRY(write_tensor(tensor, nested));
                            out.write_message(K_ATTRIBUTE_TENSORS, nested);
                        }
                        break;
                    case OnnxAttributeType::Undefined:
                        SA_FAIL_IF(true, StatusCode::InvalidDescription,
                                   "the attribute '{}' states no type", attribute.name);
                }
                out.write_varint(K_ATTRIBUTE_TYPE, static_cast<std::uint64_t>(attribute.type));
                return Status();
            }

            /** @brief Writes @p info as a ValueInfoProto of a tensor type into @p out. */
            void write_value_info(const OnnxValueInfo& info, ProtobufWireWriter& out)
            {
                ProtobufWireWriter tensor_type;
                tensor_type.write_varint(K_TENSOR_TYPE_ELEMENT,
                                         static_cast<std::uint64_t>(info.element_type));
                if (info.has_shape)
                {
                    ProtobufWireWriter shape;
                    for (const OnnxDimension& dimension : info.dimensions)
                    {
                        ProtobufWireWriter dim;
                        if (!dimension.parameter.empty())
                            dim.write_string(K_DIMENSION_PARAMETER, dimension.parameter);
                        else if (dimension.value >= 0)
                            dim.write_varint(K_DIMENSION_VALUE,
                                             static_cast<std::uint64_t>(dimension.value));
                        shape.write_message(K_SHAPE_DIMENSION, dim);
                    }
                    tensor_type.write_message(K_TENSOR_TYPE_SHAPE, shape);
                }
                ProtobufWireWriter type;
                type.write_message(K_TYPE_TENSOR, tensor_type);
                out.write_string(K_VALUE_NAME, info.name);
                out.write_message(K_VALUE_TYPE, type);
            }

            /** @brief Writes @p graph as a GraphProto into @p out. */
            Status write_graph(const OnnxGraph& graph, ProtobufWireWriter& out)
            {
                for (const OnnxNode& node : graph.nodes)
                {
                    ProtobufWireWriter message;
                    for (const std::string& input : node.inputs)
                        message.write_string(K_NODE_INPUT, input);
                    for (const std::string& output : node.outputs)
                        message.write_string(K_NODE_OUTPUT, output);
                    if (!node.name.empty())
                        message.write_string(K_NODE_NAME, node.name);
                    message.write_string(K_NODE_OP_TYPE, node.op_type);
                    for (const OnnxAttribute& attribute : node.attributes)
                    {
                        ProtobufWireWriter nested;
                        SA_TRY(write_attribute(attribute, nested));
                        message.write_message(K_NODE_ATTRIBUTE, nested);
                    }
                    if (!node.domain.empty())
                        message.write_string(K_NODE_DOMAIN, node.domain);
                    out.write_message(K_GRAPH_NODE, message);
                }
                out.write_string(K_GRAPH_NAME, graph.name);
                for (const OnnxTensor& tensor : graph.initializers)
                {
                    ProtobufWireWriter message;
                    SA_TRY(write_tensor(tensor, message));
                    out.write_message(K_GRAPH_INITIALIZER, message);
                }
                for (const OnnxValueInfo& input : graph.inputs)
                {
                    ProtobufWireWriter message;
                    write_value_info(input, message);
                    out.write_message(K_GRAPH_INPUT, message);
                }
                for (const OnnxValueInfo& output : graph.outputs)
                {
                    ProtobufWireWriter message;
                    write_value_info(output, message);
                    out.write_message(K_GRAPH_OUTPUT, message);
                }
                return Status();
            }
        } // namespace

        Result<std::vector<std::byte>> write_onnx_graph(const OnnxGraph& graph) noexcept
        {
            ProtobufWireWriter model;
            model.write_varint(K_MODEL_IR_VERSION, static_cast<std::uint64_t>(graph.ir_version));
            if (!graph.producer_name.empty())
                model.write_string(K_MODEL_PRODUCER_NAME, graph.producer_name);
            if (!graph.producer_version.empty())
                model.write_string(K_MODEL_PRODUCER_VERSION, graph.producer_version);
            ProtobufWireWriter body;
            SA_TRY(write_graph(graph, body));
            model.write_message(K_MODEL_GRAPH, body);
            for (const OnnxOpset& opset : graph.opsets)
            {
                ProtobufWireWriter message;
                if (!opset.domain.empty())
                    message.write_string(K_OPSET_DOMAIN, opset.domain);
                message.write_varint(K_OPSET_VERSION, static_cast<std::uint64_t>(opset.version));
                model.write_message(K_MODEL_OPSET, message);
            }
            return model.take();
        }
    } // namespace Interchange
} // namespace SushiAI
