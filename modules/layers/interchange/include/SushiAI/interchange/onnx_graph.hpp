/**************************************************************************/
/* onnx_graph.hpp                                                         */
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
 * @file onnx_graph.hpp
 * @brief Declares the graph of an ONNX model and the reader that builds it from the file's bytes.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        /** @brief Names the element type of an ONNX tensor, by the number TensorProto gives it. */
        enum class OnnxDataType : std::int32_t
        {
            Undefined = 0,  /**< No type stated. */
            Float     = 1,  /**< IEEE 754 binary32. */
            Uint8     = 2,  /**< Unsigned 8-bit integer. */
            Int8      = 3,  /**< Signed 8-bit integer. */
            Uint16    = 4,  /**< Unsigned 16-bit integer. */
            Int16     = 5,  /**< Signed 16-bit integer. */
            Int32     = 6,  /**< Signed 32-bit integer. */
            Int64     = 7,  /**< Signed 64-bit integer. */
            String    = 8,  /**< Byte strings; refused as a tensor. */
            Bool      = 9,  /**< One byte, 0 or 1. */
            Float16   = 10, /**< IEEE 754 binary16. */
            Double    = 11, /**< IEEE 754 binary64. */
            Uint32    = 12, /**< Unsigned 32-bit integer. */
            Uint64    = 13, /**< Unsigned 64-bit integer. */
            Bfloat16  = 16, /**< Brain floating point, 16 bits. */
        };

        /** @brief Returns the bytes of one element of @p type, or 0 when it has no fixed width. */
        [[nodiscard]] std::size_t onnx_element_bytes(OnnxDataType type) noexcept;

        /** @brief Holds one dimension of a declared shape. */
        struct OnnxDimension
        {
            /** @brief Holds the extent, or -1 when the dimension is symbolic or unstated. */
            std::int64_t value = -1;

            /** @brief Holds the symbol of a symbolic dimension, such as "batch"; else empty. */
            std::string parameter;
        };

        /** @brief Holds a graph input or output: its name, element type and declared shape. */
        struct OnnxValueInfo
        {
            std::string name;                                    /**< The tensor's name. */
            OnnxDataType element_type = OnnxDataType::Undefined; /**< Its element type. */
            bool has_shape = false;               /**< Whether the file states a shape. */
            std::vector<OnnxDimension> dimensions; /**< The dimensions, when has_shape. */
        };

        /** @brief Holds one tensor: an initializer, or the value of a node's attribute. */
        struct OnnxTensor
        {
            std::string name;                                 /**< Its name; may be empty. */
            std::vector<std::int64_t> dimensions;            /**< Its extents; none for a scalar. */
            OnnxDataType data_type = OnnxDataType::Undefined; /**< Its element type. */
            TensorBytes data; /**< Its elements, row-major, little-endian, in the type's layout. */

            /** @brief Returns the product of the extents, 1 for a scalar. */
            [[nodiscard]] std::size_t element_count() const noexcept;
        };

        /** @brief Names the kind of value an attribute holds, by its AttributeProto number. */
        enum class OnnxAttributeType : std::int32_t
        {
            Undefined = 0,  /**< No value. */
            Float     = 1,  /**< One float, in real. */
            Int       = 2,  /**< One integer, in integer. */
            String    = 3,  /**< One string, in text. */
            Tensor    = 4,  /**< One tensor, in tensor. */
            Graph     = 5,  /**< One subgraph, in graphs. */
            Floats    = 6,  /**< A list of floats, in reals. */
            Ints      = 7,  /**< A list of integers, in integers. */
            Strings   = 8,  /**< A list of strings, in texts. */
            Tensors   = 9,  /**< A list of tensors, in tensors. */
            Graphs    = 10, /**< A list of subgraphs, in graphs. */
        };

        struct OnnxGraph;

        /** @brief Holds one attribute of a node; the member its type names holds the value. */
        struct OnnxAttribute
        {
            std::string name;                                       /**< The attribute's name. */
            OnnxAttributeType type = OnnxAttributeType::Undefined;  /**< What it holds. */
            float real = 0.0f;                                      /**< The value of a Float. */
            std::int64_t integer = 0;                               /**< The value of an Int. */
            std::string text;                                       /**< The value of a String. */
            OnnxTensor tensor;                                      /**< The value of a Tensor. */
            std::vector<float> reals;                               /**< The values of Floats. */
            std::vector<std::int64_t> integers;                     /**< The values of Ints. */
            std::vector<std::string> texts;                         /**< The values of Strings. */
            std::vector<OnnxTensor> tensors;                        /**< The values of Tensors. */
            std::vector<OnnxGraph> graphs; /**< The one Graph, or the values of Graphs. */
        };

        /** @brief Holds one operator application of the graph. */
        struct OnnxNode
        {
            std::string name;                       /**< The node's name; may be empty. */
            std::string op_type;                    /**< The operator, such as "Conv". */
            std::string domain;                     /**< The operator domain; empty is ai.onnx. */
            std::vector<std::string> inputs;        /**< The tensors it reads; "" is absent. */
            std::vector<std::string> outputs;       /**< The tensors it writes. */
            std::vector<OnnxAttribute> attributes;  /**< Its attributes, in file order. */

            /** @brief Returns the attribute named @p attribute_name, or null when absent. */
            [[nodiscard]] const OnnxAttribute*
            find_attribute(std::string_view attribute_name) const noexcept;
        };

        /** @brief Holds one operator set the model imports. */
        struct OnnxOpset
        {
            std::string domain;          /**< The domain; empty is ai.onnx. */
            std::int64_t version = 0;    /**< The version of the set. */
        };

        /** @brief Holds the graph of an ONNX model, or a subgraph held by an attribute. */
        struct OnnxGraph
        {
            std::string name;                         /**< The graph's name. */
            std::int64_t ir_version = 0;              /**< The IR version; 0 in a subgraph. */
            std::string producer_name;                /**< The exporter; empty in a subgraph. */
            std::string producer_version;             /**< The exporter's version. */
            std::vector<OnnxOpset> opsets;            /**< The operator sets; none in a subgraph. */
            std::vector<OnnxNode> nodes;              /**< The nodes, in file order. */
            std::vector<OnnxTensor> initializers;     /**< The stored tensors. */
            std::vector<OnnxValueInfo> inputs;        /**< The inputs, initializers included. */
            std::vector<OnnxValueInfo> outputs;       /**< The outputs. */

            /** @brief Returns the initializer named @p tensor_name, or null when absent. */
            [[nodiscard]] const OnnxTensor*
            find_initializer(std::string_view tensor_name) const noexcept;

            /** @brief Returns the version of the operator set of @p domain, or 0 when unstated. */
            [[nodiscard]] std::int64_t opset_version(std::string_view domain) const noexcept;
        };

        /**
         * @brief Reads the ModelProto in @p bytes into its graph.
         * @param bytes The file; must outlive the graph, whose raw_data tensors view it.
         * @param origin Names the file in a message.
         * @return InvalidFile for a malformed message, no graph or bytes that do not fill a
         *         tensor's extents; Unsupported, naming it, for external data, a string
         *         tensor or a sparse value.
         */
        [[nodiscard]] Result<OnnxGraph> read_onnx_graph(SushiRuntime::span<const std::byte> bytes,
                                                        std::string_view origin) noexcept;
    } // namespace Interchange
} // namespace SushiAI
