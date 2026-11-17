/**************************************************************************/
/* onnx_graph.cpp                                                         */
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
 * @file onnx_graph.cpp
 * @brief Defines the field tables of the ONNX messages and the reader of a model's graph.
 * @author Mustafa Garip
 */

#include <SushiAI/interchange/onnx_graph.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/protobuf_wire.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>
#include "wire_values.hpp"

namespace SushiAI
{
    namespace Interchange
    {
        std::size_t onnx_element_bytes(OnnxDataType type) noexcept
        {
            switch (type)
            {
                case OnnxDataType::Uint8:
                case OnnxDataType::Int8:
                case OnnxDataType::Bool:
                    return 1;
                case OnnxDataType::Uint16:
                case OnnxDataType::Int16:
                case OnnxDataType::Float16:
                case OnnxDataType::Bfloat16:
                    return 2;
                case OnnxDataType::Float:
                case OnnxDataType::Int32:
                case OnnxDataType::Uint32:
                    return 4;
                case OnnxDataType::Int64:
                case OnnxDataType::Uint64:
                case OnnxDataType::Double:
                    return 8;
                case OnnxDataType::Undefined:
                case OnnxDataType::String:
                    return 0;
            }
            return 0;
        }

        std::size_t OnnxTensor::element_count() const noexcept
        {
            std::size_t count = 1;
            for (std::int64_t extent : dimensions)
                count *= static_cast<std::size_t>(extent);
            return count;
        }

        const OnnxAttribute*
        OnnxNode::find_attribute(std::string_view attribute_name) const noexcept
        {
            for (const OnnxAttribute& attribute : attributes)
                if (attribute.name == attribute_name)
                    return &attribute;
            return nullptr;
        }

        const OnnxTensor* OnnxGraph::find_initializer(std::string_view tensor_name) const noexcept
        {
            for (const OnnxTensor& tensor : initializers)
                if (tensor.name == tensor_name)
                    return &tensor;
            return nullptr;
        }

        std::int64_t OnnxGraph::opset_version(std::string_view domain) const noexcept
        {
            const bool default_domain = domain.empty() || domain == "ai.onnx";
            for (const OnnxOpset& opset : opsets)
            {
                const bool same = default_domain
                                      ? (opset.domain.empty() || opset.domain == "ai.onnx")
                                      : opset.domain == domain;
                if (same)
                    return opset.version;
            }
            return 0;
        }

        namespace
        {
            using Detail::BytesField;
            using Detail::FieldSite;

            /** @brief Holds the deepest nesting of subgraphs the reader accepts. */
            constexpr std::size_t K_MAX_GRAPH_DEPTH = 16;

            /** @brief Holds the value of TensorProto.data_location that marks external data. */
            constexpr std::int64_t K_DATA_LOCATION_EXTERNAL = 1;

            /** @brief Names the fields of ModelProto that the reader reads. */
            enum class ModelField : std::uint32_t
            {
                IRVersion       = 1, /**< ir_version, int64. */
                ProducerName    = 2, /**< producer_name, string. */
                ProducerVersion = 3, /**< producer_version, string. */
                Graph           = 7, /**< graph, GraphProto. */
                OpsetImport     = 8, /**< opset_import, repeated OperatorSetIdProto. */
            };

            /** @brief Names the fields of OperatorSetIdProto. */
            enum class OpsetField : std::uint32_t
            {
                Domain  = 1, /**< domain, string. */
                Version = 2, /**< version, int64. */
            };

            /** @brief Names the fields of GraphProto that the reader reads or refuses. */
            enum class GraphField : std::uint32_t
            {
                Node              = 1,  /**< node, repeated NodeProto. */
                Name              = 2,  /**< name, string. */
                Initializer       = 5,  /**< initializer, repeated TensorProto. */
                Input             = 11, /**< input, repeated ValueInfoProto. */
                Output            = 12, /**< output, repeated ValueInfoProto. */
                SparseInitializer = 15, /**< sparse_initializer, refused. */
            };

            /** @brief Names the fields of NodeProto. */
            enum class NodeField : std::uint32_t
            {
                Input     = 1, /**< input, repeated string. */
                Output    = 2, /**< output, repeated string. */
                Name      = 3, /**< name, string. */
                OpType    = 4, /**< op_type, string. */
                Attribute = 5, /**< attribute, repeated AttributeProto. */
                Domain    = 7, /**< domain, string. */
            };

            /** @brief Names the fields of TensorProto. */
            enum class TensorField : std::uint32_t
            {
                Dims         = 1,  /**< dims, repeated int64. */
                DataType     = 2,  /**< data_type, int32. */
                FloatData    = 4,  /**< float_data, repeated float. */
                Int32Data    = 5,  /**< int32_data, repeated int32. */
                StringData   = 6,  /**< string_data, repeated bytes; refused. */
                Int64Data    = 7,  /**< int64_data, repeated int64. */
                Name         = 8,  /**< name, string. */
                RawData      = 9,  /**< raw_data, bytes. */
                DoubleData   = 10, /**< double_data, repeated double. */
                Uint64Data   = 11, /**< uint64_data, repeated uint64. */
                ExternalData = 13, /**< external_data, repeated StringStringEntryProto; refused. */
                DataLocation = 14, /**< data_location, DataLocation. */
            };

            /** @brief Names the fields of AttributeProto that the reader reads or refuses. */
            enum class AttributeField : std::uint32_t
            {
                Name          = 1,  /**< name, string. */
                Real          = 2,  /**< f, float. */
                Integer       = 3,  /**< i, int64. */
                Text          = 4,  /**< s, bytes. */
                Tensor        = 5,  /**< t, TensorProto. */
                Graph         = 6,  /**< g, GraphProto. */
                Reals         = 7,  /**< floats, repeated float. */
                Integers      = 8,  /**< ints, repeated int64. */
                Texts         = 9,  /**< strings, repeated bytes. */
                Tensors       = 10, /**< tensors, repeated TensorProto. */
                Graphs        = 11, /**< graphs, repeated GraphProto. */
                TypeProto     = 14, /**< tp, refused. */
                TypeProtos    = 15, /**< type_protos, refused. */
                Type          = 20, /**< type, AttributeType. */
                SparseTensor  = 22, /**< sparse_tensor, refused. */
                SparseTensors = 23, /**< sparse_tensors, refused. */
            };

            /** @brief Names the fields of ValueInfoProto. */
            enum class ValueInfoField : std::uint32_t
            {
                Name = 1, /**< name, string. */
                Type = 2, /**< type, TypeProto. */
            };

            /** @brief Names the field of TypeProto the reader reads; other kinds are skipped. */
            enum class TypeField : std::uint32_t
            {
                Tensor = 1, /**< tensor_type, TypeProto.Tensor. */
            };

            /** @brief Names the fields of TypeProto.Tensor. */
            enum class TypeTensorField : std::uint32_t
            {
                ElementType = 1, /**< elem_type, int32. */
                Shape       = 2, /**< shape, TensorShapeProto. */
            };

            /** @brief Names the field of TensorShapeProto. */
            enum class ShapeField : std::uint32_t
            {
                Dimension = 1, /**< dim, repeated Dimension. */
            };

            /** @brief Names the fields of TensorShapeProto.Dimension. */
            enum class DimensionField : std::uint32_t
            {
                Value     = 1, /**< dim_value, int64. */
                Parameter = 2, /**< dim_param, string. */
            };

            /** @brief Appends the low @p width bytes of @p value to @p out, low byte first. */
            void put_little_endian(std::vector<std::byte>& out, std::uint64_t value,
                                   std::size_t width)
            {
                for (std::size_t i = 0; i < width; ++i)
                    out.push_back(static_cast<std::byte>((value >> (8u * i)) & 0xFFu));
            }

            /** @brief Returns the product of @p dimensions, or fails when one is negative or the
             *         product overflows. */
            Result<std::size_t> checked_count(const std::vector<std::int64_t>& dimensions,
                                              const FieldSite& site, std::string_view tensor_name)
                noexcept
            {
                std::size_t count = 1;
                for (std::int64_t extent : dimensions)
                {
                    SA_FAIL_IF(extent < 0, StatusCode::InvalidFile,
                               "'{}': tensor '{}' has the negative extent {}", site.origin,
                               tensor_name, extent);
                    const std::size_t unsigned_extent = static_cast<std::size_t>(extent);
                    SA_FAIL_IF(unsigned_extent != 0 &&
                                   count > std::numeric_limits<std::size_t>::max() /
                                               unsigned_extent,
                               StatusCode::InvalidFile,
                               "'{}': tensor '{}' has more elements than fit a size", site.origin,
                               tensor_name);
                    count *= unsigned_extent;
                }
                return count;
            }

            /** @brief Holds the typed fields of one tensor before they are assembled. */
            struct TypedData
            {
                std::vector<float> floats;          /**< float_data. */
                std::vector<double> doubles;        /**< double_data. */
                std::vector<std::int64_t> int32s;   /**< int32_data. */
                std::vector<std::int64_t> int64s;   /**< int64_data. */
                std::vector<std::int64_t> uint64s;  /**< uint64_data, as bits. */
            };

            /** @brief Returns the typed fields in the little-endian layout of @p width. */
            std::vector<std::byte> assemble_typed(const TypedData& typed, std::size_t width)
            {
                std::vector<std::byte> out;
                for (float value : typed.floats)
                {
                    std::uint32_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    put_little_endian(out, bits, width);
                }
                for (double value : typed.doubles)
                {
                    std::uint64_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    put_little_endian(out, bits, width);
                }
                for (std::int64_t value : typed.int32s)
                    put_little_endian(out, static_cast<std::uint64_t>(value), width);
                for (std::int64_t value : typed.int64s)
                    put_little_endian(out, static_cast<std::uint64_t>(value), width);
                for (std::int64_t value : typed.uint64s)
                    put_little_endian(out, static_cast<std::uint64_t>(value), width);
                return out;
            }

            /** @brief Reads one TensorProto into @p out. */
            Status read_tensor(IO::ProtobufWire reader, std::string_view origin, OnnxTensor& out)
            {
                const FieldSite site{origin, "TensorProto"};
                TypedData typed;
                BytesField raw;
                bool has_raw = false;
                bool external = false;
                bool has_strings = false;
                std::int64_t data_type = 0;

                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    switch (static_cast<TensorField>(field.number))
                    {
                        case TensorField::Dims:
                            SA_TRY(Detail::read_integers(reader, field, site, out.dimensions));
                            break;
                        case TensorField::DataType:
                        {
                                SA_TRY_ASSIGN(data_type, Detail::read_integer(reader, field, site));
                                break;
                        }
                        case TensorField::FloatData:
                            SA_TRY(Detail::read_reals(reader, field, site, typed.floats));
                            break;
                        case TensorField::Int32Data:
                            SA_TRY(Detail::read_integers(reader, field, site, typed.int32s));
                            break;
                        case TensorField::StringData:
                            has_strings = true;
                            SA_TRY(reader.skip(field.type));
                            break;
                        case TensorField::Int64Data:
                            SA_TRY(Detail::read_integers(reader, field, site, typed.int64s));
                            break;
                        case TensorField::Name:
                        {
                                SA_TRY_ASSIGN(out.name, Detail::read_text(reader, field, site));
                                break;
                        }
                        case TensorField::RawData:
                        {
                                SA_TRY_ASSIGN(raw, Detail::read_bytes(reader, field, site));
                                has_raw = true;
                                break;
                        }
                        case TensorField::DoubleData:
                            SA_TRY(Detail::read_doubles(reader, field, site, typed.doubles));
                            break;
                        case TensorField::Uint64Data:
                            SA_TRY(Detail::read_integers(reader, field, site, typed.uint64s));
                            break;
                        case TensorField::ExternalData:
                            external = true;
                            SA_TRY(reader.skip(field.type));
                            break;
                        case TensorField::DataLocation:
                        {
                            SA_TRY_ASSIGN(const std::int64_t location,
                                          Detail::read_integer(reader, field, site));
                            external = external || location == K_DATA_LOCATION_EXTERNAL;
                            break;
                        }
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }

                SA_FAIL_IF(external, StatusCode::Unsupported,
                           "'{}': tensor '{}' keeps its data in an external file, which the "
                           "importer does not read", origin, out.name);
                SA_FAIL_IF(has_strings ||
                               data_type == static_cast<std::int64_t>(OnnxDataType::String),
                           StatusCode::Unsupported, "'{}': tensor '{}' holds strings", origin,
                           out.name);
                out.data_type = static_cast<OnnxDataType>(data_type);
                const std::size_t width = onnx_element_bytes(out.data_type);
                SA_FAIL_IF(width == 0, StatusCode::InvalidFile,
                           "'{}': tensor '{}' has the element type {}, which has no width", origin,
                           out.name, data_type);

                SA_TRY_ASSIGN(const std::size_t count,
                              checked_count(out.dimensions, site, out.name));
                if (has_raw)
                {
                    SA_FAIL_IF(raw.bytes.size() != count * width, StatusCode::InvalidFile,
                               "'{}': tensor '{}' holds {} raw bytes, its extents and type need {}",
                               origin, out.name, raw.bytes.size(), count * width);
                    out.data = TensorBytes::viewing(raw.bytes, raw.offset);
                    return Status();
                }
                std::vector<std::byte> assembled = assemble_typed(typed, width);
                SA_FAIL_IF(!assembled.empty() && assembled.size() != count * width,
                           StatusCode::InvalidFile,
                           "'{}': tensor '{}' holds {} bytes in its typed fields, its extents and "
                           "type need {}", origin, out.name, assembled.size(), count * width);
                out.data = TensorBytes::owning(std::move(assembled));
                return Status();
            }

            /** @brief Reads one GraphProto into @p out, with subgraphs nested to @p depth. */
            Status read_graph(IO::ProtobufWire reader, std::string_view origin, std::size_t depth,
                              OnnxGraph& out);

            /** @brief Returns the attribute type a payload field implies. */
            OnnxAttributeType implied_type(AttributeField field) noexcept
            {
                switch (field)
                {
                    case AttributeField::Real:     return OnnxAttributeType::Float;
                    case AttributeField::Integer:  return OnnxAttributeType::Int;
                    case AttributeField::Text:     return OnnxAttributeType::String;
                    case AttributeField::Tensor:   return OnnxAttributeType::Tensor;
                    case AttributeField::Graph:    return OnnxAttributeType::Graph;
                    case AttributeField::Reals:    return OnnxAttributeType::Floats;
                    case AttributeField::Integers: return OnnxAttributeType::Ints;
                    case AttributeField::Texts:    return OnnxAttributeType::Strings;
                    case AttributeField::Tensors:  return OnnxAttributeType::Tensors;
                    case AttributeField::Graphs:   return OnnxAttributeType::Graphs;
                    default:                       return OnnxAttributeType::Undefined;
                }
            }

            /** @brief Reads one AttributeProto into @p out. */
            Status read_attribute(IO::ProtobufWire reader, std::string_view origin,
                                  std::size_t depth, OnnxAttribute& out)
            {
                const FieldSite site{origin, "AttributeProto"};
                OnnxAttributeType stated = OnnxAttributeType::Undefined;
                OnnxAttributeType implied = OnnxAttributeType::Undefined;
                bool refused = false;

                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    const AttributeField which = static_cast<AttributeField>(field.number);
                    const OnnxAttributeType payload = implied_type(which);
                    if (payload != OnnxAttributeType::Undefined)
                        implied = payload;

                    switch (which)
                    {
                        case AttributeField::Name:
                        {
                                SA_TRY_ASSIGN(out.name, Detail::read_text(reader, field, site));
                                break;
                        }
                        case AttributeField::Real:
                        {
                                SA_TRY_ASSIGN(out.real, Detail::read_real(reader, field, site));
                                break;
                        }
                        case AttributeField::Integer:
                        {
                                SA_TRY_ASSIGN(out.integer,
                                              Detail::read_integer(reader, field, site));
                                break;
                        }
                        case AttributeField::Text:
                        {
                                SA_TRY_ASSIGN(out.text, Detail::read_text(reader, field, site));
                                break;
                        }
                        case AttributeField::Tensor:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            SA_TRY(read_tensor(inner, origin, out.tensor));
                            break;
                        }
                        case AttributeField::Graph:
                        case AttributeField::Graphs:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            OnnxGraph graph;
                            SA_TRY(read_graph(inner, origin, depth + 1, graph));
                            out.graphs.push_back(std::move(graph));
                            break;
                        }
                        case AttributeField::Reals:
                            SA_TRY(Detail::read_reals(reader, field, site, out.reals));
                            break;
                        case AttributeField::Integers:
                            SA_TRY(Detail::read_integers(reader, field, site, out.integers));
                            break;
                        case AttributeField::Texts:
                        {
                            SA_TRY_ASSIGN(std::string text, Detail::read_text(reader, field, site));
                            out.texts.push_back(std::move(text));
                            break;
                        }
                        case AttributeField::Tensors:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            OnnxTensor tensor;
                            SA_TRY(read_tensor(inner, origin, tensor));
                            out.tensors.push_back(std::move(tensor));
                            break;
                        }
                        case AttributeField::Type:
                        {
                            SA_TRY_ASSIGN(const std::int64_t value,
                                          Detail::read_integer(reader, field, site));
                            stated = static_cast<OnnxAttributeType>(value);
                            break;
                        }
                        case AttributeField::TypeProto:
                        case AttributeField::TypeProtos:
                        case AttributeField::SparseTensor:
                        case AttributeField::SparseTensors:
                            refused = true;
                            SA_TRY(reader.skip(field.type));
                            break;
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }

                SA_FAIL_IF(refused, StatusCode::Unsupported,
                           "'{}': attribute '{}' holds a sparse tensor or a type, which the "
                           "importer does not read", origin, out.name);
                out.type = stated != OnnxAttributeType::Undefined ? stated : implied;
                return Status();
            }

            /** @brief Reads one NodeProto into @p out. */
            Status read_node(IO::ProtobufWire reader, std::string_view origin, std::size_t depth,
                             OnnxNode& out)
            {
                const FieldSite site{origin, "NodeProto"};
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    switch (static_cast<NodeField>(field.number))
                    {
                        case NodeField::Input:
                        {
                            SA_TRY_ASSIGN(std::string name, Detail::read_text(reader, field, site));
                            out.inputs.push_back(std::move(name));
                            break;
                        }
                        case NodeField::Output:
                        {
                            SA_TRY_ASSIGN(std::string name, Detail::read_text(reader, field, site));
                            out.outputs.push_back(std::move(name));
                            break;
                        }
                        case NodeField::Name:
                        {
                                SA_TRY_ASSIGN(out.name, Detail::read_text(reader, field, site));
                                break;
                        }
                        case NodeField::OpType:
                        {
                                SA_TRY_ASSIGN(out.op_type, Detail::read_text(reader, field, site));
                                break;
                        }
                        case NodeField::Attribute:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            OnnxAttribute attribute;
                            SA_TRY(read_attribute(inner, origin, depth, attribute));
                            out.attributes.push_back(std::move(attribute));
                            break;
                        }
                        case NodeField::Domain:
                        {
                                SA_TRY_ASSIGN(out.domain, Detail::read_text(reader, field, site));
                                break;
                        }
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }
                return Status();
            }

            /** @brief Reads one TensorShapeProto.Dimension into @p out. */
            Status read_dimension(IO::ProtobufWire reader, const FieldSite& site,
                                  OnnxDimension& out)
            {
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    switch (static_cast<DimensionField>(field.number))
                    {
                        case DimensionField::Value:
                        {
                                SA_TRY_ASSIGN(out.value, Detail::read_integer(reader, field, site));
                                break;
                        }
                        case DimensionField::Parameter:
                        {
                                SA_TRY_ASSIGN(out.parameter,
                                              Detail::read_text(reader, field, site));
                                break;
                        }
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }
                return Status();
            }

            /** @brief Reads one TensorShapeProto into the dimensions of @p out. */
            Status read_shape(IO::ProtobufWire reader, const FieldSite& site, OnnxValueInfo& out)
            {
                out.has_shape = true;
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    if (static_cast<ShapeField>(field.number) != ShapeField::Dimension)
                    {
                        SA_TRY(reader.skip(field.type));
                        continue;
                    }
                    SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                  Detail::read_submessage(reader, field, site));
                    OnnxDimension dimension;
                    SA_TRY(read_dimension(inner, site, dimension));
                    out.dimensions.push_back(std::move(dimension));
                }
                return Status();
            }

            /** @brief Reads TypeProto.Tensor into the element type and shape of @p out. */
            Status read_tensor_type(IO::ProtobufWire reader, const FieldSite& site,
                                    OnnxValueInfo& out)
            {
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    switch (static_cast<TypeTensorField>(field.number))
                    {
                        case TypeTensorField::ElementType:
                        {
                            SA_TRY_ASSIGN(const std::int64_t value,
                                          Detail::read_integer(reader, field, site));
                            out.element_type = static_cast<OnnxDataType>(value);
                            break;
                        }
                        case TypeTensorField::Shape:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            SA_TRY(read_shape(inner, site, out));
                            break;
                        }
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }
                return Status();
            }

            /** @brief Reads one ValueInfoProto into @p out. */
            Status read_value_info(IO::ProtobufWire reader, std::string_view origin,
                                   OnnxValueInfo& out)
            {
                const FieldSite site{origin, "ValueInfoProto"};
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    switch (static_cast<ValueInfoField>(field.number))
                    {
                        case ValueInfoField::Name:
                        {
                                SA_TRY_ASSIGN(out.name, Detail::read_text(reader, field, site));
                                break;
                        }
                        case ValueInfoField::Type:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire type,
                                          Detail::read_submessage(reader, field, site));
                            const FieldSite type_site{origin, "TypeProto"};
                            while (!type.at_end())
                            {
                                SA_TRY_ASSIGN(const IO::WireField kind, type.next_field());
                                if (static_cast<TypeField>(kind.number) != TypeField::Tensor)
                                {
                                    SA_TRY(type.skip(kind.type));
                                    continue;
                                }
                                SA_TRY_ASSIGN(IO::ProtobufWire tensor_type,
                                              Detail::read_submessage(type, kind, type_site));
                                SA_TRY(read_tensor_type(tensor_type, type_site, out));
                            }
                            break;
                        }
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }
                return Status();
            }

            Status read_graph(IO::ProtobufWire reader, std::string_view origin, std::size_t depth,
                              OnnxGraph& out)
            {
                const FieldSite site{origin, "GraphProto"};
                SA_FAIL_IF(depth > K_MAX_GRAPH_DEPTH, StatusCode::InvalidFile,
                           "'{}': subgraphs nest deeper than {}", origin, K_MAX_GRAPH_DEPTH);
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    switch (static_cast<GraphField>(field.number))
                    {
                        case GraphField::Node:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            OnnxNode node;
                            SA_TRY(read_node(inner, origin, depth, node));
                            out.nodes.push_back(std::move(node));
                            break;
                        }
                        case GraphField::Name:
                        {
                                SA_TRY_ASSIGN(out.name, Detail::read_text(reader, field, site));
                                break;
                        }
                        case GraphField::Initializer:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            OnnxTensor tensor;
                            SA_TRY(read_tensor(inner, origin, tensor));
                            out.initializers.push_back(std::move(tensor));
                            break;
                        }
                        case GraphField::Input:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            OnnxValueInfo info;
                            SA_TRY(read_value_info(inner, origin, info));
                            out.inputs.push_back(std::move(info));
                            break;
                        }
                        case GraphField::Output:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            OnnxValueInfo info;
                            SA_TRY(read_value_info(inner, origin, info));
                            out.outputs.push_back(std::move(info));
                            break;
                        }
                        case GraphField::SparseInitializer:
                            SA_FAIL_IF(true, StatusCode::Unsupported,
                                       "'{}': the graph '{}' holds a sparse initializer, which the "
                                       "importer does not read", origin, out.name);
                            break;
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }
                return Status();
            }

            /** @brief Reads one OperatorSetIdProto into @p out. */
            Status read_opset(IO::ProtobufWire reader, std::string_view origin, OnnxOpset& out)
            {
                const FieldSite site{origin, "OperatorSetIdProto"};
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    switch (static_cast<OpsetField>(field.number))
                    {
                        case OpsetField::Domain:
                        {
                                SA_TRY_ASSIGN(out.domain, Detail::read_text(reader, field, site));
                                break;
                        }
                        case OpsetField::Version:
                        {
                                SA_TRY_ASSIGN(out.version,
                                              Detail::read_integer(reader, field, site));
                                break;
                        }
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }
                return Status();
            }
        } // namespace

        Result<OnnxGraph> read_onnx_graph(SushiRuntime::span<const std::byte> bytes,
                                          std::string_view origin) noexcept
        {
            const FieldSite site{origin, "ModelProto"};
            IO::ProtobufWire reader(bytes, origin);
            OnnxGraph out;
            bool has_graph = false;

            while (!reader.at_end())
            {
                SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                switch (static_cast<ModelField>(field.number))
                {
                    case ModelField::IRVersion:
                    {
                            SA_TRY_ASSIGN(out.ir_version,
                                          Detail::read_integer(reader, field, site));
                            break;
                    }
                    case ModelField::ProducerName:
                    {
                            SA_TRY_ASSIGN(out.producer_name,
                                          Detail::read_text(reader, field, site));
                            break;
                    }
                    case ModelField::ProducerVersion:
                    {
                            SA_TRY_ASSIGN(out.producer_version,
                                          Detail::read_text(reader, field, site));
                            break;
                    }
                    case ModelField::Graph:
                    {
                        SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                      Detail::read_submessage(reader, field, site));
                        has_graph = true;
                        // The model's own fields were read above; the graph fills the rest.
                        SA_TRY(read_graph(inner, origin, 0, out));
                        break;
                    }
                    case ModelField::OpsetImport:
                    {
                        SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                      Detail::read_submessage(reader, field, site));
                        OnnxOpset opset;
                        SA_TRY(read_opset(inner, origin, opset));
                        out.opsets.push_back(std::move(opset));
                        break;
                    }
                    default:
                        SA_TRY(reader.skip(field.type));
                        break;
                }
            }

            SA_FAIL_IF(!has_graph, StatusCode::InvalidFile,
                       "'{}': the ModelProto holds no graph", origin);
            return out;
        }
    } // namespace Interchange
} // namespace SushiAI
