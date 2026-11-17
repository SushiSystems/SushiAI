/**************************************************************************/
/* caffe_net.cpp                                                          */
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
 * @file caffe_net.cpp
 * @brief Defines the field tables of the Caffe messages and the readers of a net's two files.
 * @author Mustafa Garip
 */

#include <SushiAI/interchange/caffe_net.hpp>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include <SushiAI/io/protobuf_wire.hpp>
#include "caffe_v1_types.hpp"
#include "wire_values.hpp"

namespace SushiAI
{
    namespace Interchange
    {
        std::size_t CaffeBlob::element_count() const noexcept
        {
            return data.size() / sizeof(float);
        }

        std::size_t CaffeNet::count_layers(std::string_view layer_name) const noexcept
        {
            std::size_t count = 0;
            for (const CaffeLayer& layer : layers)
                if (layer.name == layer_name)
                    ++count;
            return count;
        }

        const CaffeLayer* CaffeNet::find_layer(std::string_view layer_name) const noexcept
        {
            for (const CaffeLayer& layer : layers)
                if (layer.name == layer_name)
                    return &layer;
            return nullptr;
        }

        namespace
        {
            using Detail::BytesField;
            using Detail::FieldSite;

            /** @brief Holds the bytes of one float32. */
            constexpr std::size_t K_FLOAT_BYTES = sizeof(float);

            /** @brief Holds the extents a legacy blob states: num, channels, height, width. */
            constexpr std::size_t K_LEGACY_EXTENTS = 4;

            /** @brief Names the fields of NetParameter that the binary reader reads. */
            enum class NetField : std::uint32_t
            {
                Name     = 1,   /**< name, string. */
                V1Layers = 2,   /**< layers, repeated V1LayerParameter. */
                Layers   = 100, /**< layer, repeated LayerParameter. */
            };

            /** @brief Names the fields of LayerParameter that the binary reader reads. */
            enum class LayerField : std::uint32_t
            {
                Name   = 1, /**< name, string. */
                Type   = 2, /**< type, string. */
                Bottom = 3, /**< bottom, repeated string. */
                Top    = 4, /**< top, repeated string. */
                Blobs  = 7, /**< blobs, repeated BlobProto. */
            };

            /** @brief Names the fields of V1LayerParameter that the binary reader reads. */
            enum class V1LayerField : std::uint32_t
            {
                Bottom = 2, /**< bottom, repeated string. */
                Top    = 3, /**< top, repeated string. */
                Name   = 4, /**< name, string. */
                Type   = 5, /**< type, LayerType. */
                Blobs  = 6, /**< blobs, repeated BlobProto. */
            };

            /** @brief Names the fields of BlobProto. */
            enum class BlobField : std::uint32_t
            {
                Num        = 1, /**< num, int32, a legacy extent. */
                Channels   = 2, /**< channels, int32, a legacy extent. */
                Height     = 3, /**< height, int32, a legacy extent. */
                Width      = 4, /**< width, int32, a legacy extent. */
                Data       = 5, /**< data, repeated float. */
                Shape      = 7, /**< shape, BlobShape. */
                DoubleData = 8, /**< double_data, repeated double; refused. */
            };

            /** @brief Names the field of BlobShape. */
            enum class BlobShapeField : std::uint32_t
            {
                Dim = 1, /**< dim, repeated int64. */
            };

            /** @brief Returns the legacy extent field @p field as an index, or K_LEGACY_EXTENTS. */
            std::size_t legacy_index(BlobField field) noexcept
            {
                switch (field)
                {
                    case BlobField::Num:      return 0;
                    case BlobField::Channels: return 1;
                    case BlobField::Height:   return 2;
                    case BlobField::Width:    return 3;
                    default:                  return K_LEGACY_EXTENTS;
                }
            }

            /** @brief Reads one BlobShape into @p dimensions. */
            Status read_blob_shape(IO::ProtobufWire reader, std::string_view origin,
                                   std::vector<std::int64_t>& dimensions)
            {
                const FieldSite site{origin, "BlobShape"};
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    if (static_cast<BlobShapeField>(field.number) == BlobShapeField::Dim)
                        SA_TRY(Detail::read_integers(reader, field, site, dimensions));
                    else
                        SA_TRY(reader.skip(field.type));
                }
                return Status();
            }

            /** @brief Reads one BlobProto into @p out; the layer checks its floats. */
            Status read_blob(IO::ProtobufWire reader, std::string_view origin, CaffeBlob& out)
            {
                const FieldSite site{origin, "BlobProto"};
                std::vector<std::int64_t> shape;
                bool has_shape = false;
                std::vector<std::int64_t> legacy(K_LEGACY_EXTENTS, 0);
                bool has_legacy = false;
                std::vector<BytesField> packed;
                std::vector<float> loose;
                bool has_doubles = false;

                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    const BlobField which = static_cast<BlobField>(field.number);
                    switch (which)
                    {
                        case BlobField::Num:
                        case BlobField::Channels:
                        case BlobField::Height:
                        case BlobField::Width:
                        {
                            SA_TRY_ASSIGN(legacy[legacy_index(which)],
                                          Detail::read_integer(reader, field, site));
                            has_legacy = true;
                            break;
                        }
                        case BlobField::Data:
                        {
                            if (field.type == IO::WireType::LengthDelimited)
                            {
                                SA_TRY_ASSIGN(BytesField part,
                                              Detail::read_bytes(reader, field, site));
                                SA_FAIL_IF(part.bytes.size() % K_FLOAT_BYTES != 0,
                                           StatusCode::InvalidFile,
                                           "'{}': a packed float field of a BlobProto holds {} "
                                           "bytes, which is not a whole number of floats", origin,
                                           part.bytes.size());
                                packed.push_back(part);
                            }
                            else
                            {
                                SA_TRY(Detail::read_reals(reader, field, site, loose));
                            }
                            break;
                        }
                        case BlobField::Shape:
                        {
                            SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                          Detail::read_submessage(reader, field, site));
                            SA_TRY(read_blob_shape(inner, origin, shape));
                            has_shape = true;
                            break;
                        }
                        case BlobField::DoubleData:
                            has_doubles = true;
                            SA_TRY(reader.skip(field.type));
                            break;
                        default:
                            SA_TRY(reader.skip(field.type));
                            break;
                    }
                }

                SA_FAIL_IF(has_doubles, StatusCode::Unsupported,
                           "'{}': a blob stores its elements as doubles, which the importer does "
                           "not read", origin);
                if (has_shape)
                    out.dimensions = std::move(shape);
                else if (has_legacy)
                    out.dimensions = std::move(legacy);

                if (packed.size() == 1 && loose.empty())
                {
                    out.data = TensorBytes::viewing(packed.front().bytes, packed.front().offset);
                    return Status();
                }
                std::vector<std::byte> owned;
                for (const BytesField& part : packed)
                    owned.insert(owned.end(), part.bytes.begin(), part.bytes.end());
                for (float value : loose)
                {
                    std::uint32_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    for (std::size_t i = 0; i < K_FLOAT_BYTES; ++i)
                        owned.push_back(static_cast<std::byte>((bits >> (8u * i)) & 0xFFu));
                }
                out.data = TensorBytes::owning(std::move(owned));
                return Status();
            }

            /** @brief Fails when the floats of @p blob do not fill its extents. */
            Status check_blob(const CaffeBlob& blob, const CaffeLayer& layer, std::size_t index,
                              std::string_view origin) noexcept
            {
                if (blob.dimensions.empty())
                    return Status();
                std::size_t count = 1;
                for (std::int64_t extent : blob.dimensions)
                {
                    SA_FAIL_IF(extent < 0, StatusCode::InvalidFile,
                               "'{}': blob {} of layer '{}' has the negative extent {}", origin,
                               index, layer.name, extent);
                    count *= static_cast<std::size_t>(extent);
                }
                SA_FAIL_IF(count != blob.element_count(), StatusCode::InvalidFile,
                           "'{}': blob {} of layer '{}' holds {} floats, its extents give {}",
                           origin, index, layer.name, blob.element_count(), count);
                return Status();
            }

            /** @brief Reads one LayerParameter (V2) or V1LayerParameter into @p out. */
            Status read_layer(IO::ProtobufWire reader, std::string_view origin, bool legacy,
                              CaffeLayer& out)
            {
                const FieldSite site{origin, legacy ? "V1LayerParameter" : "LayerParameter"};
                out.legacy = legacy;
                while (!reader.at_end())
                {
                    SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                    // The two messages number their fields differently; one map each.
                    bool is_name = false;
                    bool is_type = false;
                    bool is_bottom = false;
                    bool is_top = false;
                    bool is_blob = false;
                    if (legacy)
                    {
                        const V1LayerField which = static_cast<V1LayerField>(field.number);
                        is_name = which == V1LayerField::Name;
                        is_type = which == V1LayerField::Type;
                        is_bottom = which == V1LayerField::Bottom;
                        is_top = which == V1LayerField::Top;
                        is_blob = which == V1LayerField::Blobs;
                    }
                    else
                    {
                        const LayerField which = static_cast<LayerField>(field.number);
                        is_name = which == LayerField::Name;
                        is_type = which == LayerField::Type;
                        is_bottom = which == LayerField::Bottom;
                        is_top = which == LayerField::Top;
                        is_blob = which == LayerField::Blobs;
                    }

                    if (is_name)
                    {
                        SA_TRY_ASSIGN(out.name, Detail::read_text(reader, field, site));
                    }
                    else if (is_type && legacy)
                    {
                        SA_TRY_ASSIGN(const std::int64_t number,
                                      Detail::read_integer(reader, field, site));
                        const Detail::CaffeV1Type* row = Detail::find_v1_type_by_number(number);
                        SA_FAIL_IF(row == nullptr, StatusCode::InvalidFile,
                                   "'{}': the V1 layer '{}' has the type number {}, which the "
                                   "table does not know", origin, out.name, number);
                        out.type = std::string(row->type_name);
                    }
                    else if (is_type)
                    {
                        SA_TRY_ASSIGN(out.type, Detail::read_text(reader, field, site));
                    }
                    else if (is_bottom)
                    {
                        SA_TRY_ASSIGN(std::string bottom, Detail::read_text(reader, field, site));
                        out.bottoms.push_back(std::move(bottom));
                    }
                    else if (is_top)
                    {
                        SA_TRY_ASSIGN(std::string top, Detail::read_text(reader, field, site));
                        out.tops.push_back(std::move(top));
                    }
                    else if (is_blob)
                    {
                        SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                      Detail::read_submessage(reader, field, site));
                        CaffeBlob blob;
                        SA_TRY(read_blob(inner, origin, blob));
                        out.blobs.push_back(std::move(blob));
                    }
                    else
                    {
                        SA_TRY(reader.skip(field.type));
                    }
                }
                for (std::size_t i = 0; i < out.blobs.size(); ++i)
                    SA_TRY(check_blob(out.blobs[i], out, i, origin));
                return Status();
            }

            /** @brief Parses the integer @p field holds; InvalidFile names its line and column. */
            Result<std::int64_t> text_integer(const IO::TextField& field,
                                              std::string_view origin) noexcept
            {
                std::int64_t value = 0;
                const char* last = field.text.data() + field.text.size();
                const std::from_chars_result parsed =
                    std::from_chars(field.text.data(), last, value);
                SA_FAIL_IF(field.kind != IO::TextKind::Number || parsed.ec != std::errc() ||
                               parsed.ptr != last,
                           StatusCode::InvalidFile,
                           "'{}': line {}, column {}: '{}' needs an integer, found '{}'", origin,
                           field.line, field.column, field.name, field.text);
                return value;
            }

            /** @brief Returns the text of @p field when it is a string or a bare word. */
            Result<std::string> text_value(const IO::TextField& field,
                                           std::string_view origin) noexcept
            {
                SA_FAIL_IF(field.kind != IO::TextKind::String &&
                               field.kind != IO::TextKind::Identifier,
                           StatusCode::InvalidFile,
                           "'{}': line {}, column {}: '{}' needs a string", origin, field.line,
                           field.column, field.name);
                return field.text;
            }

            /** @brief Reads one layer message of a prototxt into @p out. */
            Status layer_from_text(const IO::TextField& message, std::string_view origin,
                                   bool legacy, CaffeLayer& out)
            {
                out.legacy = legacy;
                out.parameters = message;
                for (const IO::TextField& field : message.fields)
                {
                    if (field.name == "name")
                    {
                        SA_TRY_ASSIGN(out.name, text_value(field, origin));
                    }
                    else if (field.name == "type")
                    {
                        if (legacy && field.kind == IO::TextKind::Identifier)
                        {
                            const Detail::CaffeV1Type* row =
                                Detail::find_v1_type_by_identifier(field.text);
                            SA_FAIL_IF(row == nullptr, StatusCode::InvalidFile,
                                       "'{}': line {}, column {}: the V1 layer type '{}' is not "
                                       "in the table", origin, field.line, field.column,
                                       field.text);
                            out.type = std::string(row->type_name);
                        }
                        else
                        {
                            SA_TRY_ASSIGN(out.type, text_value(field, origin));
                        }
                    }
                    else if (field.name == "bottom")
                    {
                        SA_TRY_ASSIGN(std::string bottom, text_value(field, origin));
                        out.bottoms.push_back(std::move(bottom));
                    }
                    else if (field.name == "top")
                    {
                        SA_TRY_ASSIGN(std::string top, text_value(field, origin));
                        out.tops.push_back(std::move(top));
                    }
                }
                SA_FAIL_IF(out.name.empty() || out.type.empty(), StatusCode::InvalidFile,
                           "'{}': line {}, column {}: a layer needs a name and a type", origin,
                           message.line, message.column);
                return Status();
            }

            /** @brief Reads the shape message of an input_shape field into @p out. */
            Status shape_from_text(const IO::TextField& message, std::string_view origin,
                                   std::vector<std::int64_t>& out)
            {
                for (const IO::TextField* dim : IO::find_fields(message, "dim"))
                {
                    SA_TRY_ASSIGN(const std::int64_t extent, text_integer(*dim, origin));
                    out.push_back(extent);
                }
                return Status();
            }
        } // namespace

        Result<CaffeNet> read_caffe_prototxt(std::string_view text,
                                             std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(const IO::TextField root, IO::parse_protobuf_text(text, origin));
            CaffeNet net;
            std::vector<std::int64_t> flat_dimensions;

            for (const IO::TextField& field : root.fields)
            {
                if (field.name == "name")
                {
                    SA_TRY_ASSIGN(net.name, text_value(field, origin));
                }
                else if (field.name == "input")
                {
                    SA_TRY_ASSIGN(std::string input, text_value(field, origin));
                    net.inputs.push_back(std::move(input));
                }
                else if (field.name == "input_dim")
                {
                    SA_TRY_ASSIGN(const std::int64_t extent, text_integer(field, origin));
                    flat_dimensions.push_back(extent);
                }
                else if (field.name == "input_shape")
                {
                    std::vector<std::int64_t> shape;
                    SA_TRY(shape_from_text(field, origin, shape));
                    net.input_shapes.push_back(std::move(shape));
                }
                else if (field.name == "layer" || field.name == "layers")
                {
                    CaffeLayer layer;
                    SA_TRY(layer_from_text(field, origin, field.name == "layers", layer));
                    net.layers.push_back(std::move(layer));
                }
            }

            if (!flat_dimensions.empty())
            {
                SA_FAIL_IF(!net.input_shapes.empty() ||
                               flat_dimensions.size() != K_LEGACY_EXTENTS * net.inputs.size(),
                           StatusCode::InvalidFile,
                           "'{}': input_dim holds {} numbers for {} inputs, expected four each "
                           "and no input_shape beside them", origin, flat_dimensions.size(),
                           net.inputs.size());
                for (std::size_t i = 0; i < net.inputs.size(); ++i)
                    net.input_shapes.emplace_back(
                        flat_dimensions.begin() + static_cast<std::ptrdiff_t>(i * K_LEGACY_EXTENTS),
                        flat_dimensions.begin() +
                            static_cast<std::ptrdiff_t>((i + 1) * K_LEGACY_EXTENTS));
            }
            return net;
        }

        Result<CaffeNet> read_caffe_model(SushiRuntime::span<const std::byte> bytes,
                                          std::string_view origin) noexcept
        {
            const FieldSite site{origin, "NetParameter"};
            IO::ProtobufWire reader(bytes, origin);
            CaffeNet net;
            while (!reader.at_end())
            {
                SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
                switch (static_cast<NetField>(field.number))
                {
                    case NetField::Name:
                    {
                        SA_TRY_ASSIGN(net.name, Detail::read_text(reader, field, site));
                        break;
                    }
                    case NetField::V1Layers:
                    case NetField::Layers:
                    {
                        SA_TRY_ASSIGN(IO::ProtobufWire inner,
                                      Detail::read_submessage(reader, field, site));
                        CaffeLayer layer;
                        SA_TRY(read_layer(inner, origin,
                                          static_cast<NetField>(field.number) == NetField::V1Layers,
                                          layer));
                        net.layers.push_back(std::move(layer));
                        break;
                    }
                    default:
                        SA_TRY(reader.skip(field.type));
                        break;
                }
            }
            return net;
        }

        Status attach_caffe_weights(CaffeNet& net, const CaffeNet& weights)
        {
            std::vector<std::string> missing;
            std::vector<std::pair<std::size_t, const CaffeLayer*>> placements;
            for (const CaffeLayer& source : weights.layers)
            {
                if (source.blobs.empty())
                    continue;
                const std::size_t matches = net.count_layers(source.name);
                if (matches == 0)
                {
                    missing.push_back(source.name);
                    continue;
                }
                SA_FAIL_IF(matches > 1, StatusCode::Incompatible,
                           "the net names {} layers '{}', so the weights cannot say which one "
                           "they belong to", matches, source.name);
                for (std::size_t i = 0; i < net.layers.size(); ++i)
                {
                    const CaffeLayer& target = net.layers[i];
                    if (target.name != source.name)
                        continue;
                    SA_FAIL_IF(!target.blobs.empty(), StatusCode::Incompatible,
                               "the layer '{}' of the net already holds blobs", target.name);
                    SA_FAIL_IF(target.type != source.type, StatusCode::Incompatible,
                               "the layer '{}' is a {} in the net and a {} in the weights",
                               target.name, target.type, source.type);
                    placements.emplace_back(i, &source);
                }
            }

            if (!missing.empty())
            {
                std::string names;
                for (const std::string& name : missing)
                    names += (names.empty() ? "'" : ", '") + name + "'";
                SA_FAIL_IF(true, StatusCode::Incompatible,
                           "the weights hold blobs for the layers {}, which the net does not "
                           "define", names);
            }
            for (const auto& placement : placements)
                net.layers[placement.first].blobs = placement.second->blobs;
            return Status();
        }
    } // namespace Interchange
} // namespace SushiAI
