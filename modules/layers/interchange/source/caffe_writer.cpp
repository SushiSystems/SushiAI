/**************************************************************************/
/* caffe_writer.cpp                                                       */
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
 * @file caffe_writer.cpp
 * @brief Implements the writers of a Caffe prototxt and caffemodel.
 * @author Mustafa Garip
 */

#include <SushiAI/interchange/caffe_writer.hpp>

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
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include <SushiAI/io/protobuf_text_writer.hpp>
#include <SushiAI/io/protobuf_wire_writer.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        namespace
        {
            using IO::ProtobufWireWriter;
            using IO::TextField;
            using IO::TextKind;

            /** @brief Holds the field numbers of NetParameter, LayerParameter and BlobProto. */
            enum CaffeField : std::uint32_t
            {
                K_NET_NAME = 1,
                K_NET_LAYER = 100,
                K_LAYER_NAME = 1,
                K_LAYER_TYPE = 2,
                K_LAYER_BOTTOM = 3,
                K_LAYER_TOP = 4,
                K_LAYER_BLOBS = 7,
                K_BLOB_DATA = 5,
                K_BLOB_SHAPE = 7,
                K_SHAPE_DIM = 1,
            };

            /** @brief Lists the layer fields the writer writes from the layer's members. */
            constexpr std::string_view K_MEMBER_FIELDS[] = {"name", "type", "bottom", "top"};

            /** @brief Returns a field named @p name of @p kind holding @p text. */
            [[nodiscard]] TextField leaf(std::string_view name, TextKind kind, std::string text)
            {
                TextField field;
                field.name = std::string(name);
                field.kind = kind;
                field.text = std::move(text);
                return field;
            }

            /** @brief Returns whether the writer writes @p name from the layer's members. */
            [[nodiscard]] bool is_member_field(std::string_view name) noexcept
            {
                for (std::string_view member : K_MEMBER_FIELDS)
                {
                    if (name == member)
                        return true;
                }
                return false;
            }

            /** @brief Returns the `layer` message of @p layer. */
            [[nodiscard]] TextField layer_message(const CaffeLayer& layer)
            {
                TextField message;
                message.name = "layer";
                message.fields.push_back(leaf("name", TextKind::String, layer.name));
                message.fields.push_back(leaf("type", TextKind::String, layer.type));
                for (const std::string& bottom : layer.bottoms)
                    message.fields.push_back(leaf("bottom", TextKind::String, bottom));
                for (const std::string& top : layer.tops)
                    message.fields.push_back(leaf("top", TextKind::String, top));
                for (const TextField& field : layer.parameters.fields)
                {
                    if (!is_member_field(field.name))
                        message.fields.push_back(field);
                }
                return message;
            }

            /** @brief Writes @p blob as a BlobProto into @p out. */
            Status write_blob(const CaffeLayer& layer, std::size_t index, ProtobufWireWriter& out)
            {
                const CaffeBlob& blob = layer.blobs[index];
                std::size_t elements = 1;
                for (std::int64_t extent : blob.dimensions)
                    elements *= static_cast<std::size_t>(extent);
                SA_FAIL_IF(elements * sizeof(float) != blob.data.size(),
                           StatusCode::InvalidDescription,
                           "blob {} of the layer '{}' holds {} floats, but its extents need {}",
                           index, layer.name, blob.data.size() / sizeof(float), elements);

                ProtobufWireWriter shape;
                shape.write_packed_varints(K_SHAPE_DIM,
                                           SushiRuntime::span<const std::int64_t>(
                                               blob.dimensions.data(), blob.dimensions.size()));
                std::vector<float> values(blob.data.size() / sizeof(float));
                std::memcpy(values.data(), blob.data.bytes().data(), blob.data.size());
                out.write_packed_floats(K_BLOB_DATA,
                                        SushiRuntime::span<const float>(values.data(),
                                                                        values.size()));
                out.write_message(K_BLOB_SHAPE, shape);
                return Status();
            }
        } // namespace

        Result<std::string> write_caffe_prototxt(const CaffeNet& net) noexcept
        {
            SA_FAIL_IF(net.input_shapes.size() != net.inputs.size(),
                       StatusCode::InvalidDescription,
                       "the net '{}' states {} inputs and {} input shapes", net.name,
                       net.inputs.size(), net.input_shapes.size());
            for (const CaffeLayer& layer : net.layers)
            {
                SA_FAIL_IF(layer.name.empty() || layer.type.empty(),
                           StatusCode::InvalidDescription,
                           "the layer '{}' of type '{}' needs a name and a type", layer.name,
                           layer.type);
            }

            TextField root;
            if (!net.name.empty())
                root.fields.push_back(leaf("name", TextKind::String, net.name));
            for (std::size_t i = 0; i < net.inputs.size(); ++i)
            {
                root.fields.push_back(leaf("input", TextKind::String, net.inputs[i]));
                TextField shape;
                shape.name = "input_shape";
                for (std::int64_t extent : net.input_shapes[i])
                    shape.fields.push_back(leaf("dim", TextKind::Number, std::to_string(extent)));
                root.fields.push_back(std::move(shape));
            }
            for (const CaffeLayer& layer : net.layers)
                root.fields.push_back(layer_message(layer));
            return IO::write_protobuf_text(root);
        }

        Result<std::vector<std::byte>> write_caffe_model(const CaffeNet& net) noexcept
        {
            ProtobufWireWriter model;
            if (!net.name.empty())
                model.write_string(K_NET_NAME, net.name);
            for (const CaffeLayer& layer : net.layers)
            {
                if (layer.blobs.empty())
                    continue;
                ProtobufWireWriter message;
                message.write_string(K_LAYER_NAME, layer.name);
                message.write_string(K_LAYER_TYPE, layer.type);
                for (const std::string& bottom : layer.bottoms)
                    message.write_string(K_LAYER_BOTTOM, bottom);
                for (const std::string& top : layer.tops)
                    message.write_string(K_LAYER_TOP, top);
                for (std::size_t i = 0; i < layer.blobs.size(); ++i)
                {
                    ProtobufWireWriter blob;
                    SA_TRY(write_blob(layer, i, blob));
                    message.write_message(K_LAYER_BLOBS, blob);
                }
                model.write_message(K_NET_LAYER, message);
            }
            return model.take();
        }
    } // namespace Interchange
} // namespace SushiAI
