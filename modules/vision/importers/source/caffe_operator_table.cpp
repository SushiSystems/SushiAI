/**************************************************************************/
/* caffe_operator_table.cpp                                               */
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
 * @file caffe_operator_table.cpp
 * @brief Defines the table of Caffe layer translators and the helpers every translator shares.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail::Caffe
        {
            namespace
            {
                /** @brief Lists every layer type this importer translates, one translator each. */
                constexpr OperatorTableRow K_OPERATOR_TABLE[] = {
                    {"Input", &translate_input},
                    {"Convolution", &translate_convolution},
                    {"Pooling", &translate_pooling},
                    {"ReLU", &translate_relu},
                    {"InnerProduct", &translate_inner_product},
                    {"Concat", &translate_concat},
                    {"Eltwise", &translate_eltwise},
                    {"Softmax", &translate_softmax},
                    {"Dropout", &translate_dropout},
                    {"BatchNorm", &translate_batch_norm},
                    {"Scale", &translate_scale},
                };
            } // namespace

            const OperatorTableRow* find_operator_translator(std::string_view type) noexcept
            {
                for (const OperatorTableRow& row : K_OPERATOR_TABLE)
                {
                    if (row.type == type)
                        return &row;
                }
                return nullptr;
            }

            std::string describe_layer(const Interchange::CaffeLayer& layer)
            {
                return "layer '" + layer.name + "' (" + layer.type + ")";
            }

            Status expect_shape(const Interchange::CaffeLayer& layer, std::size_t bottoms)
            {
                SA_FAIL_IF(layer.bottoms.size() != bottoms || layer.tops.size() != 1,
                           StatusCode::InvalidDescription,
                           "{} has {} bottoms and {} tops; this type takes {} and 1",
                           describe_layer(layer), layer.bottoms.size(), layer.tops.size(), bottoms);
                return Status();
            }

            Result<BlobValue> read_bottom(const TranslationProgress& progress,
                                          const Interchange::CaffeLayer& layer,
                                          std::size_t position)
            {
                SA_FAIL_IF(position >= layer.bottoms.size(), StatusCode::InvalidDescription,
                           "{} has {} bottoms and reads number {}", describe_layer(layer),
                           layer.bottoms.size(), position);
                const NamedBlob* found = nullptr;
                for (const NamedBlob& blob : progress.blobs)
                {
                    if (blob.name == layer.bottoms[position])
                        found = &blob;
                }
                SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                           "{} reads the blob '{}', which no layer above it writes",
                           describe_layer(layer), layer.bottoms[position]);
                return found->value;
            }

            Status expect_image(const Interchange::CaffeLayer& layer, const BlobValue& value)
            {
                SA_FAIL_IF(!value.is_image, StatusCode::Unsupported,
                           "{} reads a blob that is no longer an image; the rows of this type "
                           "take [H, W, C]",
                           describe_layer(layer));
                return Status();
            }

            void write_blob(TranslationProgress& progress, const std::string& name,
                            const BlobValue& value, std::int64_t writer)
            {
                for (NamedBlob& blob : progress.blobs)
                {
                    if (blob.name == name)
                    {
                        blob.value = value;
                        blob.writer = writer;
                        return;
                    }
                }
                progress.blobs.push_back(NamedBlob{name, value, writer});
            }

            std::int64_t find_reader(const TranslationProgress& progress, const std::string& name,
                                     std::int64_t after) noexcept
            {
                const std::vector<Interchange::CaffeLayer>& layers = progress.net->layers;
                for (std::size_t at = static_cast<std::size_t>(after + 1); at < layers.size(); ++at)
                {
                    for (const std::string& bottom : layers[at].bottoms)
                    {
                        if (bottom == name)
                            return static_cast<std::int64_t>(at);
                    }
                }
                return -1;
            }

            Config::Value make_attributes(std::vector<std::string> keys,
                                          std::vector<Config::Value> values)
            {
                SA_FATAL_IF(keys.size() != values.size(),
                            "a row's attributes have {} keys and {} values", keys.size(),
                            values.size());
                return Config::Value::object(std::move(keys), std::move(values));
            }

            Config::Value pair_value(std::int64_t height, std::int64_t width)
            {
                if (height == width)
                    return Config::Value::integer(height);
                return Config::Value::array(
                    {Config::Value::integer(height), Config::Value::integer(width)});
            }

            std::int64_t append_row(TranslationProgress& progress, std::string_view type,
                                    Config::Value attributes,
                                    SushiRuntime::span<const std::int64_t> sources)
            {
                Config::LayerSpec row;
                row.type = std::string(type);
                row.attributes = std::move(attributes);
                row.from.reserve(sources.size());
                for (const std::int64_t source : sources)
                    row.from.push_back(Config::SourceSpec{source, std::string()});

                std::vector<Config::LayerSpec>& rows = progress.spec.layers;
                rows.push_back(std::move(row));
                return static_cast<std::int64_t>(rows.size()) - 1;
            }

            std::string entry_name(std::int64_t row, std::string_view field)
            {
                return std::to_string(row).append(".").append(field);
            }

            Status expect_blobs(const TranslationProgress& progress,
                                const Interchange::CaffeLayer& layer, std::size_t needed)
            {
                SA_FAIL_IF(progress.has_weights && layer.blobs.size() < needed,
                           StatusCode::Incompatible,
                           "{} holds {} blobs in the caffemodel, and its weights need {}",
                           describe_layer(layer), layer.blobs.size(), needed);
                return Status();
            }

            Result<std::vector<float>> read_blob(const Interchange::CaffeLayer& layer,
                                                 std::size_t position, std::size_t expected)
            {
                const Interchange::CaffeBlob& blob = layer.blobs[position];
                SA_FAIL_IF(blob.element_count() != expected, StatusCode::Incompatible,
                           "{} holds {} floats in blob {}, and its description gives {}",
                           describe_layer(layer), blob.element_count(), position, expected);
                std::vector<float> values(expected);
                if (expected != 0)
                    std::memcpy(values.data(), blob.data.bytes().data(), expected * sizeof(float));
                return values;
            }

            void append_tensor(TranslationProgress& progress, std::string name,
                               const Shape& shape, const std::vector<float>& values)
            {
                IO::ImportedTensor tensor;
                tensor.record.name = std::move(name);
                tensor.record.dtype = DType::Float32;
                tensor.record.shape = shape;
                tensor.payload.resize(values.size() * sizeof(float));
                if (!values.empty())
                    std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
                progress.tensors.push_back(std::move(tensor));
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
