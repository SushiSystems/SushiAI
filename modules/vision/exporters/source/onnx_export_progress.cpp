/**************************************************************************/
/* onnx_export_progress.cpp                                               */
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
 * @file onnx_export_progress.cpp
 * @brief Defines the helpers the ONNX row writers share.
 * @author Mustafa Garip
 */

#include "onnx_export_progress.hpp"

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
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>
#include <SushiAI/io/imported_model.hpp>
#include "export_walk.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the rank of one image sample, [H, W, C]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Returns the bytes of @p count elements of @p size bytes at @p data. */
                std::vector<std::byte> copy_bytes(const void* data, std::size_t count,
                                                  std::size_t size)
                {
                    std::vector<std::byte> bytes(count * size);
                    if (!bytes.empty())
                        std::memcpy(bytes.data(), data, bytes.size());
                    return bytes;
                }
            } // namespace

            RowKeys row_keys(const OnnxRowCall& call)
            {
                return RowKeys{call.spec, call.index, K_ONNX_FORMAT};
            }

            Status check_source_count(const OnnxRowCall& call, std::size_t count)
            {
                return check_read_count(call.index, *call.spec, call.sources.size(), count);
            }

            Status check_image(const OnnxRowValue& value)
            {
                SA_FAIL_IF(value.shape.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "it reads '{}' of rank {}, and its ONNX operator reads an image "
                           "[H, W, C]",
                           value.name, value.shape.size());
                return Status();
            }

            void add_float_initializer(OnnxExportProgress& progress, std::string name,
                                       std::vector<std::int64_t> extents,
                                       const std::vector<float>& values)
            {
                Interchange::OnnxTensor tensor;
                tensor.name = std::move(name);
                tensor.dimensions = std::move(extents);
                tensor.data_type = Interchange::OnnxDataType::Float;
                tensor.data = Interchange::TensorBytes::owning(
                    copy_bytes(values.data(), values.size(), sizeof(float)));
                progress.graph.initializers.push_back(std::move(tensor));
            }

            void add_int_initializer(OnnxExportProgress& progress, std::string name,
                                     std::vector<std::int64_t> extents,
                                     const std::vector<std::int64_t>& values)
            {
                Interchange::OnnxTensor tensor;
                tensor.name = std::move(name);
                tensor.dimensions = std::move(extents);
                tensor.data_type = Interchange::OnnxDataType::Int64;
                tensor.data = Interchange::TensorBytes::owning(
                    copy_bytes(values.data(), values.size(), sizeof(std::int64_t)));
                progress.graph.initializers.push_back(std::move(tensor));
            }

            Interchange::OnnxAttribute make_int_attribute(std::string name, std::int64_t value)
            {
                Interchange::OnnxAttribute attribute;
                attribute.name = std::move(name);
                attribute.type = Interchange::OnnxAttributeType::Int;
                attribute.integer = value;
                return attribute;
            }

            Interchange::OnnxAttribute make_ints_attribute(std::string name,
                                                           std::vector<std::int64_t> values)
            {
                Interchange::OnnxAttribute attribute;
                attribute.name = std::move(name);
                attribute.type = Interchange::OnnxAttributeType::Ints;
                attribute.integers = std::move(values);
                return attribute;
            }

            Interchange::OnnxAttribute make_real_attribute(std::string name, float value)
            {
                Interchange::OnnxAttribute attribute;
                attribute.name = std::move(name);
                attribute.type = Interchange::OnnxAttributeType::Float;
                attribute.real = value;
                return attribute;
            }

            Interchange::OnnxAttribute make_text_attribute(std::string name, std::string value)
            {
                Interchange::OnnxAttribute attribute;
                attribute.name = std::move(name);
                attribute.type = Interchange::OnnxAttributeType::String;
                attribute.text = std::move(value);
                return attribute;
            }

            void add_node(OnnxExportProgress& progress, const OnnxRowCall& call,
                          std::string_view op_type, std::string_view suffix,
                          std::vector<std::string> inputs, std::vector<std::string> outputs,
                          std::vector<Interchange::OnnxAttribute> attributes)
            {
                Interchange::OnnxNode node;
                node.name = row_tensor_name(call.index, suffix);
                node.op_type = std::string(op_type);
                node.inputs = std::move(inputs);
                node.outputs = std::move(outputs);
                node.attributes = std::move(attributes);
                progress.graph.nodes.push_back(std::move(node));
            }

            void define_row_value(OnnxExportProgress& progress, const OnnxRowCall& call,
                                  std::vector<std::int64_t> shape)
            {
                progress.values[call.index] =
                    OnnxRowValue{progress.value_names[call.index], std::move(shape), {}};
            }

            const std::string& row_output(const OnnxExportProgress& progress,
                                          const OnnxRowCall& call)
            {
                return progress.value_names[call.index];
            }

            Status write_pointwise_row(const OnnxRowCall& call, OnnxExportProgress& progress,
                                       std::string_view op_type)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                add_node(progress, call, op_type, "node", {input.name},
                         {row_output(progress, call)});
                define_row_value(progress, call, input.shape);
                return Status();
            }

            Status write_binary_row(const OnnxRowCall& call, OnnxExportProgress& progress,
                                    std::string_view op_type)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_TRY(check_source_count(call, 2));
                const OnnxRowValue& first = *call.sources[0];
                const OnnxRowValue& second = *call.sources[1];
                SA_FAIL_IF(first.shape != second.shape, StatusCode::Unsupported,
                           "its operands '{}' and '{}' differ in shape, and the ONNX importer "
                           "reads two operands of equal extents only",
                           first.name, second.name);
                add_node(progress, call, op_type, "node", {first.name, second.name},
                         {row_output(progress, call)});
                define_row_value(progress, call, first.shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
