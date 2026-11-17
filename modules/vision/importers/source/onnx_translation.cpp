/**************************************************************************/
/* onnx_translation.cpp                                                   */
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
 * @file onnx_translation.cpp
 * @brief Defines the helpers every ONNX operator translator shares.
 * @author Mustafa Garip
 */

#include "onnx_translation.hpp"

#include <algorithm>
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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the value of auto_pad that states no padding rule. */
                constexpr std::string_view K_AUTO_PAD_NOT_SET = "NOTSET";

                /** @brief Holds the number of image axes a window covers. */
                constexpr std::size_t K_IMAGE_AXES = 2;

                /** @brief Reads the @p index-th element of @p bytes as a value of type T. */
                template<typename T>
                T load_element(SushiRuntime::span<const std::byte> bytes,
                          std::size_t index) noexcept
                {
                    T value;
                    std::memcpy(&value, bytes.data() + index * sizeof(T), sizeof(T));
                    return value;
                }

                /** @brief Returns the vector of a pair attribute, or @p fallback when absent. */
                Result<std::vector<std::int64_t>> read_pair(const Interchange::OnnxNode& node,
                                                            std::string_view name,
                                                            std::vector<std::int64_t> fallback,
                                                            std::size_t length)
                {
                    SA_TRY_ASSIGN(std::vector<std::int64_t> stated,
                                  read_ints_attribute(node, name));
                    if (stated.empty())
                        return fallback;
                    SA_FAIL_IF(stated.size() != length, StatusCode::Unsupported,
                               "{}: '{}' lists {} values, and only {} are translated",
                               describe_node(node), name, stated.size(), length);
                    return stated;
                }
            } // namespace

            std::string describe_node(const Interchange::OnnxNode& node)
            {
                if (node.name.empty())
                    return "node (" + node.op_type + ")";
                return "node '" + node.name + "' (" + node.op_type + ")";
            }

            bool has_operand(const Interchange::OnnxNode& node, std::size_t position) noexcept
            {
                return position < node.inputs.size() && !node.inputs[position].empty();
            }

            Status check_operand_count(const Interchange::OnnxNode& node, std::size_t fewest,
                                       std::size_t most)
            {
                SA_FAIL_IF(node.inputs.size() < fewest || node.inputs.size() > most,
                           StatusCode::InvalidDescription,
                           "{} reads {} operands, and the operator takes {} to {}",
                           describe_node(node), node.inputs.size(), fewest, most);
                for (std::size_t at = 0; at < fewest; ++at)
                {
                    SA_FAIL_IF(!has_operand(node, at), StatusCode::InvalidDescription,
                               "{} leaves out the required operand {}", describe_node(node), at);
                }
                return Status();
            }

            Status check_output_count(const Interchange::OnnxNode& node, std::size_t count)
            {
                SA_FAIL_IF(node.outputs.size() != count, StatusCode::Unsupported,
                           "{} writes {} outputs, and the translation reads {}",
                           describe_node(node),
                           node.outputs.size(), count);
                return Status();
            }

            Status reject_unknown_attributes(const Interchange::OnnxNode& node,
                                             SushiRuntime::span<const std::string_view> known)
            {
                for (const Interchange::OnnxAttribute& attribute : node.attributes)
                {
                    const bool listed =
                        std::find(known.begin(), known.end(), std::string_view(attribute.name)) !=
                        known.end();
                    SA_FAIL_IF(!listed, StatusCode::Unsupported,
                               "{} states the attribute '{}', which the translation does not read",
                               describe_node(node), attribute.name);
                }
                return Status();
            }

            Result<std::int64_t> read_int_attribute(const Interchange::OnnxNode& node,
                                                    std::string_view name, std::int64_t fallback)
            {
                const Interchange::OnnxAttribute* attribute = node.find_attribute(name);
                if (attribute == nullptr)
                    return fallback;
                SA_FAIL_IF(attribute->type != Interchange::OnnxAttributeType::Int,
                           StatusCode::InvalidDescription, "{}: '{}' is not an integer",
                           describe_node(node), name);
                return attribute->integer;
            }

            Result<std::vector<std::int64_t>> read_ints_attribute(const Interchange::OnnxNode& node,
                                                                  std::string_view name)
            {
                const Interchange::OnnxAttribute* attribute = node.find_attribute(name);
                if (attribute == nullptr)
                    return std::vector<std::int64_t>();
                SA_FAIL_IF(attribute->type != Interchange::OnnxAttributeType::Ints,
                           StatusCode::InvalidDescription, "{}: '{}' is not a list of integers",
                           describe_node(node), name);
                return attribute->integers;
            }

            Result<float> read_real_attribute(const Interchange::OnnxNode& node,
                                              std::string_view name, float fallback)
            {
                const Interchange::OnnxAttribute* attribute = node.find_attribute(name);
                if (attribute == nullptr)
                    return fallback;
                SA_FAIL_IF(attribute->type != Interchange::OnnxAttributeType::Float,
                           StatusCode::InvalidDescription, "{}: '{}' is not a float",
                           describe_node(node), name);
                return attribute->real;
            }

            Result<std::string> read_text_attribute(const Interchange::OnnxNode& node,
                                                    std::string_view name,
                                                    std::string_view fallback)
            {
                const Interchange::OnnxAttribute* attribute = node.find_attribute(name);
                if (attribute == nullptr)
                    return std::string(fallback);
                SA_FAIL_IF(attribute->type != Interchange::OnnxAttributeType::String,
                           StatusCode::InvalidDescription, "{}: '{}' is not a string",
                           describe_node(node), name);
                return attribute->text;
            }

            Result<const OnnxValue*> find_value(OnnxProgress& progress,
                                                const Interchange::OnnxNode& node,
                                                const std::string& name)
            {
                const auto found = progress.values.find(name);
                if (found != progress.values.end())
                    return &found->second;
                const bool constant =
                    progress.constants.count(name) != 0 ||
                    progress.graph->find_initializer(name) != nullptr;
                SA_FAIL_IF(constant, StatusCode::Unsupported,
                           "{} reads the constant '{}' where a computed value is needed",
                           describe_node(node), name);
                return Status::from_message(StatusCode::InvalidDescription, [&]()
                {
                    return describe_node(node) + " reads '" + name +
                           "', which no earlier node or graph input writes";
                });
            }

            Result<const OnnxValue*> find_operand_value(OnnxProgress& progress,
                                                        const Interchange::OnnxNode& node,
                                                        std::size_t position)
            {
                SA_FAIL_IF(!has_operand(node, position), StatusCode::InvalidDescription,
                           "{} leaves out operand {}", describe_node(node), position);
                return find_value(progress, node, node.inputs[position]);
            }

            Result<OnnxConstant> read_constant_tensor(const Interchange::OnnxTensor& tensor)
            {
                using Interchange::OnnxDataType;
                const std::size_t count = tensor.element_count();
                const SushiRuntime::span<const std::byte> bytes = tensor.data.bytes();
                const std::size_t width = Interchange::onnx_element_bytes(tensor.data_type);
                const bool supported = tensor.data_type == OnnxDataType::Float ||
                                       tensor.data_type == OnnxDataType::Double ||
                                       tensor.data_type == OnnxDataType::Int32 ||
                                       tensor.data_type == OnnxDataType::Int64;
                SA_FAIL_IF(!supported, StatusCode::Unsupported,
                           "the tensor '{}' has the ONNX element type {}, and only float32, "
                           "float64, int32 and int64 are read",
                           tensor.name, static_cast<std::int32_t>(tensor.data_type));
                SA_FAIL_IF(bytes.size() != count * width, StatusCode::InvalidDescription,
                           "the tensor '{}' holds {} bytes for {} elements", tensor.name,
                           bytes.size(), count);

                OnnxConstant constant;
                constant.dimensions = tensor.dimensions;
                for (std::size_t at = 0; at < count; ++at)
                {
                    switch (tensor.data_type)
                    {
                        case OnnxDataType::Float:
                            constant.reals.push_back(load_element<float>(bytes, at));
                            break;
                        case OnnxDataType::Double:
                            constant.reals.push_back(
                                static_cast<float>(load_element<double>(bytes, at)));
                            break;
                        case OnnxDataType::Int32:
                            constant.integers.push_back(load_element<std::int32_t>(bytes, at));
                            break;
                        default:
                            constant.integers.push_back(load_element<std::int64_t>(bytes, at));
                            break;
                    }
                }
                constant.is_real = tensor.data_type == OnnxDataType::Float ||
                                   tensor.data_type == OnnxDataType::Double;
                return constant;
            }

            void define_constant(OnnxProgress& progress, const std::string& name,
                                 OnnxConstant constant)
            {
                progress.constants[name] = std::move(constant);
            }

            Result<const OnnxConstant*> find_operand_constant(OnnxProgress& progress,
                                                              const Interchange::OnnxNode& node,
                                                              std::size_t position)
            {
                SA_FAIL_IF(!has_operand(node, position), StatusCode::InvalidDescription,
                           "{} leaves out operand {}", describe_node(node), position);
                const std::string& name = node.inputs[position];
                const auto cached = progress.constants.find(name);
                if (cached != progress.constants.end())
                    return &cached->second;
                SA_FAIL_IF(progress.values.count(name) != 0, StatusCode::Unsupported,
                           "{} reads '{}' as a constant, but it is computed at run time",
                           describe_node(node), name);

                for (const Interchange::NamedConstant& folded : progress.folded->constants)
                {
                    if (folded.name != name)
                        continue;
                    OnnxConstant constant;
                    constant.dimensions = folded.value.dimensions;
                    constant.is_real = folded.value.kind == Interchange::ConstantKind::Real;
                    constant.reals = folded.value.reals;
                    constant.integers = folded.value.integers;
                    define_constant(progress, name, std::move(constant));
                    return &progress.constants.at(name);
                }

                const Interchange::OnnxTensor* initializer = progress.graph->find_initializer(name);
                SA_FAIL_IF(initializer == nullptr, StatusCode::InvalidDescription,
                           "{} reads '{}', which no initializer or constant node holds",
                           describe_node(node), name);
                SA_TRY_ASSIGN(OnnxConstant constant, read_constant_tensor(*initializer));
                define_constant(progress, name, std::move(constant));
                return &progress.constants.at(name);
            }

            Result<const OnnxConstant*> find_real_operand(OnnxProgress& progress,
                                                          const Interchange::OnnxNode& node,
                                                          std::size_t position, std::size_t rank)
            {
                SA_TRY_ASSIGN(const OnnxConstant* constant,
                              find_operand_constant(progress, node, position));
                SA_FAIL_IF(!constant->is_real, StatusCode::Unsupported,
                           "{}: operand {} holds integers, and a floating tensor is needed",
                           describe_node(node), position);
                SA_FAIL_IF(constant->dimensions.size() != rank, StatusCode::InvalidDescription,
                           "{}: operand {} has rank {}, and rank {} is needed", describe_node(node),
                           position, constant->dimensions.size(), rank);
                return constant;
            }

            void define_value(OnnxProgress& progress, const std::string& name, std::int64_t row,
                              std::vector<std::int64_t> dimensions,
                              std::vector<std::int64_t> flattened_from)
            {
                OnnxValue value;
                value.row = row;
                value.dimensions = std::move(dimensions);
                value.flattened_from = std::move(flattened_from);
                progress.values[name] = std::move(value);
            }

            std::int64_t append_row(OnnxProgress& progress, std::string_view type,
                                    std::vector<std::string> keys,
                                    std::vector<Config::Value> values,
                                    const std::vector<std::int64_t>& sources)
            {
                const std::int64_t index = static_cast<std::int64_t>(progress.layers.size());
                const std::int64_t above = index == 0 ? Config::K_MODEL_INPUT : index - 1;
                keys.insert(keys.begin(), "type");
                values.insert(values.begin(), Config::Value::string(std::string(type)));

                const bool reads_above = sources.size() == 1 && sources.front() == above;
                if (!reads_above)
                {
                    std::vector<Config::Value> entries;
                    for (const std::int64_t source : sources)
                    {
                        entries.push_back(source == Config::K_MODEL_INPUT
                                              ? Config::Value::string("input")
                                              : Config::Value::integer(source));
                    }
                    keys.insert(keys.begin() + 1, "from");
                    values.insert(values.begin() + 1,
                                  entries.size() == 1 ? std::move(entries.front())
                                                      : Config::Value::array(std::move(entries)));
                }
                progress.layers.push_back(
                    Config::Value::object(std::move(keys), std::move(values)));
                return index;
            }

            std::int64_t append_pointwise_row(OnnxProgress& progress, std::string_view type,
                                              const OnnxValue& value)
            {
                return append_row(progress, type, {}, {}, {value.row});
            }

            Status translate_binary_row(const Interchange::OnnxNode& node, OnnxProgress& progress,
                                        std::string_view row_type)
            {
                SA_TRY(check_operand_count(node, 2, 2));
                SA_TRY(check_output_count(node, 1));
                SA_TRY_ASSIGN(const OnnxValue* first, find_operand_value(progress, node, 0));
                SA_TRY_ASSIGN(const OnnxValue* second, find_operand_value(progress, node, 1));
                SA_FAIL_IF(first->dimensions != second->dimensions, StatusCode::Unsupported,
                           "{} combines two values of different extents, and the {} row does not "
                           "broadcast across the channel-first extents",
                           describe_node(node), row_type);

                const std::int64_t row =
                    append_row(progress, row_type, {}, {}, {first->row, second->row});
                define_value(progress, node.outputs[0], row, first->dimensions);
                return Status();
            }

            void append_flatten(OnnxProgress& progress, const std::string& output,
                                const OnnxValue& input)
            {
                std::int64_t features = 1;
                for (const std::int64_t extent : input.dimensions)
                    features *= extent;
                const std::int64_t row = append_pointwise_row(progress, "flatten", input);
                define_value(progress, output, row, {features}, input.dimensions);
            }

            Status append_linear(OnnxProgress& progress, const Interchange::OnnxNode& node,
                                 const OnnxValue& input,
                                 const std::vector<std::int64_t>& weight_extents,
                                 const std::vector<float>& weight, const std::vector<float>* bias)
            {
                SA_FAIL_IF(input.dimensions.size() != 1, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and a linear row reads one vector",
                           describe_node(node), input.dimensions.size());
                const std::int64_t outputs = weight_extents[0];
                const std::int64_t features = weight_extents[1];
                SA_FAIL_IF(features != input.dimensions[0], StatusCode::InvalidDescription,
                           "{}: the weight reads {} features, and the vector has {}",
                           describe_node(node), features, input.dimensions[0]);
                SA_FAIL_IF(bias != nullptr && static_cast<std::int64_t>(bias->size()) != outputs,
                           StatusCode::InvalidDescription,
                           "{}: the bias holds {} values for {} units",
                           describe_node(node), bias->size(), outputs);

                std::vector<float> placed = weight;
                if (!input.flattened_from.empty())
                {
                    SA_TRY_ASSIGN(placed, Interchange::flatten_weights_to_channels_last(
                                              SushiRuntime::span<const float>(weight.data(),
                                                                              weight.size()),
                                              outputs, input.flattened_from[0],
                                              input.flattened_from[1], input.flattened_from[2]));
                }

                const std::int64_t row = append_row(
                    progress, "linear", {"units", "use_bias"},
                    {Config::Value::integer(outputs), Config::Value::boolean(bias != nullptr)},
                    {input.row});
                SA_TRY(store_weight(progress, row, "weight", {outputs, features}, placed));
                if (bias != nullptr)
                    SA_TRY(store_weight(progress, row, "bias", {outputs}, *bias));
                define_value(progress, node.outputs[0], row, {outputs});
                return Status();
            }

            Status store_weight(OnnxProgress& progress, std::int64_t row, std::string_view field,
                                const std::vector<std::int64_t>& shape,
                                const std::vector<float>& values)
            {
                SA_TRY_ASSIGN(const Shape extents,
                              Shape::make(SushiRuntime::span<const std::int64_t>(shape.data(),
                                                                                 shape.size())));
                SA_FAIL_IF(static_cast<std::size_t>(extents.num_elements()) != values.size(),
                           StatusCode::InvalidDescription,
                           "the weight '{}.{}' has the extents {} and {} values", row, field,
                           extents.to_string(), values.size());
                IO::ImportedTensor tensor;
                tensor.record.name = std::to_string(row) + "." + std::string(field);
                tensor.record.dtype = DType::Float32;
                tensor.record.shape = extents;
                tensor.payload.resize(values.size() * sizeof(float));
                if (!values.empty())
                    std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
                progress.weights.push_back(std::move(tensor));
                return Status();
            }

            Config::Value pair_value(std::int64_t height, std::int64_t width)
            {
                if (height == width)
                    return Config::Value::integer(height);
                return Config::Value::array(
                    {Config::Value::integer(height), Config::Value::integer(width)});
            }

            Result<OnnxWindow> read_window(const Interchange::OnnxNode& node,
                                           const std::vector<std::int64_t>& kernel)
            {
                SA_TRY_ASSIGN(const std::string auto_pad,
                              read_text_attribute(node, "auto_pad", K_AUTO_PAD_NOT_SET));
                SA_FAIL_IF(auto_pad != K_AUTO_PAD_NOT_SET, StatusCode::Unsupported,
                           "{} states auto_pad '{}', and only explicit pads are translated",
                           describe_node(node), auto_pad);

                SA_TRY_ASSIGN(const std::vector<std::int64_t> stated_kernel,
                              read_pair(node, "kernel_shape", kernel, K_IMAGE_AXES));
                SA_FAIL_IF(stated_kernel.empty(), StatusCode::InvalidDescription,
                           "{} states no kernel_shape", describe_node(node));
                SA_FAIL_IF(!kernel.empty() && stated_kernel != kernel,
                           StatusCode::InvalidDescription,
                           "{}: kernel_shape is {} by {}, and the weight is {} by {}",
                           describe_node(node), stated_kernel[0], stated_kernel[1], kernel[0],
                           kernel[1]);
                SA_TRY_ASSIGN(const std::vector<std::int64_t> strides,
                              read_pair(node, "strides", {1, 1}, K_IMAGE_AXES));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> dilations,
                              read_pair(node, "dilations", {1, 1}, K_IMAGE_AXES));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> pads,
                              read_pair(node, "pads", {0, 0, 0, 0}, 2 * K_IMAGE_AXES));

                OnnxWindow window;
                for (std::size_t axis = 0; axis < K_IMAGE_AXES; ++axis)
                {
                    window.kernel[axis] = stated_kernel[axis];
                    window.stride[axis] = strides[axis];
                    window.dilation[axis] = dilations[axis];
                    window.pad_begin[axis] = pads[axis];
                    window.pad_end[axis] = pads[axis + K_IMAGE_AXES];
                    SA_FAIL_IF(window.kernel[axis] < 1 || window.stride[axis] < 1 ||
                                   window.dilation[axis] < 1 || window.pad_begin[axis] < 0 ||
                                   window.pad_end[axis] < 0,
                               StatusCode::InvalidDescription,
                               "{}: the window needs a positive kernel, stride and dilation and "
                               "pads of 0 or more",
                               describe_node(node));
                }
                return window;
            }

            Result<std::int64_t> window_extent(const Interchange::OnnxNode& node,
                                               const OnnxWindow& window, std::size_t axis,
                                               std::int64_t input)
            {
                const std::int64_t span = input + window.pad_begin[axis] + window.pad_end[axis] -
                                          window.dilation[axis] * (window.kernel[axis] - 1) - 1;
                SA_FAIL_IF(span < 0, StatusCode::InvalidDescription,
                           "{}: a window of {} does not fit an extent of {}", describe_node(node),
                           window.kernel[axis], input);
                return span / window.stride[axis] + 1;
            }

            std::int64_t default_opset(const OnnxProgress& progress) noexcept
            {
                return progress.graph->opset_version("");
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
