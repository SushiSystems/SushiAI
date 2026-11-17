/**************************************************************************/
/* constant_folding.cpp                                                   */
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
 * @file constant_folding.cpp
 * @brief Defines the evaluation of the shape arithmetic an ONNX export leaves in its nodes.
 * @author Mustafa Garip
 */

#include <SushiAI/interchange/constant_folding.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        namespace
        {
            /** @brief Holds the most elements a constant the folder reads or computes may have. */
            constexpr std::size_t K_MAX_FOLDED_ELEMENTS = 1024;

            /** @brief Holds the number ONNX gives the int32 element type. */
            constexpr std::int64_t K_CAST_INT32 = 6;

            /** @brief Holds the number ONNX gives the int64 element type. */
            constexpr std::int64_t K_CAST_INT64 = 7;

            /** @brief Holds the number ONNX gives the float32 element type. */
            constexpr std::int64_t K_CAST_FLOAT = 1;

            /** @brief Returns the product of @p dimensions from @p first up to @p last. */
            std::size_t product(const std::vector<std::int64_t>& dimensions, std::size_t first,
                                std::size_t last) noexcept
            {
                std::size_t count = 1;
                for (std::size_t i = first; i < last; ++i)
                    count *= static_cast<std::size_t>(dimensions[i]);
                return count;
            }

            /** @brief Returns the signed little-endian integer of @p width bytes at @p at. */
            std::int64_t read_signed(const std::byte* at, std::size_t width) noexcept
            {
                std::uint64_t value = 0;
                for (std::size_t i = 0; i < width; ++i)
                    value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(at[i]))
                             << (8u * i);
                if (width < 8 && (value >> (8u * width - 1u)) != 0)
                    value |= ~std::uint64_t(0) << (8u * width);
                return static_cast<std::int64_t>(value);
            }

            /** @brief Returns the constant @p tensor holds, or nothing when it is unreadable. */
            std::optional<ConstantValue> decode_tensor(const OnnxTensor& tensor)
            {
                const std::size_t count = tensor.element_count();
                if (count > K_MAX_FOLDED_ELEMENTS)
                    return std::nullopt;
                const std::size_t width = onnx_element_bytes(tensor.data_type);
                const SushiRuntime::span<const std::byte> bytes = tensor.data.bytes();
                if (bytes.size() != count * width)
                    return std::nullopt;

                ConstantValue out;
                out.dimensions = tensor.dimensions;
                if (tensor.data_type == OnnxDataType::Int64 ||
                    tensor.data_type == OnnxDataType::Int32)
                {
                    out.kind = ConstantKind::Integer;
                    for (std::size_t i = 0; i < count; ++i)
                        out.integers.push_back(read_signed(bytes.data() + i * width, width));
                    return out;
                }
                if (tensor.data_type == OnnxDataType::Float)
                {
                    out.kind = ConstantKind::Real;
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        const std::uint32_t bits = static_cast<std::uint32_t>(
                            read_signed(bytes.data() + i * width, width));
                        float value = 0.0f;
                        std::memcpy(&value, &bits, sizeof(value));
                        out.reals.push_back(value);
                    }
                    return out;
                }
                return std::nullopt;
            }

            /** @brief Returns the number of elements of @p value. */
            std::size_t element_count(const ConstantValue& value) noexcept
            {
                return value.kind == ConstantKind::Integer ? value.integers.size()
                                                           : value.reals.size();
            }

            /** @brief Returns a one-dimensional integer constant holding @p values. */
            ConstantValue integer_vector(std::vector<std::int64_t> values)
            {
                ConstantValue out;
                out.kind = ConstantKind::Integer;
                out.dimensions = {static_cast<std::int64_t>(values.size())};
                out.integers = std::move(values);
                return out;
            }

            /** @brief Holds what the evaluators know: constants, and the shapes of tensors. */
            class Environment
            {
                public:
                    /** @brief Constructs the environment over @p graph, which must outlive it. */
                    explicit Environment(const OnnxGraph& graph) : graph_(graph)
                    {
                        for (const OnnxValueInfo& input : graph.inputs)
                        {
                            const bool static_shape =
                                input.has_shape &&
                                std::all_of(input.dimensions.begin(), input.dimensions.end(),
                                            [](const OnnxDimension& d) { return d.value >= 0; });
                            if (!static_shape)
                                continue;
                            std::vector<std::int64_t> shape;
                            for (const OnnxDimension& d : input.dimensions)
                                shape.push_back(d.value);
                            shapes_[input.name] = std::move(shape);
                        }
                        for (const OnnxTensor& tensor : graph.initializers)
                            shapes_[tensor.name] = tensor.dimensions;
                    }

                    /** @brief Returns the constant named @p name, or null when it is none. */
                    const ConstantValue* find(const std::string& name)
                    {
                        const auto known = values_.find(name);
                        if (known != values_.end())
                            return &known->second;
                        const OnnxTensor* initializer = graph_.find_initializer(name);
                        if (initializer == nullptr)
                            return nullptr;
                        std::optional<ConstantValue> decoded = decode_tensor(*initializer);
                        if (!decoded.has_value())
                            return nullptr;
                        return &(values_[name] = std::move(*decoded));
                    }

                    /** @brief Returns the fully stated shape of @p name, or null when it is not. */
                    const std::vector<std::int64_t>* shape_of(const std::string& name) const
                    {
                        const auto constant = values_.find(name);
                        if (constant != values_.end())
                            return &constant->second.dimensions;
                        const auto shape = shapes_.find(name);
                        return shape == shapes_.end() ? nullptr : &shape->second;
                    }

                    /** @brief Records @p value as the constant @p name. */
                    void set(const std::string& name, ConstantValue value)
                    {
                        values_[name] = std::move(value);
                    }

                private:
                    /** @brief Holds the graph whose initializers are read on demand. */
                    const OnnxGraph& graph_;

                    /** @brief Holds the constants found or computed so far. */
                    std::map<std::string, ConstantValue> values_;

                    /** @brief Holds the stated shapes of the graph inputs and initializers. */
                    std::map<std::string, std::vector<std::int64_t>> shapes_;
            };

            /** @brief Names one evaluation: true when it stored its outputs in the environment. */
            using Evaluator = Result<bool> (*)(const OnnxNode& node, Environment& environment);

            /** @brief Returns the constant at input @p index of @p node, or null. */
            const ConstantValue* constant_input(const OnnxNode& node, Environment& environment,
                                                std::size_t index)
            {
                if (index >= node.inputs.size() || node.inputs[index].empty())
                    return nullptr;
                return environment.find(node.inputs[index]);
            }

            /** @brief Returns the integer attribute @p name of @p node, or @p fallback. */
            std::int64_t integer_attribute(const OnnxNode& node, std::string_view name,
                                           std::int64_t fallback) noexcept
            {
                const OnnxAttribute* attribute = node.find_attribute(name);
                if (attribute == nullptr || attribute->type != OnnxAttributeType::Int)
                    return fallback;
                return attribute->integer;
            }

            /** @brief Returns the axis @p axis of a tensor of rank @p rank as a position, or -1. */
            std::int64_t normalize_axis(std::int64_t axis, std::int64_t rank) noexcept
            {
                if (axis < -rank || axis >= rank)
                    return -1;
                return axis < 0 ? axis + rank : axis;
            }

            /** @brief Copies the elements of @p source named by @p indices along an axis. */
            template<typename T>
            std::vector<T> gather_elements(const std::vector<T>& source, std::size_t outer,
                                           std::size_t axis_extent, std::size_t inner,
                                           const std::vector<std::size_t>& indices)
            {
                std::vector<T> out;
                out.reserve(outer * indices.size() * inner);
                for (std::size_t o = 0; o < outer; ++o)
                    for (std::size_t index : indices)
                        for (std::size_t i = 0; i < inner; ++i)
                            out.push_back(source[(o * axis_extent + index) * inner + i]);
                return out;
            }

            /** @brief Joins @p parts along an axis, each @p extents[k] long on it. */
            template<typename T>
            std::vector<T> concat_elements(const std::vector<const std::vector<T>*>& parts,
                                           const std::vector<std::size_t>& extents,
                                           std::size_t outer, std::size_t inner)
            {
                std::vector<T> out;
                for (std::size_t o = 0; o < outer; ++o)
                    for (std::size_t k = 0; k < parts.size(); ++k)
                    {
                        const std::size_t chunk = extents[k] * inner;
                        const auto first =
                            parts[k]->begin() + static_cast<std::ptrdiff_t>(o * chunk);
                        out.insert(out.end(), first, first + static_cast<std::ptrdiff_t>(chunk));
                    }
                return out;
            }

            /** @brief Returns the constant a Constant node's value attribute holds, if any. */
            std::optional<ConstantValue> constant_attribute_value(const OnnxNode& node)
            {
                if (const OnnxAttribute* tensor = node.find_attribute("value"))
                    return decode_tensor(tensor->tensor);
                if (const OnnxAttribute* scalar = node.find_attribute("value_int"))
                {
                    ConstantValue out;
                    out.integers = {scalar->integer};
                    return out;
                }
                if (const OnnxAttribute* list = node.find_attribute("value_ints"))
                    return integer_vector(list->integers);
                if (const OnnxAttribute* scalar = node.find_attribute("value_float"))
                {
                    ConstantValue out;
                    out.kind = ConstantKind::Real;
                    out.reals = {scalar->real};
                    return out;
                }
                if (const OnnxAttribute* list = node.find_attribute("value_floats"))
                {
                    ConstantValue out;
                    out.kind = ConstantKind::Real;
                    out.dimensions = {static_cast<std::int64_t>(list->reals.size())};
                    out.reals = list->reals;
                    return out;
                }
                return std::nullopt;
            }

            /** @brief Evaluates Constant from its value attribute. */
            Result<bool> evaluate_constant(const OnnxNode& node, Environment& environment)
            {
                if (node.outputs.size() != 1)
                    return false;
                std::optional<ConstantValue> value = constant_attribute_value(node);
                if (!value.has_value() || element_count(*value) > K_MAX_FOLDED_ELEMENTS)
                    return false;
                environment.set(node.outputs.front(), std::move(*value));
                return true;
            }

            /** @brief Evaluates Shape of a tensor whose extents are all stated. */
            Result<bool> evaluate_shape(const OnnxNode& node, Environment& environment)
            {
                if (node.inputs.size() != 1 || node.outputs.size() != 1)
                    return false;
                const std::vector<std::int64_t>* shape = environment.shape_of(node.inputs.front());
                if (shape == nullptr)
                    return false;
                const std::int64_t rank = static_cast<std::int64_t>(shape->size());
                std::int64_t start = integer_attribute(node, "start", 0);
                std::int64_t end = integer_attribute(node, "end", rank);
                start = std::clamp(start < 0 ? start + rank : start, std::int64_t(0), rank);
                end = std::clamp(end < 0 ? end + rank : end, std::int64_t(0), rank);
                std::vector<std::int64_t> slice;
                for (std::int64_t i = start; i < end; ++i)
                    slice.push_back((*shape)[static_cast<std::size_t>(i)]);
                environment.set(node.outputs.front(), integer_vector(std::move(slice)));
                return true;
            }

            /** @brief Evaluates Gather over constants. */
            Result<bool> evaluate_gather(const OnnxNode& node, Environment& environment)
            {
                const ConstantValue* data = constant_input(node, environment, 0);
                const ConstantValue* indices = constant_input(node, environment, 1);
                if (data == nullptr || indices == nullptr || node.outputs.size() != 1 ||
                    indices->kind != ConstantKind::Integer)
                    return false;
                const std::int64_t rank = static_cast<std::int64_t>(data->dimensions.size());
                const std::int64_t axis = normalize_axis(integer_attribute(node, "axis", 0), rank);
                SA_FAIL_IF(axis < 0, StatusCode::InvalidDescription,
                           "node '{}' (Gather): the axis is out of range for rank {}", node.name,
                           rank);

                const std::size_t axis_position = static_cast<std::size_t>(axis);
                const std::size_t axis_extent =
                    static_cast<std::size_t>(data->dimensions[axis_position]);
                std::vector<std::size_t> positions;
                for (std::int64_t index : indices->integers)
                {
                    const std::int64_t wrapped =
                        index < 0 ? index + static_cast<std::int64_t>(axis_extent) : index;
                    SA_FAIL_IF(wrapped < 0 || wrapped >= static_cast<std::int64_t>(axis_extent),
                               StatusCode::InvalidDescription,
                               "node '{}' (Gather): the index {} is out of range for an extent "
                               "of {}", node.name, index, axis_extent);
                    positions.push_back(static_cast<std::size_t>(wrapped));
                }

                const std::size_t outer = product(data->dimensions, 0, axis_position);
                const std::size_t inner = product(data->dimensions, axis_position + 1,
                                                  data->dimensions.size());
                ConstantValue out;
                out.kind = data->kind;
                out.dimensions.assign(data->dimensions.begin(),
                                      data->dimensions.begin() + axis);
                out.dimensions.insert(out.dimensions.end(), indices->dimensions.begin(),
                                      indices->dimensions.end());
                out.dimensions.insert(out.dimensions.end(), data->dimensions.begin() + axis + 1,
                                      data->dimensions.end());
                if (data->kind == ConstantKind::Integer)
                    out.integers = gather_elements(data->integers, outer, axis_extent, inner,
                                                   positions);
                else
                    out.reals = gather_elements(data->reals, outer, axis_extent, inner, positions);
                environment.set(node.outputs.front(), std::move(out));
                return true;
            }

            /** @brief Evaluates Unsqueeze over a constant; the axes are an attribute or input. */
            Result<bool> evaluate_unsqueeze(const OnnxNode& node, Environment& environment)
            {
                const ConstantValue* data = constant_input(node, environment, 0);
                if (data == nullptr || node.outputs.size() != 1)
                    return false;
                std::vector<std::int64_t> axes;
                if (const OnnxAttribute* attribute = node.find_attribute("axes"))
                {
                    axes = attribute->integers;
                }
                else
                {
                    const ConstantValue* from_input = constant_input(node, environment, 1);
                    if (from_input == nullptr || from_input->kind != ConstantKind::Integer)
                        return false;
                    axes = from_input->integers;
                }

                const std::int64_t out_rank =
                    static_cast<std::int64_t>(data->dimensions.size() + axes.size());
                std::vector<bool> inserted(static_cast<std::size_t>(out_rank), false);
                for (std::int64_t axis : axes)
                {
                    const std::int64_t position = normalize_axis(axis, out_rank);
                    SA_FAIL_IF(position < 0 || inserted[static_cast<std::size_t>(position)],
                               StatusCode::InvalidDescription,
                               "node '{}' (Unsqueeze): the axis {} is out of range or repeated",
                               node.name, axis);
                    inserted[static_cast<std::size_t>(position)] = true;
                }
                ConstantValue out = *data;
                out.dimensions.clear();
                std::size_t next = 0;
                for (bool is_new : inserted)
                    out.dimensions.push_back(is_new ? 1 : data->dimensions[next++]);
                environment.set(node.outputs.front(), std::move(out));
                return true;
            }

            /** @brief Evaluates Concat over constants of one element kind. */
            Result<bool> evaluate_concat(const OnnxNode& node, Environment& environment)
            {
                if (node.outputs.size() != 1 || node.inputs.empty())
                    return false;
                std::vector<const ConstantValue*> parts;
                for (std::size_t i = 0; i < node.inputs.size(); ++i)
                {
                    const ConstantValue* part = constant_input(node, environment, i);
                    if (part == nullptr)
                        return false;
                    parts.push_back(part);
                }
                const ConstantValue& first = *parts.front();
                const std::int64_t rank = static_cast<std::int64_t>(first.dimensions.size());
                const std::int64_t axis = normalize_axis(integer_attribute(node, "axis", 0), rank);
                SA_FAIL_IF(axis < 0, StatusCode::InvalidDescription,
                           "node '{}' (Concat): the axis is out of range for rank {}", node.name,
                           rank);

                const std::size_t axis_position = static_cast<std::size_t>(axis);
                std::vector<std::size_t> extents;
                std::size_t total = 0;
                for (const ConstantValue* part : parts)
                {
                    bool same = part->kind == first.kind &&
                                part->dimensions.size() == first.dimensions.size();
                    for (std::size_t d = 0; same && d < first.dimensions.size(); ++d)
                        same = d == axis_position || part->dimensions[d] == first.dimensions[d];
                    SA_FAIL_IF(!same, StatusCode::InvalidDescription,
                               "node '{}' (Concat): the inputs differ in element type, rank or an "
                               "extent off the axis", node.name);
                    extents.push_back(static_cast<std::size_t>(part->dimensions[axis_position]));
                    total += extents.back();
                }

                const std::size_t outer = product(first.dimensions, 0, axis_position);
                const std::size_t inner = product(first.dimensions, axis_position + 1,
                                                  first.dimensions.size());
                ConstantValue out;
                out.kind = first.kind;
                out.dimensions = first.dimensions;
                out.dimensions[axis_position] = static_cast<std::int64_t>(total);
                if (first.kind == ConstantKind::Integer)
                {
                    std::vector<const std::vector<std::int64_t>*> lists;
                    for (const ConstantValue* part : parts)
                        lists.push_back(&part->integers);
                    out.integers = concat_elements(lists, extents, outer, inner);
                }
                else
                {
                    std::vector<const std::vector<float>*> lists;
                    for (const ConstantValue* part : parts)
                        lists.push_back(&part->reals);
                    out.reals = concat_elements(lists, extents, outer, inner);
                }
                environment.set(node.outputs.front(), std::move(out));
                return true;
            }

            /** @brief Evaluates Cast between int32, int64 and float32. */
            Result<bool> evaluate_cast(const OnnxNode& node, Environment& environment)
            {
                const ConstantValue* data = constant_input(node, environment, 0);
                if (data == nullptr || node.outputs.size() != 1)
                    return false;
                const std::int64_t target = integer_attribute(node, "to", 0);
                ConstantValue out;
                out.dimensions = data->dimensions;
                if (target == K_CAST_INT64 || target == K_CAST_INT32)
                {
                    out.kind = ConstantKind::Integer;
                    if (data->kind == ConstantKind::Integer)
                        out.integers = data->integers;
                    else
                        for (float value : data->reals)
                            out.integers.push_back(static_cast<std::int64_t>(value));
                }
                else if (target == K_CAST_FLOAT)
                {
                    out.kind = ConstantKind::Real;
                    if (data->kind == ConstantKind::Real)
                        out.reals = data->reals;
                    else
                        for (std::int64_t value : data->integers)
                            out.reals.push_back(static_cast<float>(value));
                }
                else
                {
                    return false;
                }
                environment.set(node.outputs.front(), std::move(out));
                return true;
            }

            /** @brief Names an operator the folder evaluates and the function that does it. */
            struct EvaluatorRow
            {
                std::string_view op_type; /**< The operator. */
                Evaluator evaluate;       /**< Its evaluation. */
            };

            /** @brief Lists the operators of the shape arithmetic. */
            const EvaluatorRow K_EVALUATOR_ROWS[] = {
                {"Constant", evaluate_constant}, {"Shape", evaluate_shape},
                {"Gather", evaluate_gather},     {"Unsqueeze", evaluate_unsqueeze},
                {"Concat", evaluate_concat},     {"Cast", evaluate_cast},
            };

            /** @brief Returns the evaluator of @p op_type in the default domain, or null. */
            Evaluator evaluator_of(const OnnxNode& node) noexcept
            {
                if (!node.domain.empty() && node.domain != "ai.onnx")
                    return nullptr;
                for (const EvaluatorRow& row : K_EVALUATOR_ROWS)
                    if (row.op_type == node.op_type)
                        return row.evaluate;
                return nullptr;
            }

            /** @brief Returns whether @p node is an operator whose inputs the folding feeds. */
            bool is_sink(const OnnxNode& node) noexcept
            {
                return node.op_type == "Reshape" || node.op_type == "Resize" ||
                       node.op_type == "Expand";
            }
        } // namespace

        Result<FoldedGraph> fold_constants(const OnnxGraph& graph)
        {
            Environment environment(graph);
            const std::size_t count = graph.nodes.size();

            std::vector<bool> computable(count, false);
            for (std::size_t i = 0; i < count; ++i)
            {
                const Evaluator evaluate = evaluator_of(graph.nodes[i]);
                if (evaluate == nullptr)
                    continue;
                SA_TRY_ASSIGN(const bool done, evaluate(graph.nodes[i], environment));
                computable[i] = done;
            }

            std::map<std::string, std::vector<std::size_t>> consumers;
            for (std::size_t i = 0; i < count; ++i)
                for (const std::string& input : graph.nodes[i].inputs)
                    consumers[input].push_back(i);
            std::set<std::string> graph_outputs;
            for (const OnnxValueInfo& output : graph.outputs)
                graph_outputs.insert(output.name);

            // The walk runs backwards so the readers of a node are decided before it.
            std::vector<bool> folded(count, false);
            for (std::size_t i = count; i-- > 0;)
            {
                if (!computable[i])
                    continue;
                bool read = false;
                bool allowed = true;
                for (const std::string& output : graph.nodes[i].outputs)
                {
                    allowed = allowed && graph_outputs.count(output) == 0;
                    const auto found = consumers.find(output);
                    if (found == consumers.end())
                        continue;
                    for (std::size_t reader : found->second)
                    {
                        read = true;
                        allowed = allowed && (folded[reader] || is_sink(graph.nodes[reader]));
                    }
                }
                folded[i] = read && allowed;
            }

            FoldedGraph out;
            std::set<std::string> used;
            for (std::size_t i = 0; i < count; ++i)
            {
                if (folded[i])
                {
                    ++out.folded_count;
                    continue;
                }
                out.nodes.push_back(graph.nodes[i]);
                for (const std::string& input : graph.nodes[i].inputs)
                    used.insert(input);
            }
            for (std::size_t i = 0; i < count; ++i)
            {
                if (!folded[i])
                    continue;
                for (const std::string& output : graph.nodes[i].outputs)
                {
                    const ConstantValue* value = environment.find(output);
                    if (value != nullptr && used.count(output) != 0)
                        out.constants.push_back(NamedConstant{output, *value});
                }
            }
            return out;
        }
    } // namespace Interchange
} // namespace SushiAI
