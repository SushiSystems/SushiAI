/**************************************************************************/
/* axis_translation.cpp                                                   */
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
 * @file axis_translation.cpp
 * @brief Defines the rewrite of axis attributes and filters to the channel-last layout.
 * @author Mustafa Garip
 */

#include <SushiAI/interchange/axis_translation.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/filter_layout.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        namespace
        {
            /** @brief Holds the rank of a convolution weight. */
            constexpr std::size_t K_FILTER_RANK = 4;

            /** @brief Names one attribute of one operator that carries an axis. */
            struct AxisAttributeRow
            {
                std::string_view op_type;        /**< The operator. */
                std::string_view attribute_name; /**< The attribute. */
                AxisAttributeKind kind;          /**< How it carries the axis. */
            };

            /** @brief Lists every attribute the importers' operator sets carry an axis in. */
            const AxisAttributeRow K_AXIS_ATTRIBUTE_ROWS[] = {
                {"Concat", "axis", AxisAttributeKind::Axis},
                {"Softmax", "axis", AxisAttributeKind::Axis},
                {"LogSoftmax", "axis", AxisAttributeKind::Axis},
                {"Split", "axis", AxisAttributeKind::Axis},
                {"Gather", "axis", AxisAttributeKind::Axis},
                {"ArgMax", "axis", AxisAttributeKind::Axis},
                {"ArgMin", "axis", AxisAttributeKind::Axis},
                {"Slice", "axes", AxisAttributeKind::Axes},
                {"ReduceMean", "axes", AxisAttributeKind::Axes},
                {"ReduceSum", "axes", AxisAttributeKind::Axes},
                {"ReduceMax", "axes", AxisAttributeKind::Axes},
                {"ReduceMin", "axes", AxisAttributeKind::Axes},
                {"ReduceProd", "axes", AxisAttributeKind::Axes},
                {"ReduceL1", "axes", AxisAttributeKind::Axes},
                {"ReduceL2", "axes", AxisAttributeKind::Axes},
                {"ReduceSumSquare", "axes", AxisAttributeKind::Axes},
                {"ReduceLogSum", "axes", AxisAttributeKind::Axes},
                {"ReduceLogSumExp", "axes", AxisAttributeKind::Axes},
                {"Transpose", "perm", AxisAttributeKind::Permutation},
                {"Reshape", "shape", AxisAttributeKind::Shape},
                {"Flatten", "axis", AxisAttributeKind::FlattenAxis},
            };

            /** @brief Returns the channel-last position of the valid axis @p axis. */
            std::int64_t moved_axis(std::int64_t axis, std::int64_t rank) noexcept
            {
                if (axis == 0)
                    return 0;
                if (axis == 1)
                    return rank - 1;
                return axis - 1;
            }

            /** @brief Returns the channel-first position of the valid axis @p axis. */
            std::int64_t restored_axis(std::int64_t axis, std::int64_t rank) noexcept
            {
                if (axis == 0 || rank < 2)
                    return axis;
                if (axis == rank - 1)
                    return 1;
                return axis + 1;
            }

            /** @brief Returns InvalidDescription unless every extent is positive. */
            Status require_positive(const std::vector<std::int64_t>& extents)
            {
                for (std::int64_t extent : extents)
                    SA_FAIL_IF(extent <= 0, StatusCode::InvalidDescription,
                               "a convolution weight has the extent {}, which is not positive",
                               extent);
                return Status();
            }
        } // namespace

        Result<std::int64_t> channels_last_axis(std::int64_t axis, std::int64_t rank) noexcept
        {
            SA_FAIL_IF(axis < -rank || axis >= rank, StatusCode::InvalidDescription,
                       "the axis {} is out of range for a tensor of rank {}", axis, rank);
            const std::int64_t positive = axis < 0 ? axis + rank : axis;
            return moved_axis(positive, rank);
        }

        Result<std::vector<std::int64_t>>
        channels_last_axes(const std::vector<std::int64_t>& axes, std::int64_t rank)
        {
            std::vector<std::int64_t> out;
            out.reserve(axes.size());
            for (std::int64_t axis : axes)
            {
                SA_TRY_ASSIGN(const std::int64_t moved, channels_last_axis(axis, rank));
                out.push_back(moved);
            }
            return out;
        }

        Result<std::vector<std::int64_t>>
        channels_last_permutation(const std::vector<std::int64_t>& permutation)
        {
            const std::int64_t rank = static_cast<std::int64_t>(permutation.size());
            std::vector<std::int64_t> out(permutation.size(), -1);
            for (std::int64_t i = 0; i < rank; ++i)
            {
                const std::int64_t source = permutation[static_cast<std::size_t>(i)];
                SA_FAIL_IF(source < 0 || source >= rank, StatusCode::InvalidDescription,
                           "the permutation entry {} is out of range for rank {}", source, rank);
                out[static_cast<std::size_t>(moved_axis(i, rank))] = moved_axis(source, rank);
            }
            std::vector<bool> seen(permutation.size(), false);
            for (std::int64_t value : out)
            {
                SA_FAIL_IF(seen[static_cast<std::size_t>(value)], StatusCode::InvalidDescription,
                           "the permutation lists the axis {} twice", value);
                seen[static_cast<std::size_t>(value)] = true;
            }
            return out;
        }

        Result<std::vector<std::int64_t>>
        channels_last_shape(const std::vector<std::int64_t>& shape)
        {
            const std::int64_t rank = static_cast<std::int64_t>(shape.size());
            std::vector<std::int64_t> out(shape.size(), 0);
            for (std::int64_t i = 0; i < rank; ++i)
                out[static_cast<std::size_t>(moved_axis(i, rank))] =
                    shape[static_cast<std::size_t>(i)];
            return out;
        }

        Result<std::int64_t> channels_last_flatten_axis(std::int64_t axis,
                                                        std::int64_t rank) noexcept
        {
            SA_FAIL_IF(axis < -rank || axis > rank, StatusCode::InvalidDescription,
                       "the Flatten axis {} is out of range for a tensor of rank {}", axis, rank);
            const std::int64_t positive = axis < 0 ? axis + rank : axis;
            SA_FAIL_IF(positive > 1 && positive < rank, StatusCode::Unsupported,
                       "a Flatten at the axis {} of a tensor of rank {} splits the channels from "
                       "the rows, which the channel-last layout cannot state", axis, rank);
            return positive;
        }

        AxisAttributeKind axis_attribute_kind(std::string_view op_type,
                                              std::string_view attribute_name) noexcept
        {
            for (const AxisAttributeRow& row : K_AXIS_ATTRIBUTE_ROWS)
                if (row.op_type == op_type && row.attribute_name == attribute_name)
                    return row.kind;
            return AxisAttributeKind::None;
        }

        Result<std::vector<std::int64_t>>
        translate_axis_attribute(std::string_view op_type, std::string_view attribute_name,
                                 const std::vector<std::int64_t>& values, std::int64_t rank)
        {
            switch (axis_attribute_kind(op_type, attribute_name))
            {
                case AxisAttributeKind::None:
                    return values;
                case AxisAttributeKind::Axis:
                case AxisAttributeKind::Axes:
                    return channels_last_axes(values, rank);
                case AxisAttributeKind::Permutation:
                    return channels_last_permutation(values);
                case AxisAttributeKind::Shape:
                    return channels_last_shape(values);
                case AxisAttributeKind::FlattenAxis:
                {
                    SA_FAIL_IF(values.size() != 1, StatusCode::InvalidDescription,
                               "the Flatten axis holds {} values, expected 1", values.size());
                    SA_TRY_ASSIGN(const std::int64_t axis,
                                  channels_last_flatten_axis(values.front(), rank));
                    return std::vector<std::int64_t>{axis};
                }
            }
            return values;
        }

        Result<std::vector<std::int64_t>>
        filter_record_dimensions(const std::vector<std::int64_t>& dimensions, bool depthwise)
        {
            SA_FAIL_IF(dimensions.size() != K_FILTER_RANK, StatusCode::InvalidDescription,
                       "a convolution weight has 4 extents, this one has {}", dimensions.size());
            for (std::int64_t extent : dimensions)
                SA_FAIL_IF(extent <= 0, StatusCode::InvalidDescription,
                           "a convolution weight has the extent {}, which is not positive", extent);

            const std::int64_t outputs = dimensions[0];
            const std::int64_t inputs = dimensions[1];
            const std::int64_t height = dimensions[2];
            const std::int64_t width = dimensions[3];
            if (depthwise)
            {
                SA_FAIL_IF(inputs != 1, StatusCode::InvalidDescription,
                           "a depthwise weight reads one channel per filter, this one reads {}",
                           inputs);
                return std::vector<std::int64_t>{height, width, outputs};
            }
            return std::vector<std::int64_t>{outputs, height, width, inputs};
        }

        Result<std::vector<float>>
        filter_to_record_layout(SushiRuntime::span<const float> weights,
                                const std::vector<std::int64_t>& dimensions, bool depthwise)
        {
            SA_TRY_ASSIGN(const std::vector<std::int64_t> record,
                          filter_record_dimensions(dimensions, depthwise));
            IO::FilterExtents extents;
            extents.outputs = dimensions[0];
            extents.inputs = dimensions[1];
            extents.height = dimensions[2];
            extents.width = dimensions[3];
            const std::size_t count = static_cast<std::size_t>(extents.outputs * extents.inputs *
                                                               extents.height * extents.width);
            SA_FAIL_IF(weights.size() != count, StatusCode::InvalidDescription,
                       "the weight holds {} floats, its extents need {}", weights.size(), count);

            std::vector<float> out(count);
            if (depthwise)
                IO::copy_depthwise_kernel_last(weights.data(), extents, out.data());
            else
                IO::copy_kernel_last(weights.data(), extents, out.data());
            return out;
        }

        Result<std::vector<float>>
        flatten_weights_to_channels_last(SushiRuntime::span<const float> weights,
                                         std::int64_t units, std::int64_t channels,
                                         std::int64_t height, std::int64_t width)
        {
            SA_FAIL_IF(units <= 0 || channels <= 0 || height <= 0 || width <= 0,
                       StatusCode::InvalidDescription,
                       "an inner-product weight of {} units over {} by {} by {} needs positive "
                       "extents",
                       units, channels, height, width);
            const std::size_t inputs = static_cast<std::size_t>(channels * height * width);
            const std::size_t count = static_cast<std::size_t>(units) * inputs;
            SA_FAIL_IF(weights.size() != count, StatusCode::InvalidDescription,
                       "the weight holds {} floats, its extents need {}", weights.size(), count);

            const std::size_t pixels = static_cast<std::size_t>(height * width);
            const std::size_t channel_count = static_cast<std::size_t>(channels);
            std::vector<float> moved(count);
            for (std::size_t unit = 0; unit < static_cast<std::size_t>(units); ++unit)
            {
                const float* from = weights.data() + unit * inputs;
                float* to = moved.data() + unit * inputs;
                for (std::size_t channel = 0; channel < channel_count; ++channel)
                {
                    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
                        to[pixel * channel_count + channel] = from[channel * pixels + pixel];
                }
            }
            return moved;
        }

        Result<std::int64_t> channels_first_axis(std::int64_t axis, std::int64_t rank) noexcept
        {
            SA_FAIL_IF(axis < -rank || axis >= rank, StatusCode::InvalidDescription,
                       "the axis {} is out of range for a tensor of rank {}", axis, rank);
            const std::int64_t positive = axis < 0 ? axis + rank : axis;
            return restored_axis(positive, rank);
        }

        Result<std::vector<std::int64_t>>
        channels_first_permutation(const std::vector<std::int64_t>& permutation)
        {
            const std::int64_t rank = static_cast<std::int64_t>(permutation.size());
            std::vector<std::int64_t> out(permutation.size(), -1);
            for (std::int64_t i = 0; i < rank; ++i)
            {
                const std::int64_t source = permutation[static_cast<std::size_t>(i)];
                SA_FAIL_IF(source < 0 || source >= rank, StatusCode::InvalidDescription,
                           "the permutation entry {} is out of range for rank {}", source, rank);
                out[static_cast<std::size_t>(restored_axis(i, rank))] =
                    restored_axis(source, rank);
            }
            std::vector<bool> seen(permutation.size(), false);
            for (std::int64_t value : out)
            {
                SA_FAIL_IF(seen[static_cast<std::size_t>(value)], StatusCode::InvalidDescription,
                           "the permutation lists the axis {} twice", value);
                seen[static_cast<std::size_t>(value)] = true;
            }
            return out;
        }

        Result<std::vector<float>>
        filter_to_source_layout(SushiRuntime::span<const float> weights,
                                const std::vector<std::int64_t>& record_dimensions,
                                bool depthwise)
        {
            const std::size_t rank = depthwise ? 3 : K_FILTER_RANK;
            SA_FAIL_IF(record_dimensions.size() != rank, StatusCode::InvalidDescription,
                       "the record of a {} weight has {} extents, this one has {}",
                       depthwise ? "depthwise" : "convolution", rank, record_dimensions.size());
            SA_TRY(require_positive(record_dimensions));

            IO::FilterExtents extents;
            if (depthwise)
            {
                extents.outputs = record_dimensions[2];
                extents.inputs = 1;
                extents.height = record_dimensions[0];
                extents.width = record_dimensions[1];
            }
            else
            {
                extents.outputs = record_dimensions[0];
                extents.inputs = record_dimensions[3];
                extents.height = record_dimensions[1];
                extents.width = record_dimensions[2];
            }
            const std::size_t count = static_cast<std::size_t>(extents.outputs * extents.inputs *
                                                               extents.height * extents.width);
            SA_FAIL_IF(weights.size() != count, StatusCode::InvalidDescription,
                       "the weight holds {} floats, its extents need {}", weights.size(), count);

            std::vector<float> out(count);
            if (depthwise)
                IO::copy_depthwise_kernel_first(weights.data(), extents, out.data());
            else
                IO::copy_kernel_first(weights.data(), extents, out.data());
            return out;
        }

        Result<std::vector<float>>
        flatten_weights_to_channels_first(SushiRuntime::span<const float> weights,
                                          std::int64_t units, std::int64_t channels,
                                          std::int64_t height, std::int64_t width)
        {
            SA_FAIL_IF(units <= 0 || channels <= 0 || height <= 0 || width <= 0,
                       StatusCode::InvalidDescription,
                       "an inner-product weight of {} units over {} by {} by {} needs positive "
                       "extents",
                       units, channels, height, width);
            const std::size_t inputs = static_cast<std::size_t>(channels * height * width);
            const std::size_t count = static_cast<std::size_t>(units) * inputs;
            SA_FAIL_IF(weights.size() != count, StatusCode::InvalidDescription,
                       "the weight holds {} floats, its extents need {}", weights.size(), count);

            const std::size_t pixels = static_cast<std::size_t>(height * width);
            const std::size_t channel_count = static_cast<std::size_t>(channels);
            std::vector<float> moved(count);
            for (std::size_t unit = 0; unit < static_cast<std::size_t>(units); ++unit)
            {
                const float* from = weights.data() + unit * inputs;
                float* to = moved.data() + unit * inputs;
                for (std::size_t channel = 0; channel < channel_count; ++channel)
                {
                    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
                        to[channel * pixels + pixel] = from[pixel * channel_count + channel];
                }
            }
            return moved;
        }
    } // namespace Interchange
} // namespace SushiAI
