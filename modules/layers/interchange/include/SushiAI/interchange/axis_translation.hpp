/**************************************************************************/
/* axis_translation.hpp                                                   */
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
 * @file axis_translation.hpp
 * @brief Declares the axis and filter rewrites between channel-first and channel-last layouts.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        /**
         * @brief Returns the channel-last position of the channel-first axis @p axis.
         *
         * The channels (axis 1) move to the end and later axes move one place down.
         * @param axis A position in [-rank, rank); a negative one counts from the end.
         * @return The non-negative position; InvalidDescription for an axis out of range.
         */
        [[nodiscard]] Result<std::int64_t> channels_last_axis(std::int64_t axis,
                                                              std::int64_t rank) noexcept;

        /** @brief Applies channels_last_axis to every entry of @p axes. */
        [[nodiscard]] Result<std::vector<std::int64_t>>
        channels_last_axes(const std::vector<std::int64_t>& axes, std::int64_t rank);

        /**
         * @brief Rewrites the permutation of a Transpose.
         *
         * Output axis i of the channel-first node takes input axis @p permutation[i]; the
         * channel-last result has output axis channels_last_axis(i) take input axis
         * channels_last_axis(permutation[i]).
         * @return InvalidDescription when @p permutation is not a permutation of 0 to rank - 1.
         */
        [[nodiscard]] Result<std::vector<std::int64_t>>
        channels_last_permutation(const std::vector<std::int64_t>& permutation);

        /**
         * @brief Rewrites the target shape of a Reshape between two image-layout tensors.
         *
         * Each entry moves with its axis, so a 0 (copy the input extent at this position) and a
         * -1 (infer) keep their meaning. It holds when the input and the output are both
         * channel-first image tensors of the same rank.
         */
        [[nodiscard]] Result<std::vector<std::int64_t>>
        channels_last_shape(const std::vector<std::int64_t>& shape);

        /**
         * @brief Rewrites the axis of a Flatten when it keeps its meaning (0, 1 or the rank).
         *
         * A row comes out in channel-last order, so the importer permutes the weight after it.
         * @return Unsupported for an axis from 2 to rank - 1.
         */
        [[nodiscard]] Result<std::int64_t> channels_last_flatten_axis(std::int64_t axis,
                                                                      std::int64_t rank) noexcept;

        /** @brief Names how an attribute of an operator carries an axis. */
        enum class AxisAttributeKind : std::uint8_t
        {
            None,        /**< The attribute carries no axis. */
            Axis,        /**< One axis, such as Concat's axis or Softmax's axis. */
            Axes,        /**< A list of axes, such as Slice's or a reduction's axes. */
            Permutation, /**< Transpose's perm. */
            Shape,       /**< A target shape, such as Reshape's shape attribute. */
            FlattenAxis, /**< Flatten's axis. */
        };

        /**
         * @brief Returns how attribute @p attribute_name of operator @p op_type carries an axis.
         * @return None for an attribute that holds no axis, so it is copied as it is.
         */
        [[nodiscard]] AxisAttributeKind axis_attribute_kind(std::string_view op_type,
                                                            std::string_view attribute_name)
            noexcept;

        /**
         * @brief Rewrites the integers of one attribute from channel-first to channel-last.
         * @param values The attribute's integers; a single-valued attribute passes one.
         * @param rank The rank of the tensor the attribute refers to.
         * @return @p values unchanged for an attribute whose kind is None; the rewritten
         *         integers otherwise; InvalidDescription or Unsupported as the rewrite
         *         it applies.
         */
        [[nodiscard]] Result<std::vector<std::int64_t>>
        translate_axis_attribute(std::string_view op_type, std::string_view attribute_name,
                                 const std::vector<std::int64_t>& values, std::int64_t rank);

        /**
         * @brief Returns the extents of a convolution weight as the record stores it.
         * @param dimensions The weight stored [out, in, kh, kw].
         * @param depthwise Whether the weight is [C, 1, kh, kw], stored as [kh, kw, C].
         * @return [out, kh, kw, in], or [kh, kw, C] when @p depthwise; InvalidDescription for a
         *         rank other than 4, a non-positive extent, or a depthwise second extent not 1.
         */
        [[nodiscard]] Result<std::vector<std::int64_t>>
        filter_record_dimensions(const std::vector<std::int64_t>& dimensions, bool depthwise);

        /**
         * @brief Permutes a convolution weight from [out, in, kh, kw] to the record layout.
         * @param weights The weight's floats, row-major over @p dimensions.
         * @return The floats in the layout filter_record_dimensions names; InvalidDescription
         *         as filter_record_dimensions, or when @p weights holds a different count
         *         than the extents give.
         */
        [[nodiscard]] Result<std::vector<float>>
        filter_to_record_layout(SushiRuntime::span<const float> weights,
                                const std::vector<std::int64_t>& dimensions, bool depthwise);

        /**
         * @brief Moves the inputs of every output row of an inner-product weight from the order
         *        [C, H, W] of a channel-first blob to the order [H, W, C] a flatten row yields.
         * @param weights The weight's floats, [units, channels * height * width], row-major.
         * @return The floats in the [units, height * width * channels] order; InvalidDescription
         *         for a non-positive extent or a count other than the extents give.
         */
        [[nodiscard]] Result<std::vector<float>>
        flatten_weights_to_channels_last(SushiRuntime::span<const float> weights,
                                         std::int64_t units, std::int64_t channels,
                                         std::int64_t height, std::int64_t width);

        /**
         * @brief Returns the channel-first position of the channel-last axis @p axis, the
         *        inverse of channels_last_axis.
         * @param axis A position in [-rank, rank); a negative one counts from the end.
         * @return The non-negative position; InvalidDescription for an axis out of range.
         */
        [[nodiscard]] Result<std::int64_t> channels_first_axis(std::int64_t axis,
                                                               std::int64_t rank) noexcept;

        /**
         * @brief Rewrites a channel-last permutation to channel-first, the inverse of
         *        channels_last_permutation.
         * @return InvalidDescription when @p permutation is not a permutation of 0 to rank - 1.
         */
        [[nodiscard]] Result<std::vector<std::int64_t>>
        channels_first_permutation(const std::vector<std::int64_t>& permutation);

        /**
         * @brief Permutes a convolution weight from the record layout back to the layout of
         *        the source framework, the inverse of filter_to_record_layout.
         * @param weights The floats, row-major over @p record_dimensions.
         * @param record_dimensions [out, kh, kw, in], or [kh, kw, C] when @p depthwise.
         * @return The floats [out, in, kh, kw], or [C, 1, kh, kw]; InvalidDescription for a
         *         wrong rank, a non-positive extent or a count the extents do not give.
         */
        [[nodiscard]] Result<std::vector<float>>
        filter_to_source_layout(SushiRuntime::span<const float> weights,
                                const std::vector<std::int64_t>& record_dimensions,
                                bool depthwise);

        /**
         * @brief Moves the inputs of every output row of an inner-product weight from the order
         *        [H, W, C] of a flatten row to the order [C, H, W] of a channel-first blob, the
         *        inverse of flatten_weights_to_channels_last.
         * @param weights The floats, [units, height * width * channels], row-major.
         * @return The floats in the [units, channels * height * width] order; InvalidDescription
         *         for a non-positive extent or a count other than the extents give.
         */
        [[nodiscard]] Result<std::vector<float>>
        flatten_weights_to_channels_first(SushiRuntime::span<const float> weights,
                                          std::int64_t units, std::int64_t channels,
                                          std::int64_t height, std::int64_t width);
    } // namespace Interchange
} // namespace SushiAI
