/**************************************************************************/
/* permuted_copy.hpp                                                      */
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
 * @file permuted_copy.hpp
 * @brief Declares the plan of a copy that permutes the non-batch axes of a contiguous value.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Holds the number of axes the strided copy of SushiBLAS reads. */
        inline constexpr std::size_t K_COPY_AXES = 4;

        /** @brief Holds one axis of the copy, in the order of the result. */
        struct CopyAxis
        {
            std::int64_t extent = 1; /**< Elements along the axis. */
            std::int64_t stride = 0; /**< Step in the source between two of them. */
        };

        /** @brief Holds the axes a permuting copy walks and whether they fit in four. */
        struct PermutedCopy
        {
            /** @brief The axes in result order, padded in front with axes of extent 1. */
            std::array<CopyAxis, K_COPY_AXES> axes{};

            /** @brief Whether the permutation needs no more than K_COPY_AXES runs of axes. */
            bool fits = true;
        };

        /**
         * @brief Reports whether @p permutation lists each of 0 .. size - 1 exactly once.
         * @param permutation Axes of a value, the batch axis left out.
         * @return True for a permutation of at most K_MAX_RANK - 1 axes.
         */
        [[nodiscard]] inline bool is_axis_permutation(
            SushiRuntime::span<const std::int64_t> permutation) noexcept
        {
            if (permutation.size() >= static_cast<std::size_t>(K_MAX_RANK))
            {
                return false;
            }

            std::array<bool, static_cast<std::size_t>(K_MAX_RANK)> seen{};
            for (const std::int64_t axis : permutation)
            {
                if (axis < 0 || axis >= static_cast<std::int64_t>(permutation.size())
                    || seen[static_cast<std::size_t>(axis)])
                {
                    return false;
                }

                seen[static_cast<std::size_t>(axis)] = true;
            }

            return true;
        }

        /**
         * @brief Returns the permutation that undoes @p permutation.
         * @pre @p permutation passes is_axis_permutation.
         * @return The inverse in its first permutation.size() entries.
         */
        [[nodiscard]] inline std::array<std::int64_t, static_cast<std::size_t>(K_MAX_RANK)>
        inverse_axis_permutation(SushiRuntime::span<const std::int64_t> permutation) noexcept
        {
            std::array<std::int64_t, static_cast<std::size_t>(K_MAX_RANK)> inverse{};
            for (std::size_t axis = 0; axis < permutation.size(); ++axis)
            {
                inverse[static_cast<std::size_t>(permutation[axis])] =
                    static_cast<std::int64_t>(axis);
            }

            return inverse;
        }

        /**
         * @brief Reads the permutation a K_TRANSPOSE node holds in its attribute slots.
         * @param axes The per-sample axes of the operand, its rank less the batch axis.
         * @return Entry k in slot k, in the first @p axes entries.
         */
        [[nodiscard]] inline std::array<std::int64_t, static_cast<std::size_t>(K_MAX_RANK)>
        permutation_of(const Node& node, std::size_t axes) noexcept
        {
            std::array<std::int64_t, static_cast<std::size_t>(K_MAX_RANK)> permutation{};
            for (std::size_t axis = 0; axis < axes; ++axis)
            {
                permutation[axis] = static_cast<std::int64_t>(node.attributes.get(axis));
            }

            return permutation;
        }

        /**
         * @brief Plans the copy of a contiguous @p source into the value its non-batch axes
         *        take in the order @p permutation names.
         * @param source The shape of the operand; the batch axis comes first and stays first.
         * @param permutation Entry k names the operand axis that becomes result axis k.
         * @pre @p source has a rank of at least 2; @p permutation passes is_axis_permutation.
         * @return The axes in result order, extent-1 axes dropped and operand neighbours joined.
         */
        [[nodiscard]] inline PermutedCopy plan_permuted_copy(
            const Shape& source, SushiRuntime::span<const std::int64_t> permutation) noexcept
        {
            const std::size_t rank = static_cast<std::size_t>(source.rank());

            std::array<std::int64_t, static_cast<std::size_t>(K_MAX_RANK)> strides{};
            std::int64_t running = 1;
            for (std::size_t axis = rank; axis-- > 0;)
            {
                strides[axis] = running;
                running *= source[static_cast<std::int32_t>(axis)];
            }

            std::array<CopyAxis, static_cast<std::size_t>(K_MAX_RANK)> runs{};
            std::size_t count = 0;
            for (std::size_t position = 0; position < rank; ++position)
            {
                const std::size_t axis =
                    position == 0 ? 0 : static_cast<std::size_t>(permutation[position - 1]) + 1;
                const std::int64_t extent = source[static_cast<std::int32_t>(axis)];
                if (extent == 1)
                {
                    continue;
                }

                if (count > 0 && runs[count - 1].stride == strides[axis] * extent)
                {
                    runs[count - 1].extent *= extent;
                    runs[count - 1].stride = strides[axis];
                    continue;
                }

                runs[count++] = CopyAxis{extent, strides[axis]};
            }

            PermutedCopy plan;
            plan.fits = count <= K_COPY_AXES;
            if (!plan.fits)
            {
                return plan;
            }

            // A value of one element has no run; it copies as one axis of extent 1.
            if (count == 0)
            {
                runs[count++] = CopyAxis{1, 1};
            }

            // Padding axes sit in front, each as wide as everything behind it.
            const std::size_t padding = K_COPY_AXES - count;
            for (std::size_t axis = 0; axis < count; ++axis)
            {
                plan.axes[padding + axis] = runs[axis];
            }

            for (std::size_t axis = padding; axis-- > 0;)
            {
                plan.axes[axis] = CopyAxis{
                    1, plan.axes[axis + 1].stride * plan.axes[axis + 1].extent};
            }

            return plan;
        }
    } // namespace Graph
} // namespace SushiAI
