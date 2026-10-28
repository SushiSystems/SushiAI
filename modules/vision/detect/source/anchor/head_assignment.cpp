/**************************************************************************/
/* head_assignment.cpp                                                    */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file head_assignment.cpp
 * @brief Implements the assignment of a model's anchors to its heads by area and stride.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_assignment.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Reports whether @p side is a finite length above zero. */
            bool is_positive_side(float side) noexcept
            {
                return std::isfinite(side) && side > 0.0f;
            }

            /** @brief Returns the area of @p anchor in double. */
            double measure_area(AnchorSize anchor) noexcept
            {
                return static_cast<double>(anchor.width) * static_cast<double>(anchor.height);
            }

            /** @brief Checks that there are anchors and heads, each usable, and enough anchors. */
            Status check_anchors_and_strides(SushiRuntime::span<const AnchorSize> anchors,
                                             SushiRuntime::span<const std::int64_t> strides)
            {
                SA_FAIL_IF(anchors.empty(), StatusCode::InvalidDescription,
                           "a head assignment needs at least one anchor");
                SA_FAIL_IF(strides.empty(), StatusCode::InvalidDescription,
                           "a head assignment needs at least one stride");

                for (std::size_t index = 0; index < anchors.size(); ++index)
                {
                    const AnchorSize anchor = anchors[index];
                    SA_FAIL_IF(!is_positive_side(anchor.width) || !is_positive_side(anchor.height),
                               StatusCode::InvalidDescription,
                               "anchor {} must have finite sides above zero; got {} by {}", index,
                               anchor.width, anchor.height);
                }
                for (std::size_t head = 0; head < strides.size(); ++head)
                {
                    SA_FAIL_IF(strides[head] <= 0, StatusCode::InvalidDescription,
                               "the stride of head {} must be above zero; got {}", head,
                               strides[head]);
                }

                SA_FAIL_IF(anchors.size() < strides.size(), StatusCode::Incompatible,
                           "{} anchors cannot give each of {} heads one", anchors.size(),
                           strides.size());
                return Status();
            }

            /** @brief Checks that @p per_head has one positive count per head, summing to all. */
            Status check_counts(SushiRuntime::span<const std::int64_t> per_head,
                                std::size_t head_count, std::int64_t anchor_count)
            {
                SA_FAIL_IF(per_head.size() != head_count, StatusCode::Incompatible,
                           "per_head has {} counts for {} heads", per_head.size(), head_count);

                for (std::size_t head = 0; head < per_head.size(); ++head)
                    SA_FAIL_IF(per_head[head] <= 0, StatusCode::InvalidDescription,
                               "head {} must take at least one anchor; per_head gives it {}", head,
                               per_head[head]);

                std::int64_t total = 0;
                for (std::size_t head = 0; head < per_head.size(); ++head)
                {
                    // total stays at or below anchor_count, so the sum cannot overflow.
                    SA_FAIL_IF(per_head[head] > anchor_count - total, StatusCode::Incompatible,
                               "per_head asks for more than the {} anchors by head {}",
                               anchor_count, head);
                    total += per_head[head];
                }

                SA_FAIL_IF(total != anchor_count, StatusCode::Incompatible,
                           "per_head sums to {}; there are {} anchors", total, anchor_count);
                return Status();
            }

            /** @brief Returns the anchor indices in ascending area, the lower index on a tie. */
            std::vector<std::int64_t> order_by_area(SushiRuntime::span<const AnchorSize> anchors)
            {
                std::vector<std::int64_t> order(anchors.size());
                std::iota(order.begin(), order.end(), std::int64_t{0});
                std::stable_sort(order.begin(), order.end(),
                                 [&anchors](std::int64_t left, std::int64_t right)
                                 {
                                     return measure_area(anchors[static_cast<std::size_t>(left)]) <
                                            measure_area(anchors[static_cast<std::size_t>(right)]);
                                 });
                return order;
            }

            /** @brief Returns the head indices in ascending stride, the lower index on a tie. */
            std::vector<std::size_t> order_by_stride(SushiRuntime::span<const std::int64_t> strides)
            {
                std::vector<std::size_t> order(strides.size());
                std::iota(order.begin(), order.end(), std::size_t{0});
                std::stable_sort(order.begin(), order.end(),
                                 [&strides](std::size_t left, std::size_t right)
                                 {
                                     return strides[left] < strides[right];
                                 });
                return order;
            }

            /**
             * @brief Returns the even split of @p anchor_count over the heads, per head index.
             * @param heads_by_stride Head indices in ascending stride; the first heads of it
             *        take the remainder, one anchor each.
             */
            std::vector<std::int64_t> split_evenly(std::int64_t anchor_count,
                                                   const std::vector<std::size_t>& heads_by_stride)
            {
                const std::int64_t head_count = static_cast<std::int64_t>(heads_by_stride.size());
                const std::int64_t quotient = anchor_count / head_count;
                const std::int64_t remainder = anchor_count % head_count;

                std::vector<std::int64_t> counts(heads_by_stride.size(), quotient);
                for (std::int64_t rank = 0; rank < remainder; ++rank)
                    ++counts[heads_by_stride[static_cast<std::size_t>(rank)]];
                return counts;
            }
        } // namespace

        Result<AnchorLayout>
        assign_anchors_to_heads(SushiRuntime::span<const AnchorSize> anchors,
                                SushiRuntime::span<const std::int64_t> strides,
                                SushiRuntime::span<const std::int64_t> per_head) noexcept
        {
            SA_TRY(check_anchors_and_strides(anchors, strides));

            const std::int64_t anchor_count = static_cast<std::int64_t>(anchors.size());
            const std::vector<std::size_t> heads_by_stride = order_by_stride(strides);

            std::vector<std::int64_t> counts;
            if (per_head.empty())
            {
                counts = split_evenly(anchor_count, heads_by_stride);
            }
            else
            {
                SA_TRY(check_counts(per_head, strides.size(), anchor_count));
                counts.assign(per_head.begin(), per_head.end());
            }

            const std::vector<std::int64_t> anchors_by_area = order_by_area(anchors);

            AnchorLayout layout;
            layout.anchors.assign(anchors.begin(), anchors.end());
            layout.head_anchors.resize(strides.size());

            // counts sums to the anchor count, so the runs below cover anchors_by_area once.
            auto next = anchors_by_area.begin();
            for (const std::size_t head : heads_by_stride)
            {
                const auto end = next + static_cast<std::ptrdiff_t>(counts[head]);
                layout.head_anchors[head].assign(next, end);
                next = end;
            }
            return layout;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
