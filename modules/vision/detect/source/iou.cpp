/**************************************************************************/
/* iou.cpp                                                                */
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
 * @file iou.cpp
 * @brief Implements intersection over union and complete IoU in float32.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/iou.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <SushiAI/core/common.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Holds 4 / pi squared, the factor of the aspect-ratio term, as float32. */
            constexpr float K_ASPECT_FACTOR =
                static_cast<float>(4.0 / (3.141592653589793 * 3.141592653589793));

            /** @brief Holds the sides, intersection and union that IoU and complete IoU share. */
            struct Overlap
            {
                float width1 = 0.0f;       /**< Width of the first box. */
                float height1 = 0.0f;      /**< Height of the first box plus K_IOU_EPS. */
                float width2 = 0.0f;       /**< Width of the second box. */
                float height2 = 0.0f;      /**< Height of the second box plus K_IOU_EPS. */
                float intersection = 0.0f; /**< Area both boxes cover. */
                float joined = 0.0f;       /**< Area either box covers plus K_IOU_EPS. */
            };

            /** @brief Returns @p value, or zero when it is negative. */
            float clamp_at_zero(float value) noexcept
            {
                return value < 0.0f ? 0.0f : value;
            }

            /** @brief Computes the shared terms of two boxes in the reference's operation order. */
            Overlap measure_overlap(const float* box1, const float* box2) noexcept
            {
                Overlap overlap;
                overlap.width1 = box1[2] - box1[0];
                overlap.height1 = box1[3] - box1[1] + K_IOU_EPS;
                overlap.width2 = box2[2] - box2[0];
                overlap.height2 = box2[3] - box2[1] + K_IOU_EPS;

                const float across =
                    clamp_at_zero(std::min(box1[2], box2[2]) - std::max(box1[0], box2[0]));
                const float down =
                    clamp_at_zero(std::min(box1[3], box2[3]) - std::max(box1[1], box2[1]));
                overlap.intersection = across * down;

                const float area1 = overlap.width1 * overlap.height1;
                const float area2 = overlap.width2 * overlap.height2;
                overlap.joined = area1 + area2 - overlap.intersection + K_IOU_EPS;
                return overlap;
            }

            /** @brief Throws unless two `[N, 4]` inputs and an `[N]` output were given. */
            void check_elementwise(std::size_t size1, std::size_t size2, std::size_t output_size)
            {
                SA_THROW_IF(size1 % K_BOX_VALUES != 0 || size2 != size1 ||
                                output_size != size1 / K_BOX_VALUES,
                            "an elementwise IoU needs two [N, 4] inputs and an [N] output; got "
                            "{}, {} and {} values",
                            size1, size2, output_size);
            }
        } // namespace

        float intersection_over_union(SushiRuntime::span<const float> box1,
                                      SushiRuntime::span<const float> box2) noexcept
        {
            const Overlap overlap = measure_overlap(box1.data(), box2.data());
            return overlap.intersection / overlap.joined;
        }

        float complete_iou(SushiRuntime::span<const float> box1,
                           SushiRuntime::span<const float> box2) noexcept
        {
            const float* a = box1.data();
            const float* b = box2.data();
            const Overlap overlap = measure_overlap(a, b);
            const float iou = overlap.intersection / overlap.joined;

            const float hull_width = std::max(a[2], b[2]) - std::min(a[0], b[0]);
            const float hull_height = std::max(a[3], b[3]) - std::min(a[1], b[1]);
            const float diagonal = hull_width * hull_width + hull_height * hull_height + K_IOU_EPS;

            const float shift_x = b[0] + b[2] - a[0] - a[2];
            const float shift_y = b[1] + b[3] - a[1] - a[3];
            const float centre_distance = (shift_x * shift_x + shift_y * shift_y) / 4.0f;

            const float angle_gap = std::atan(overlap.width2 / overlap.height2) -
                                    std::atan(overlap.width1 / overlap.height1);
            const float aspect = K_ASPECT_FACTOR * (angle_gap * angle_gap);
            const float trade_off = aspect / (1.0f - iou + aspect + K_IOU_EPS);
            return iou - (centre_distance / diagonal + aspect * trade_off);
        }

        void elementwise_iou(SushiRuntime::span<const float> boxes1,
                             SushiRuntime::span<const float> boxes2,
                             SushiRuntime::span<float> ious)
        {
            check_elementwise(boxes1.size(), boxes2.size(), ious.size());
            for (std::size_t row = 0; row < ious.size(); ++row)
            {
                const std::size_t at = row * K_BOX_VALUES;
                ious[row] = intersection_over_union(boxes1.subspan(at, K_BOX_VALUES),
                                                    boxes2.subspan(at, K_BOX_VALUES));
            }
        }

        void elementwise_complete_iou(SushiRuntime::span<const float> boxes1,
                                      SushiRuntime::span<const float> boxes2,
                                      SushiRuntime::span<float> ious)
        {
            check_elementwise(boxes1.size(), boxes2.size(), ious.size());
            for (std::size_t row = 0; row < ious.size(); ++row)
            {
                const std::size_t at = row * K_BOX_VALUES;
                ious[row] = complete_iou(boxes1.subspan(at, K_BOX_VALUES),
                                         boxes2.subspan(at, K_BOX_VALUES));
            }
        }

        void pairwise_iou(SushiRuntime::span<const float> boxes1,
                          SushiRuntime::span<const float> boxes2,
                          SushiRuntime::span<float> ious)
        {
            const std::size_t count1 = boxes1.size() / K_BOX_VALUES;
            const std::size_t count2 = boxes2.size() / K_BOX_VALUES;
            SA_THROW_IF(boxes1.size() % K_BOX_VALUES != 0 || boxes2.size() % K_BOX_VALUES != 0 ||
                            ious.size() != count1 * count2,
                        "a pairwise IoU needs [N, 4] and [M, 4] inputs and an [N, M] output; got "
                        "{}, {} and {} values",
                        boxes1.size(), boxes2.size(), ious.size());

            for (std::size_t row = 0; row < count1; ++row)
            {
                const float* a = boxes1.data() + row * K_BOX_VALUES;
                const float area1 = (a[2] - a[0]) * (a[3] - a[1]);
                for (std::size_t column = 0; column < count2; ++column)
                {
                    const float* b = boxes2.data() + column * K_BOX_VALUES;
                    const float across = clamp_at_zero(std::min(a[2], b[2]) - std::max(a[0], b[0]));
                    const float down = clamp_at_zero(std::min(a[3], b[3]) - std::max(a[1], b[1]));
                    const float intersection = across * down;
                    const float area2 = (b[2] - b[0]) * (b[3] - b[1]);
                    ious[row * count2 + column] =
                        intersection / (area1 + area2 - intersection + K_IOU_EPS);
                }
            }
        }

    } // namespace Vision::Detect
} // namespace SushiAI
