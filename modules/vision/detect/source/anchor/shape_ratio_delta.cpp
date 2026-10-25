/**************************************************************************/
/* shape_ratio_delta.cpp                                                  */
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
 * @file shape_ratio_delta.cpp
 * @brief Implements the loss and delta of one anchor head under the shape-ratio rule.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/shape_ratio_delta.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/iou.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>

#include "slot_decode.hpp"

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Holds the level count the shorter balance list answers for. */
            constexpr std::int64_t K_SHORT_BALANCE_LEVELS = 3;

            /** @brief Holds the balance of each level when there are three. */
            constexpr double K_SHORT_BALANCE[] = {4.0, 1.0, 0.4};

            /** @brief Holds the balance of each level for any other level count. */
            constexpr double K_LONG_BALANCE[] = {4.0, 1.0, 0.25, 0.06, 0.02};

            /** @brief Holds the level count the base gains are stated for. */
            constexpr double K_REFERENCE_LEVELS = 3.0;

            /** @brief Holds the class count the base class gain is stated for. */
            constexpr double K_REFERENCE_CLASSES = 80.0;

            /** @brief Holds the input side in pixels the base objectness gain is stated for. */
            constexpr double K_REFERENCE_SIDE = 640.0;

            /** @brief Holds the extents and the anchors one level's loops read. */
            struct LevelPlan
            {
                std::size_t batch = 0;   /**< Images B. */
                std::size_t height = 0;  /**< Grid rows H. */
                std::size_t width = 0;   /**< Grid columns W. */
                std::size_t slots = 0;   /**< Anchor slots A. */
                std::size_t classes = 0; /**< Class entries C. */
                std::size_t entries = 0; /**< Entries of a slot, 5 + C. */
                float scale = 1.0f;      /**< Stretch of an offset. */
                float shift = 0.0f;      /**< `(scale - 1) / 2`. */
                std::vector<AnchorSize> anchors; /**< Each slot's anchor in grid units. */
            };

            /** @brief Holds a decoded slot and the logistic of its four box entries. */
            struct DecodedBox
            {
                std::array<float, K_BOX_VALUES> box{};     /**< Centre and size, grid units. */
                std::array<float, K_BOX_VALUES> sigmoid{}; /**< Logistic of each box entry. */
                std::array<float, K_BOX_VALUES> slope{};   /**< Each box value by its logit. */
            };

            /** @brief Returns the logistic of @p value. */
            double logistic(double value) noexcept
            {
                return 1.0 / (1.0 + std::exp(-value));
            }

            /** @brief Returns the cross-entropy of the logit @p logit against @p target. */
            double cross_entropy(double logit, double target) noexcept
            {
                return std::max(logit, 0.0) - logit * target +
                       std::log1p(std::exp(-std::abs(logit)));
            }

            /** @brief Returns the plan of the level @p input describes. */
            LevelPlan make_plan(const ShapeRatioDeltaInput& input)
            {
                LevelPlan plan;
                plan.batch = static_cast<std::size_t>(input.shape.batch);
                plan.height = static_cast<std::size_t>(input.shape.height);
                plan.width = static_cast<std::size_t>(input.shape.width);
                plan.classes = static_cast<std::size_t>(input.options.num_classes);
                plan.entries = static_cast<std::size_t>(K_FIRST_CLASS_ENTRY) + plan.classes;
                plan.scale = input.options.scale_xy;
                plan.shift = (input.options.scale_xy - 1.0f) / 2.0f;

                const std::vector<std::int64_t>& mask =
                    input.layout->head_anchors[static_cast<std::size_t>(input.head_index)];
                plan.slots = mask.size();
                for (const std::int64_t anchor : mask)
                {
                    const AnchorSize size = input.layout->anchors[static_cast<std::size_t>(anchor)];
                    plan.anchors.push_back({size.width * static_cast<float>(plan.width),
                                            size.height * static_cast<float>(plan.height)});
                }
                return plan;
            }

            /** @brief Returns the index of a record's slot in `[B, H, W, A]`. */
            std::size_t slot_index(const LevelPlan& plan, const TargetAssignment& record) noexcept
            {
                const std::size_t image = static_cast<std::size_t>(record.image);
                const std::size_t row = static_cast<std::size_t>(record.row);
                const std::size_t col = static_cast<std::size_t>(record.col);
                return ((image * plan.height + row) * plan.width + col) * plan.slots +
                       static_cast<std::size_t>(record.slot);
            }

            /**
             * @brief Decodes a slot: offsets `sigmoid * scale - shift`, sides
             *        `(2 * sigmoid)^2 * anchor`.
             */
            DecodedBox decode(const float* slot, const LevelPlan& plan,
                              AnchorSize anchor) noexcept
            {
                DecodedBox decoded;
                for (std::size_t entry = 0; entry < K_BOX_VALUES; ++entry)
                    decoded.sigmoid[entry] = Detail::LogitRule::activate(slot[entry]);

                const float sx = decoded.sigmoid[K_X_ENTRY];
                const float sy = decoded.sigmoid[K_Y_ENTRY];
                const float sw = decoded.sigmoid[K_WIDTH_ENTRY];
                const float sh = decoded.sigmoid[K_HEIGHT_ENTRY];

                decoded.box[K_X_ENTRY] = sx * plan.scale - plan.shift;
                decoded.box[K_Y_ENTRY] = sy * plan.scale - plan.shift;
                decoded.box[K_WIDTH_ENTRY] = Detail::ActivatedRule::size(sw, anchor.width);
                decoded.box[K_HEIGHT_ENTRY] = Detail::ActivatedRule::size(sh, anchor.height);

                decoded.slope[K_X_ENTRY] = plan.scale * sx * (1.0f - sx);
                decoded.slope[K_Y_ENTRY] = plan.scale * sy * (1.0f - sy);
                decoded.slope[K_WIDTH_ENTRY] = 8.0f * sw * anchor.width * sw * (1.0f - sw);
                decoded.slope[K_HEIGHT_ENTRY] = 8.0f * sh * anchor.height * sh * (1.0f - sh);
                return decoded;
            }

            /** @brief Returns a record's truth box, centre and size in grid units. */
            std::array<float, K_BOX_VALUES> truth_of(const TargetAssignment& record) noexcept
            {
                return {record.box_x, record.box_y, record.box_width, record.box_height};
            }

            /** @brief Returns the corners of a centre-and-size box. */
            std::array<float, K_BOX_VALUES> corners_of(
                const std::array<float, K_BOX_VALUES>& box) noexcept
            {
                return {box[0] - box[2] / 2.0f, box[1] - box[3] / 2.0f, box[0] + box[2] / 2.0f,
                        box[1] + box[3] / 2.0f};
            }

            /** @brief Returns the complete IoU of two centre-and-size boxes. */
            float measure_complete_iou(const std::array<float, K_BOX_VALUES>& box,
                                       const std::array<float, K_BOX_VALUES>& truth) noexcept
            {
                const std::array<float, K_BOX_VALUES> a = corners_of(box);
                const std::array<float, K_BOX_VALUES> b = corners_of(truth);
                return complete_iou(SushiRuntime::span<const float>(a.data(), a.size()),
                                    SushiRuntime::span<const float>(b.data(), b.size()));
            }

            /** @brief Returns the IoU of two centre-and-size boxes. */
            float measure_iou(const std::array<float, K_BOX_VALUES>& box,
                              const std::array<float, K_BOX_VALUES>& truth) noexcept
            {
                const std::array<float, K_BOX_VALUES> a = corners_of(box);
                const std::array<float, K_BOX_VALUES> b = corners_of(truth);
                return intersection_over_union(
                    SushiRuntime::span<const float>(a.data(), a.size()),
                    SushiRuntime::span<const float>(b.data(), b.size()));
            }

            /** @brief Returns the first entry of a record's slot in the head. */
            std::size_t slot_offset(const LevelPlan& plan, const TargetAssignment& record) noexcept
            {
                return slot_index(plan, record) * plan.entries;
            }

            /** @brief Returns the size `[B, H, W, A]` of the level's objectness targets. */
            std::size_t count_slots(const LevelPlan& plan) noexcept
            {
                return plan.batch * plan.height * plan.width * plan.slots;
            }

            /** @brief Writes the targets; the caller has validated @p input. */
            void fill_targets(const ShapeRatioDeltaInput& input, const LevelPlan& plan,
                              std::vector<float>& targets)
            {
                targets.assign(count_slots(plan), 0.0f);
                for (const TargetAssignment& record : input.assignments)
                {
                    const DecodedBox decoded =
                        decode(input.head.data() + slot_offset(plan, record), plan,
                               plan.anchors[static_cast<std::size_t>(record.slot)]);
                    const float iou = measure_complete_iou(decoded.box, truth_of(record));
                    targets[slot_index(plan, record)] = std::max(iou, 0.0f);
                }
            }
        } // namespace

        double objectness_balance(std::int64_t levels, std::int64_t level)
        {
            SA_FATAL_IF(level < 0 || level >= levels || levels > K_MAX_BALANCED_LEVELS,
                        "no objectness balance for level {} of {}", level, levels);

            const std::size_t at = static_cast<std::size_t>(level);
            return levels == K_SHORT_BALANCE_LEVELS ? K_SHORT_BALANCE[at] : K_LONG_BALANCE[at];
        }

        ShapeRatioFactors derive_shape_ratio_factors(const ShapeRatioDeltaInput& input)
        {
            const std::int64_t levels =
                static_cast<std::int64_t>(input.layout->head_anchors.size());
            const double level_share = K_REFERENCE_LEVELS / static_cast<double>(levels);
            const double area = static_cast<double>(input.network.width) *
                                static_cast<double>(input.network.height) /
                                (K_REFERENCE_SIDE * K_REFERENCE_SIDE);

            ShapeRatioFactors factors;
            factors.box = static_cast<double>(input.training->box_gain) * level_share;
            factors.classes = static_cast<double>(input.training->class_gain) *
                              static_cast<double>(input.options.num_classes) /
                              K_REFERENCE_CLASSES * level_share;
            factors.objectness = static_cast<double>(input.training->objectness_gain) * area *
                                 level_share * objectness_balance(levels, input.head_index);
            return factors;
        }

        Status validate_shape_ratio_delta(const ShapeRatioDeltaInput& input)
        {
            SA_FAIL_IF(input.layout == nullptr || input.training == nullptr,
                       StatusCode::InvalidDescription,
                       "the shape-ratio delta needs a layout and training options");
            SA_TRY(validate_head(input.options, *input.layout, input.head_index));
            SA_TRY(validate_head_training(*input.training));
            SA_FAIL_IF(input.options.coding != HeadCoding::Logit, StatusCode::InvalidDescription,
                       "the shape-ratio delta reads a head of logits; head {} has another coding",
                       input.head_index);
            SA_FAIL_IF(input.shape.batch < 1 || input.shape.height < 1 || input.shape.width < 1,
                       StatusCode::InvalidDescription,
                       "the shape-ratio delta needs a head of at least one image and one cell; "
                       "got {} by {} by {}",
                       input.shape.batch, input.shape.height, input.shape.width);
            SA_FAIL_IF(input.network.height < 1 || input.network.width < 1,
                       StatusCode::InvalidDescription,
                       "the shape-ratio delta needs a positive input size; got {} by {}",
                       input.network.height, input.network.width);

            const std::int64_t levels =
                static_cast<std::int64_t>(input.layout->head_anchors.size());
            SA_FAIL_IF(levels > K_MAX_BALANCED_LEVELS, StatusCode::InvalidDescription,
                       "the objectness balance has weights for at most {} levels; the layout "
                       "has {}",
                       K_MAX_BALANCED_LEVELS, levels);

            const LevelPlan plan = make_plan(input);
            const std::size_t values = count_slots(plan) * plan.entries;
            SA_FAIL_IF(input.head.size() != values, StatusCode::InvalidDescription,
                       "head {} holds {} values; its shape needs {}", input.head_index,
                       input.head.size(), values);

            for (const TargetAssignment& record : input.assignments)
            {
                SA_FAIL_IF(record.image < 0 || record.image >= input.shape.batch ||
                               record.slot < 0 ||
                               record.slot >= static_cast<std::int64_t>(plan.slots) ||
                               record.row < 0 || record.row >= input.shape.height ||
                               record.col < 0 || record.col >= input.shape.width ||
                               record.label < 0 || record.label >= input.options.num_classes,
                           StatusCode::InvalidDescription,
                           "an assignment to image {}, slot {}, cell ({}, {}), class {} lies "
                           "outside head {}",
                           record.image, record.slot, record.row, record.col, record.label,
                           input.head_index);
            }
            return Status();
        }

        Status build_objectness_targets(const ShapeRatioDeltaInput& input,
                                        std::vector<float>& targets)
        {
            SA_TRY(validate_shape_ratio_delta(input));
            fill_targets(input, make_plan(input), targets);
            return Status();
        }

        Result<ShapeRatioLoss> measure_shape_ratio_loss(const ShapeRatioDeltaInput& input,
                                                        SushiRuntime::span<const float> targets)
        {
            SA_TRY(validate_shape_ratio_delta(input));
            const LevelPlan plan = make_plan(input);
            SA_FAIL_IF(targets.size() != count_slots(plan), StatusCode::InvalidDescription,
                       "head {} has {} slots; got {} objectness targets", input.head_index,
                       count_slots(plan), targets.size());

            ShapeRatioLoss loss;
            const std::size_t slots = count_slots(plan);
            for (std::size_t slot = 0; slot < slots; ++slot)
            {
                const float logit = input.head[slot * plan.entries + K_OBJECTNESS_ENTRY];
                loss.objectness += cross_entropy(logit, targets[slot]);
            }
            loss.objectness /= static_cast<double>(slots);

            const std::size_t count = input.assignments.size();
            for (const TargetAssignment& record : input.assignments)
            {
                const float* slot = input.head.data() + slot_offset(plan, record);
                const DecodedBox decoded =
                    decode(slot, plan, plan.anchors[static_cast<std::size_t>(record.slot)]);
                loss.box += 1.0 - measure_complete_iou(decoded.box, truth_of(record));

                if (plan.classes < 2)
                    continue;
                for (std::size_t index = 0; index < plan.classes; ++index)
                {
                    const double wanted =
                        index == static_cast<std::size_t>(record.label) ? 1.0 : 0.0;
                    loss.classes += cross_entropy(slot[K_FIRST_CLASS_ENTRY + index], wanted);
                }
            }
            if (count > 0)
            {
                loss.box /= static_cast<double>(count);
                loss.classes /= static_cast<double>(count * plan.classes);
            }

            const ShapeRatioFactors factors = derive_shape_ratio_factors(input);
            loss.total = (loss.box * factors.box + loss.objectness * factors.objectness +
                          loss.classes * factors.classes) *
                         static_cast<double>(plan.batch);
            return loss;
        }

        Status compute_shape_ratio_delta(const ShapeRatioDeltaInput& input, HeadDelta& result)
        {
            std::vector<float> targets;
            SA_TRY(build_objectness_targets(input, targets));

            const LevelPlan plan = make_plan(input);
            const ShapeRatioFactors factors = derive_shape_ratio_factors(input);
            const double batch = static_cast<double>(plan.batch);
            const std::size_t slots = count_slots(plan);
            const std::size_t count = input.assignments.size();

            HeadDelta written;
            written.delta.assign(slots * plan.entries, 0.0f);
            written.assigned_row.assign(slots, -1);
            written.assigned_class.assign(slots, -1);

            // The delta is -B times d(loss); the loss already carries a factor of B.
            const double object_scale =
                -batch * batch * factors.objectness / static_cast<double>(slots);
            for (std::size_t slot = 0; slot < slots; ++slot)
            {
                const std::size_t at = slot * plan.entries + K_OBJECTNESS_ENTRY;
                const double gap = logistic(input.head[at]) - static_cast<double>(targets[slot]);
                written.delta[at] = static_cast<float>(object_scale * gap);
            }

            const double count_value = static_cast<double>(count);
            const double box_scale = count > 0 ? batch * batch * factors.box / count_value : 0.0;
            const double class_scale =
                count > 0 && plan.classes > 1
                    ? -batch * batch * factors.classes /
                          (count_value * static_cast<double>(plan.classes))
                    : 0.0;

            for (const TargetAssignment& record : input.assignments)
            {
                const std::size_t offset = slot_offset(plan, record);
                const float* slot = input.head.data() + offset;
                float* delta = written.delta.data() + offset;
                const std::array<float, K_BOX_VALUES> truth = truth_of(record);
                const DecodedBox decoded =
                    decode(slot, plan, plan.anchors[static_cast<std::size_t>(record.slot)]);

                // d(1 - CIoU) is -d(CIoU), so the delta of the box is +B^2 * gain / n * dCIoU.
                std::array<float, K_BOX_VALUES> gradient{};
                complete_iou_gradient(
                    SushiRuntime::span<const float>(decoded.box.data(), decoded.box.size()),
                    SushiRuntime::span<const float>(truth.data(), truth.size()),
                    BoxGradient::Analytic,
                    SushiRuntime::span<float>(gradient.data(), gradient.size()));
                for (std::size_t entry = 0; entry < K_BOX_VALUES; ++entry)
                {
                    delta[entry] += static_cast<float>(box_scale * gradient[entry] *
                                                       decoded.slope[entry]);
                }

                for (std::size_t index = 0; index < plan.classes && class_scale != 0.0; ++index)
                {
                    const double wanted =
                        index == static_cast<std::size_t>(record.label) ? 1.0 : 0.0;
                    const double gap = logistic(slot[K_FIRST_CLASS_ENTRY + index]) - wanted;
                    delta[K_FIRST_CLASS_ENTRY + index] += static_cast<float>(class_scale * gap);
                }

                const std::size_t index = slot_index(plan, record);
                written.assigned_row[index] = record.truth_row;
                written.assigned_class[index] = record.label;
                written.iou_sum += static_cast<double>(measure_iou(decoded.box, truth));
            }
            written.assignments = static_cast<std::int64_t>(count);

            result = std::move(written);
            return Status();
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
