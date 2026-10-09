/**************************************************************************/
/* assigner.cpp                                                           */
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
 * @file assigner.cpp
 * @brief Implements TaskAlignedAssigner, the host label assignment of a detector.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/assigner.hpp>

#include <algorithm>
#include <cmath>

#include <SushiAI/core/common.hpp>
#include <SushiAI/vision/detect/anchors.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/iou.hpp>
#include <SushiAI/vision/detect/top_k.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Holds the margin by which a candidate anchor lies inside its box. */
            constexpr float K_INSIDE_MARGIN = 1e-9f;

            /** @brief Marks an anchor outside the mask. */
            constexpr std::uint8_t K_CLEAR = 0;

            /** @brief Marks an anchor inside the mask. */
            constexpr std::uint8_t K_SET = 1;

            /** @brief Marks an anchor of the mask that the running top-k also selected. */
            constexpr std::uint8_t K_KEPT = 2;

            /** @brief Returns @p base to the power @p exponent; 0.5 is a square root. */
            float raise(float base, float exponent) noexcept
            {
                if (exponent == 0.5f)
                    return std::sqrt(base);
                if (exponent == 1.0f)
                    return base;
                if (exponent == 2.0f)
                    return base * base;
                return std::pow(base, exponent);
            }

            /** @brief Writes @p truth with each side raised to @p min_side about its centre. */
            void enlarge_box(SushiRuntime::span<const float> truth,
                             float min_side,
                             SushiRuntime::span<float> enlarged)
            {
                xyxy_to_xywh(truth, enlarged);
                enlarged[2] = std::max(enlarged[2], min_side);
                enlarged[3] = std::max(enlarged[3], min_side);
                xywh_to_xyxy(enlarged, enlarged);
            }

            /** @brief Clears every K_SET mark of @p mask that is not in @p selected. */
            void keep_selected(SushiRuntime::span<std::uint8_t> mask,
                               const std::vector<std::int64_t>& selected) noexcept
            {
                for (const std::int64_t index : selected)
                {
                    std::uint8_t& mark = mask[static_cast<std::size_t>(index)];
                    if (mark == K_SET)
                        mark = K_KEPT;
                }
                for (std::uint8_t& mark : mask)
                    mark = mark == K_KEPT ? K_SET : K_CLEAR;
            }
        } // namespace

        TaskAlignedAssigner::TaskAlignedAssigner(const AssignerConfig& config) : config_(config)
        {
            SA_THROW_IF(config.topk <= 0 || config.topk2 <= 0 || config.num_classes <= 0,
                        "an assigner needs positive topk, topk2 and class count; got {}, {} and {}",
                        config.topk, config.topk2, config.num_classes);
            SA_THROW_IF(!(config.stride_val >= 0.0f) || !(config.eps >= 0.0f),
                        "an assigner needs a stride_val and an eps of at least 0; got {} and {}",
                        config.stride_val, config.eps);
        }

        void TaskAlignedAssigner::assign(const AssignerInputs& inputs, Assignment& assignment)
        {
            check_inputs(inputs);

            const std::size_t rows = static_cast<std::size_t>(inputs.batch * inputs.anchors);
            const std::size_t classes = static_cast<std::size_t>(config_.num_classes);
            const std::int64_t idle_label = inputs.max_boxes == 0 ? config_.num_classes : 0;

            assignment.batch = inputs.batch;
            assignment.anchors = inputs.anchors;
            assignment.num_classes = config_.num_classes;
            assignment.foreground_mask.assign(rows, K_CLEAR);
            assignment.target_boxes.assign(rows * K_BOX_VALUES, 0.0f);
            assignment.target_scores.assign(rows * classes, 0.0f);
            assignment.target_labels.assign(rows, idle_label);
            assignment.target_indices.assign(rows, 0);
            if (inputs.max_boxes == 0)
                return;

            const std::size_t anchors = static_cast<std::size_t>(inputs.anchors);
            const std::size_t max_boxes = static_cast<std::size_t>(inputs.max_boxes);
            metric_.resize(max_boxes * anchors);
            overlap_.resize(max_boxes * anchors);
            mask_.resize(max_boxes * anchors);
            holders_.resize(anchors);
            peak_metric_.resize(max_boxes);
            peak_overlap_.resize(max_boxes);
            row_.resize(anchors);

            for (std::size_t image = 0; image < static_cast<std::size_t>(inputs.batch); ++image)
            {
                select_candidates(inputs, image);
                resolve_conflicts(inputs);
                keep_second_top_k(inputs, image);
                write_targets(inputs, image, assignment);
            }
        }

        void TaskAlignedAssigner::check_inputs(const AssignerInputs& inputs) const
        {
            SA_THROW_IF(inputs.batch <= 0 || inputs.anchors <= 0 || inputs.max_boxes < 0,
                        "an assignment needs a positive batch and anchor count and a box count "
                        "of at least 0; got {}, {} and {}",
                        inputs.batch, inputs.anchors, inputs.max_boxes);

            const std::size_t batch = static_cast<std::size_t>(inputs.batch);
            const std::size_t rows = batch * static_cast<std::size_t>(inputs.anchors);
            const std::size_t classes = static_cast<std::size_t>(config_.num_classes);
            SA_THROW_IF(inputs.scores.size() != rows * classes ||
                            inputs.boxes.size() != rows * K_BOX_VALUES ||
                            inputs.anchor_points.size() != rows / batch * K_POINT_VALUES,
                        "predictions must be [{}, {}, {}] scores, [{}, {}, 4] boxes and [{}, 2] "
                        "anchor points; got {}, {} and {} values",
                        inputs.batch, inputs.anchors, config_.num_classes, inputs.batch,
                        inputs.anchors, inputs.anchors, inputs.scores.size(), inputs.boxes.size(),
                        inputs.anchor_points.size());

            const std::size_t slots = static_cast<std::size_t>(inputs.batch * inputs.max_boxes);
            SA_THROW_IF(inputs.gt_labels.size() != slots ||
                            inputs.gt_boxes.size() != slots * K_BOX_VALUES ||
                            inputs.gt_valid.size() != slots,
                        "ground truth must be [{}, {}] labels, [{}, {}, 4] boxes and [{}, {}] "
                        "flags; got {}, {} and {} values",
                        inputs.batch, inputs.max_boxes, inputs.batch, inputs.max_boxes,
                        inputs.batch, inputs.max_boxes, inputs.gt_labels.size(),
                        inputs.gt_boxes.size(), inputs.gt_valid.size());

            SA_THROW_IF(inputs.max_boxes > 0 &&
                            (inputs.anchors < config_.topk || inputs.anchors < config_.topk2),
                        "{} anchors are fewer than topk {} or topk2 {}", inputs.anchors,
                        config_.topk, config_.topk2);

            for (std::size_t slot = 0; slot < slots; ++slot)
            {
                if (inputs.gt_valid[slot] == 0)
                    continue;

                const std::int64_t label = inputs.gt_labels[slot];
                SA_THROW_IF(label < 0 || label >= config_.num_classes,
                            "ground-truth box {} of image {} has class {} of {}",
                            slot % static_cast<std::size_t>(inputs.max_boxes),
                            slot / static_cast<std::size_t>(inputs.max_boxes), label,
                            config_.num_classes);
            }
        }

        void TaskAlignedAssigner::select_candidates(const AssignerInputs& inputs, std::size_t image)
        {
            const std::size_t anchors = static_cast<std::size_t>(inputs.anchors);
            const std::size_t max_boxes = static_cast<std::size_t>(inputs.max_boxes);
            const std::size_t classes = static_cast<std::size_t>(config_.num_classes);
            const float* scores = inputs.scores.data() + image * anchors * classes;
            const float* boxes = inputs.boxes.data() + image * anchors * K_BOX_VALUES;
            const float* points = inputs.anchor_points.data();

            std::fill(metric_.begin(), metric_.end(), 0.0f);
            std::fill(overlap_.begin(), overlap_.end(), 0.0f);
            std::fill(mask_.begin(), mask_.end(), K_CLEAR);

            for (std::size_t box = 0; box < max_boxes; ++box)
            {
                const std::size_t slot = image * max_boxes + box;
                if (inputs.gt_valid[slot] == 0)
                    continue;

                const std::size_t label = static_cast<std::size_t>(inputs.gt_labels[slot]);
                const SushiRuntime::span<const float> truth_box =
                    inputs.gt_boxes.subspan(slot * K_BOX_VALUES, K_BOX_VALUES);
                float pool[K_BOX_VALUES];
                enlarge_box(truth_box, config_.stride_val, pool);

                float* metric = metric_.data() + box * anchors;
                float* overlap = overlap_.data() + box * anchors;
                std::uint8_t* mask = mask_.data() + box * anchors;

                for (std::size_t anchor = 0; anchor < anchors; ++anchor)
                {
                    const float x = points[anchor * K_POINT_VALUES];
                    const float y = points[anchor * K_POINT_VALUES + 1];
                    const bool inside =
                        x - pool[0] > K_INSIDE_MARGIN && y - pool[1] > K_INSIDE_MARGIN &&
                        pool[2] - x > K_INSIDE_MARGIN && pool[3] - y > K_INSIDE_MARGIN;
                    if (!inside)
                        continue;

                    const SushiRuntime::span<const float> predicted(
                        boxes + anchor * K_BOX_VALUES, K_BOX_VALUES);
                    const float value = complete_iou(truth_box, predicted);
                    mask[anchor] = K_SET;
                    overlap[anchor] = value < 0.0f ? 0.0f : value;
                    metric[anchor] = raise(scores[anchor * classes + label], config_.alpha) *
                                     raise(overlap[anchor], config_.beta);
                }

                select_top_k(SushiRuntime::span<const float>(metric, anchors), config_.topk,
                             selected_);
                keep_selected(SushiRuntime::span<std::uint8_t>(mask, anchors), selected_);
            }
        }

        void TaskAlignedAssigner::resolve_conflicts(const AssignerInputs& inputs)
        {
            const std::size_t anchors = static_cast<std::size_t>(inputs.anchors);
            const std::size_t max_boxes = static_cast<std::size_t>(inputs.max_boxes);

            std::fill(holders_.begin(), holders_.end(), 0);
            for (std::size_t box = 0; box < max_boxes; ++box)
            {
                const std::uint8_t* mask = mask_.data() + box * anchors;
                for (std::size_t anchor = 0; anchor < anchors; ++anchor)
                    holders_[anchor] += mask[anchor];
            }

            for (std::size_t anchor = 0; anchor < anchors; ++anchor)
            {
                if (holders_[anchor] <= 1)
                    continue;

                std::size_t winner = 0;
                for (std::size_t box = 1; box < max_boxes; ++box)
                {
                    if (overlap_[box * anchors + anchor] > overlap_[winner * anchors + anchor])
                        winner = box;
                }
                for (std::size_t box = 0; box < max_boxes; ++box)
                    mask_[box * anchors + anchor] = box == winner ? K_SET : K_CLEAR;
            }
        }

        void TaskAlignedAssigner::keep_second_top_k(const AssignerInputs& inputs, std::size_t image)
        {
            if (config_.topk2 == config_.topk)
                return;

            const std::size_t anchors = static_cast<std::size_t>(inputs.anchors);
            const std::size_t max_boxes = static_cast<std::size_t>(inputs.max_boxes);

            for (std::size_t box = 0; box < max_boxes; ++box)
            {
                if (inputs.gt_valid[image * max_boxes + box] == 0)
                    continue;

                const float* metric = metric_.data() + box * anchors;
                std::uint8_t* mask = mask_.data() + box * anchors;
                for (std::size_t anchor = 0; anchor < anchors; ++anchor)
                    row_[anchor] = mask[anchor] == K_SET ? metric[anchor] : 0.0f;

                select_top_k(SushiRuntime::span<const float>(row_.data(), anchors), config_.topk2,
                             selected_);
                keep_selected(SushiRuntime::span<std::uint8_t>(mask, anchors), selected_);
            }
        }

        void TaskAlignedAssigner::write_targets(const AssignerInputs& inputs,
                                                std::size_t image,
                                                Assignment& assignment)
        {
            const std::size_t anchors = static_cast<std::size_t>(inputs.anchors);
            const std::size_t max_boxes = static_cast<std::size_t>(inputs.max_boxes);
            const std::size_t classes = static_cast<std::size_t>(config_.num_classes);

            std::uint8_t* foreground = assignment.foreground_mask.data() + image * anchors;
            std::int64_t* indices = assignment.target_indices.data() + image * anchors;
            std::int64_t* labels = assignment.target_labels.data() + image * anchors;
            float* target_boxes = assignment.target_boxes.data() + image * anchors * K_BOX_VALUES;
            float* target_scores = assignment.target_scores.data() + image * anchors * classes;

            // resolve_conflicts left each anchor in the mask of at most one box.
            for (std::size_t box = 0; box < max_boxes; ++box)
            {
                const std::uint8_t* mask = mask_.data() + box * anchors;
                const float* metric = metric_.data() + box * anchors;
                const float* overlap = overlap_.data() + box * anchors;
                float peak_metric = 0.0f;
                float peak_overlap = 0.0f;
                for (std::size_t anchor = 0; anchor < anchors; ++anchor)
                {
                    if (mask[anchor] != K_SET)
                        continue;

                    foreground[anchor] = K_SET;
                    indices[anchor] = static_cast<std::int64_t>(box);
                    peak_metric = std::max(peak_metric, metric[anchor]);
                    peak_overlap = std::max(peak_overlap, overlap[anchor]);
                }
                peak_metric_[box] = peak_metric;
                peak_overlap_[box] = peak_overlap;
            }

            for (std::size_t anchor = 0; anchor < anchors; ++anchor)
            {
                const std::size_t box = static_cast<std::size_t>(indices[anchor]);
                const std::size_t slot = image * max_boxes + box;
                const std::int64_t label = std::max<std::int64_t>(inputs.gt_labels[slot], 0);
                const float* truth = inputs.gt_boxes.data() + slot * K_BOX_VALUES;

                labels[anchor] = label;
                std::copy(truth, truth + K_BOX_VALUES, target_boxes + anchor * K_BOX_VALUES);
                if (foreground[anchor] != K_SET)
                    continue;

                target_scores[anchor * classes + static_cast<std::size_t>(label)] =
                    metric_[box * anchors + anchor] * peak_overlap_[box] /
                    (peak_metric_[box] + config_.eps);
            }
        }

    } // namespace Vision::Detect
} // namespace SushiAI
