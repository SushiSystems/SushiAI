/**************************************************************************/
/* head_delta.cpp                                                         */
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
 * @file head_delta.cpp
 * @brief Implements the training delta of an anchor head and its truth assignment.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_delta.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include <SushiAI/core/check.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

#include "box_delta.hpp"

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            using Detail::BoxDeltaFunction;
            using Detail::BoxDeltaTerms;
            using Detail::CentreBox;

            /** @brief Holds the value that marks a slot no truth was assigned to. */
            constexpr std::int64_t K_UNASSIGNED = -1;

            /** @brief Holds a box loss and the function that adds its delta. */
            struct BoxDeltaRow
            {
                BoxLoss loss;         /**< The loss the row answers for. */
                BoxDeltaFunction add; /**< Adds one truth's box delta to a slot. */
            };

            /** @brief Holds one row per box loss whose delta is written. */
            constexpr BoxDeltaRow K_BOX_DELTA_ROWS[] = {
                {BoxLoss::SquaredError, &Detail::add_squared_error_box_delta},
            };

            /** @brief Returns the function of @p loss, or null when no row holds it. */
            BoxDeltaFunction find_box_delta(BoxLoss loss) noexcept
            {
                for (const BoxDeltaRow& row : K_BOX_DELTA_ROWS)
                {
                    if (row.loss == loss)
                        return row.add;
                }
                return nullptr;
            }

            /** @brief Holds one valid truth of an image. */
            struct Truth
            {
                CentreBox box;          /**< Centre and size as fractions of the input. */
                std::int64_t label = 0; /**< Class of the truth. */
                std::int64_t row = 0;   /**< Row of the truth in its image's padded list. */
            };

            /** @brief Holds the extents and the settings the loops over one head read. */
            struct DeltaPlan
            {
                std::size_t height = 0;  /**< Grid rows H. */
                std::size_t width = 0;   /**< Grid columns W. */
                std::size_t slots = 0;   /**< Anchor slots A. */
                std::size_t classes = 0; /**< Class entries C. */
                std::size_t entries = 0; /**< Entries of a slot, 5 + C. */
                float scale = 1.0f;      /**< Stretch of an offset. */
                float shift = 0.0f;      /**< `(scale - 1) / 2`, taken off a stretched offset. */
                const std::vector<AnchorSize>* anchors = nullptr;      /**< Every anchor. */
                const std::vector<std::int64_t>* head_slots = nullptr; /**< The head's anchors. */
                const HeadTrainingOptions* training = nullptr;         /**< Weights and gates. */
                BoxDeltaFunction add_box_delta = nullptr;              /**< The box term. */
            };

            /** @brief Holds a slot's decoded box and the activated offsets it came from. */
            struct SlotReading
            {
                CentreBox box;         /**< The decoded box. */
                float offset_x = 0.0f; /**< Activated, stretched offset along the columns. */
                float offset_y = 0.0f; /**< Activated, stretched offset along the rows. */
            };

            /** @brief Returns the logistic of a logit. */
            float logistic(float value) noexcept
            {
                return 1.0f / (1.0f + std::exp(-value));
            }

            /** @brief Returns the IoU of two boxes; 0 when they share no area. */
            float measure_box_iou(const CentreBox& a, const CentreBox& b) noexcept
            {
                const float across = std::min(a.x + a.width / 2.0f, b.x + b.width / 2.0f) -
                                     std::max(a.x - a.width / 2.0f, b.x - b.width / 2.0f);
                const float down = std::min(a.y + a.height / 2.0f, b.y + b.height / 2.0f) -
                                   std::max(a.y - a.height / 2.0f, b.y - b.height / 2.0f);

                // Both comparisons are false for a NaN side, which then reads as no overlap.
                if (!(across > 0.0f) || !(down > 0.0f))
                    return 0.0f;

                const float intersection = across * down;
                const float joined = a.width * a.height + b.width * b.height - intersection;
                if (!(joined > 0.0f))
                    return 0.0f;

                return intersection / joined;
            }

            /** @brief Decodes the slot at @p slot, which lies in cell (@p row, @p col). */
            SlotReading read_slot(const float* slot, const DeltaPlan& plan, std::size_t row,
                                  std::size_t col, AnchorSize anchor) noexcept
            {
                SlotReading reading;
                reading.offset_x = logistic(slot[K_X_ENTRY]) * plan.scale - plan.shift;
                reading.offset_y = logistic(slot[K_Y_ENTRY]) * plan.scale - plan.shift;
                reading.box.x =
                    (static_cast<float>(col) + reading.offset_x) / static_cast<float>(plan.width);
                reading.box.y =
                    (static_cast<float>(row) + reading.offset_y) / static_cast<float>(plan.height);
                reading.box.width = std::exp(slot[K_WIDTH_ENTRY]) * anchor.width;
                reading.box.height = std::exp(slot[K_HEIGHT_ENTRY]) * anchor.height;
                return reading;
            }

            /** @brief Returns the anchor of slot @p slot of the plan's head. */
            AnchorSize anchor_of_slot(const DeltaPlan& plan, std::size_t slot) noexcept
            {
                return (*plan.anchors)[static_cast<std::size_t>((*plan.head_slots)[slot])];
            }

            /** @brief Reports whether a class probability of the slot exceeds the class gate. */
            bool has_gated_class(const float* slot, const DeltaPlan& plan) noexcept
            {
                const float* classes = slot + K_FIRST_CLASS_ENTRY;
                for (std::size_t index = 0; index < plan.classes; ++index)
                {
                    if (logistic(classes[index]) > plan.training->class_gate)
                        return true;
                }
                return false;
            }

            /** @brief Returns the largest IoU of @p box with a truth; 0 without truths. */
            float measure_best_iou(const CentreBox& box, const std::vector<Truth>& truths) noexcept
            {
                float best = 0.0f;
                for (const Truth& truth : truths)
                    best = std::max(best, measure_box_iou(box, truth.box));
                return best;
            }

            /**
             * @brief Writes the objectness delta of every slot of one image, toward zero.
             * @param head The image's head values.
             * @param delta The image's deltas; a slot that overlaps a truth beyond the ignore
             *              threshold and has a gated class keeps a zero objectness delta.
             */
            void write_background_deltas(const float* head, const DeltaPlan& plan,
                                         const std::vector<Truth>& truths, float* delta) noexcept
            {
                const float ignore = plan.training->thresholds.ignore_threshold;

                for (std::size_t row = 0; row < plan.height; ++row)
                {
                    for (std::size_t col = 0; col < plan.width; ++col)
                    {
                        for (std::size_t slot = 0; slot < plan.slots; ++slot)
                        {
                            const SlotReading reading =
                                read_slot(head, plan, row, col, anchor_of_slot(plan, slot));
                            const bool ignored =
                                measure_best_iou(reading.box, truths) > ignore &&
                                has_gated_class(head, plan);

                            delta[K_OBJECTNESS_ENTRY] =
                                ignored ? 0.0f
                                        : plan.training->object_weight *
                                              (0.0f - logistic(head[K_OBJECTNESS_ENTRY]));
                            head += plan.entries;
                            delta += plan.entries;
                        }
                    }
                }
            }

            /** @brief Returns the first anchor whose shape IoU with @p size is the largest. */
            std::int64_t find_best_anchor(const std::vector<AnchorSize>& anchors,
                                          AnchorSize size) noexcept
            {
                std::int64_t best = 0;
                float best_iou = -1.0f;
                for (std::size_t index = 0; index < anchors.size(); ++index)
                {
                    const float iou = shape_iou(anchors[index], size);
                    if (iou > best_iou)
                    {
                        best_iou = iou;
                        best = static_cast<std::int64_t>(index);
                    }
                }
                return best;
            }

            /** @brief Returns the first slot of the head that uses @p anchor, or -1. */
            std::int64_t find_slot(const std::vector<std::int64_t>& head_slots,
                                   std::int64_t anchor) noexcept
            {
                for (std::size_t slot = 0; slot < head_slots.size(); ++slot)
                {
                    if (head_slots[slot] == anchor)
                        return static_cast<std::int64_t>(slot);
                }
                return -1;
            }

            /**
             * @brief Writes a slot's class deltas toward the class @p label.
             * @param delta The slot's deltas; when the entry of @p label is already non-zero
             *              that entry alone is written.
             */
            void write_class_deltas(const float* slot, const DeltaPlan& plan, std::int64_t label,
                                    float* delta) noexcept
            {
                const float* classes = slot + K_FIRST_CLASS_ENTRY;
                float* class_delta = delta + K_FIRST_CLASS_ENTRY;
                const std::size_t target = static_cast<std::size_t>(label);

                if (class_delta[target] != 0.0f)
                {
                    class_delta[target] = 1.0f - logistic(classes[target]);
                    return;
                }

                for (std::size_t index = 0; index < plan.classes; ++index)
                {
                    const float wanted = index == target ? 1.0f : 0.0f;
                    class_delta[index] = wanted - logistic(classes[index]);
                }
            }

            /** @brief Holds where one image's assignment is written and the totals it adds to. */
            struct AssignmentSink
            {
                float* delta = nullptr;                 /**< The image's deltas. */
                std::int64_t* assigned_row = nullptr;   /**< The image's `[H, W, A]` rows. */
                std::int64_t* assigned_class = nullptr; /**< The image's `[H, W, A]` classes. */
                HeadDelta* totals = nullptr;            /**< Receives the count and IoU sum. */
            };

            /**
             * @brief Assigns one truth to the slot of its best anchor and writes that slot.
             * @param head The image's head values.
             * @param sink Left untouched when another head owns the best anchor, when the
             *             truth's centre lies outside the grid or when a side is not positive.
             */
            void assign_truth(const float* head, const DeltaPlan& plan, const Truth& truth,
                              const AssignmentSink& sink) noexcept
            {
                const float grid_x = truth.box.x * static_cast<float>(plan.width);
                const float grid_y = truth.box.y * static_cast<float>(plan.height);
                const float column = std::floor(grid_x);
                const float line = std::floor(grid_y);

                // Each comparison is false for a NaN, so a truth that is not a number is left.
                if (!(column >= 0.0f && column < static_cast<float>(plan.width)) ||
                    !(line >= 0.0f && line < static_cast<float>(plan.height)) ||
                    !(truth.box.width > 0.0f) || !(truth.box.height > 0.0f))
                    return;

                const AnchorSize size{truth.box.width, truth.box.height};
                const std::int64_t found =
                    find_slot(*plan.head_slots, find_best_anchor(*plan.anchors, size));
                if (found < 0)
                    return;

                const std::size_t slot = static_cast<std::size_t>(found);
                const std::size_t row = static_cast<std::size_t>(line);
                const std::size_t col = static_cast<std::size_t>(column);
                const std::size_t position = (row * plan.width + col) * plan.slots + slot;
                const float* values = head + position * plan.entries;
                float* delta = sink.delta + position * plan.entries;
                const SlotReading reading =
                    read_slot(values, plan, row, col, anchor_of_slot(plan, slot));

                BoxDeltaTerms terms;
                terms.predicted = reading.box;
                terms.truth = truth.box;
                terms.anchor = anchor_of_slot(plan, slot);
                terms.offset_x = reading.offset_x;
                terms.offset_y = reading.offset_y;
                terms.raw_width = values[K_WIDTH_ENTRY];
                terms.raw_height = values[K_HEIGHT_ENTRY];
                terms.target_x = grid_x - column;
                terms.target_y = grid_y - line;
                terms.box_weight = plan.training->box_weight;
                terms.max_delta = plan.training->max_delta;
                plan.add_box_delta(terms, delta);

                delta[K_OBJECTNESS_ENTRY] = plan.training->object_weight *
                                            (1.0f - logistic(values[K_OBJECTNESS_ENTRY]));
                write_class_deltas(values, plan, truth.label, delta);

                sink.assigned_row[position] = truth.row;
                sink.assigned_class[position] = truth.label;
                sink.totals->assignments += 1;
                sink.totals->iou_sum +=
                    static_cast<double>(measure_box_iou(reading.box, truth.box));
            }

            /** @brief Fails fatally unless the truth batch holds its shapes and known classes. */
            void check_truth(const Detect::GroundTruthBatch& truth, std::int64_t batch,
                             std::int64_t classes)
            {
                SA_FATAL_IF(truth.batch != batch || truth.max_boxes < 0,
                            "compute_head_delta over {} images needs their truth; got a batch "
                            "of {} with {} rows per image",
                            batch, truth.batch, truth.max_boxes);

                const std::size_t rows =
                    static_cast<std::size_t>(batch) * static_cast<std::size_t>(truth.max_boxes);
                SA_FATAL_IF(truth.labels.size() != rows || truth.valid.size() != rows ||
                                truth.boxes.size() != rows * K_BOX_VALUES,
                            "a truth batch of {} rows needs {} labels, {} flags and {} box "
                            "values; got {}, {} and {}",
                            rows, rows, rows, rows * K_BOX_VALUES, truth.labels.size(),
                            truth.valid.size(), truth.boxes.size());

                for (std::size_t row = 0; row < rows; ++row)
                {
                    SA_FATAL_IF(truth.valid[row] != 0 &&
                                    (truth.labels[row] < 0 || truth.labels[row] >= classes),
                                "truth row {} names class {} of a head with {} classes", row,
                                truth.labels[row], classes);
                }
            }

            /** @brief Fails fatally for a training form whose delta is not computed. */
            void check_computed_form(const HeadOptions& options,
                                     const HeadTrainingOptions& training)
            {
                SA_FATAL_IF(options.coding != HeadCoding::Logit,
                            "compute_head_delta computes the Logit coding alone; got coding {}",
                            static_cast<int>(options.coding));
                SA_FATAL_IF(training.objectness_target != ObjectnessTarget::One,
                            "compute_head_delta computes the objectness target One alone; got "
                            "target {}",
                            static_cast<int>(training.objectness_target));
                SA_FATAL_IF(training.thresholds.match_threshold != 1.0f,
                            "compute_head_delta computes a match threshold of 1 alone; got {}",
                            training.thresholds.match_threshold);
                SA_FATAL_IF(training.class_weight != 1.0f,
                            "compute_head_delta computes a class weight of 1 alone; got {}",
                            training.class_weight);
            }

            /** @brief Collects the valid truths of image @p image as fractions of the input. */
            void collect_truths(const Detect::GroundTruthBatch& truth, std::size_t image,
                                NetworkSize network, std::vector<Truth>& truths)
            {
                const std::size_t rows = static_cast<std::size_t>(truth.max_boxes);
                const float network_width = static_cast<float>(network.width);
                const float network_height = static_cast<float>(network.height);

                truths.clear();
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const std::size_t index = image * rows + row;
                    if (truth.valid[index] == 0)
                        continue;

                    const float* corners = truth.boxes.data() + index * K_BOX_VALUES;
                    Truth collected;
                    collected.box.x = (corners[0] + corners[2]) / 2.0f / network_width;
                    collected.box.y = (corners[1] + corners[3]) / 2.0f / network_height;
                    collected.box.width = (corners[2] - corners[0]) / network_width;
                    collected.box.height = (corners[3] - corners[1]) / network_height;
                    collected.label = truth.labels[index];
                    collected.row = static_cast<std::int64_t>(row);
                    truths.push_back(collected);
                }
            }
        } // namespace

        void compute_head_delta(SushiRuntime::span<const float> head, const HeadShape& shape,
                                const HeadOptions& options, const AnchorLayout& layout,
                                std::int64_t head_index, const HeadTrainingOptions& training,
                                const Detect::GroundTruthBatch& truth, NetworkSize network,
                                HeadDelta& result)
        {
            const Status valid_head = validate_head(options, layout, head_index);
            SA_FATAL_IF(!valid_head.is_ok(),
                        "compute_head_delta was given a head that does not validate: {}",
                        std::string(valid_head.message()));
            const Status valid_training = validate_head_training(training);
            SA_FATAL_IF(!valid_training.is_ok(),
                        "compute_head_delta was given training options that do not validate: {}",
                        std::string(valid_training.message()));
            SA_FATAL_IF(shape.batch < 1 || shape.height < 1 || shape.width < 1,
                        "compute_head_delta needs positive extents; got [{}, {}, {}]",
                        shape.batch, shape.height, shape.width);
            SA_FATAL_IF(network.height < 1 || network.width < 1,
                        "compute_head_delta needs a positive network size; got {} by {}",
                        network.height, network.width);
            check_computed_form(options, training);
            check_truth(truth, shape.batch, options.num_classes);

            DeltaPlan plan;
            plan.height = static_cast<std::size_t>(shape.height);
            plan.width = static_cast<std::size_t>(shape.width);
            plan.anchors = &layout.anchors;
            plan.head_slots = &layout.head_anchors[static_cast<std::size_t>(head_index)];
            plan.slots = plan.head_slots->size();
            plan.classes = static_cast<std::size_t>(options.num_classes);
            plan.entries = static_cast<std::size_t>(K_FIRST_CLASS_ENTRY) + plan.classes;
            plan.scale = options.scale_xy;
            plan.shift = (options.scale_xy - 1.0f) / 2.0f;
            plan.training = &training;
            plan.add_box_delta = find_box_delta(training.box_loss);
            SA_FATAL_IF(plan.add_box_delta == nullptr,
                        "compute_head_delta has no box delta for box loss {}",
                        static_cast<int>(training.box_loss));

            const std::size_t images = static_cast<std::size_t>(shape.batch);
            const std::size_t image_slots = plan.height * plan.width * plan.slots;
            const std::size_t image_values = image_slots * plan.entries;
            SA_FATAL_IF(head.size() != images * image_values,
                        "compute_head_delta over {} images of {} slots needs {} head values; "
                        "got {}",
                        images, image_slots, images * image_values, head.size());

            result.delta.assign(images * image_values, 0.0f);
            result.assigned_row.assign(images * image_slots, K_UNASSIGNED);
            result.assigned_class.assign(images * image_slots, K_UNASSIGNED);
            result.assignments = 0;
            result.iou_sum = 0.0;

            std::vector<Truth> truths;
            truths.reserve(static_cast<std::size_t>(truth.max_boxes));
            for (std::size_t image = 0; image < images; ++image)
            {
                const float* image_head = head.data() + image * image_values;
                AssignmentSink sink;
                sink.delta = result.delta.data() + image * image_values;
                sink.assigned_row = result.assigned_row.data() + image * image_slots;
                sink.assigned_class = result.assigned_class.data() + image * image_slots;
                sink.totals = &result;

                collect_truths(truth, image, network, truths);
                write_background_deltas(image_head, plan, truths, sink.delta);
                for (const Truth& one : truths)
                    assign_truth(image_head, plan, one, sink);
            }
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
