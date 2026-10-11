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
#include <cstddef>
#include <string>

#include <SushiAI/core/check.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/head_truth.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

#include "box_delta.hpp"
#include "slot_decode.hpp"
#include "truth_box.hpp"

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            using Detail::BoxDeltaFunction;
            using Detail::BoxDeltaTerms;
            using Detail::CentreBox;
            using Detail::SlotBox;
            using Detail::SlotFrame;

            /** @brief Holds the value that marks a slot no truth was assigned to. */
            constexpr std::int64_t K_UNASSIGNED = -1;

            /** @brief Holds a box loss, a coding and the function that adds their delta. */
            struct BoxDeltaRow
            {
                BoxLoss loss;         /**< The loss the row answers for. */
                HeadCoding coding;    /**< The coding the row answers for. */
                BoxDeltaFunction add; /**< Adds one truth's box delta to a slot. */
            };

            /** @brief Holds one row per box loss and coding whose delta is written. */
            constexpr BoxDeltaRow K_BOX_DELTA_ROWS[] = {
                {BoxLoss::SquaredError, HeadCoding::Logit, &Detail::add_squared_error_box_delta},
                {BoxLoss::CompleteIoU, HeadCoding::Logit, &Detail::add_complete_iou_box_delta},
                {BoxLoss::CompleteIoU, HeadCoding::Activated,
                 &Detail::add_activated_complete_iou_box_delta},
            };

            /** @brief Returns the function of @p loss under @p coding, or null without a row. */
            BoxDeltaFunction find_box_delta(BoxLoss loss, HeadCoding coding) noexcept
            {
                for (const BoxDeltaRow& row : K_BOX_DELTA_ROWS)
                {
                    if (row.loss == loss && row.coding == coding)
                        return row.add;
                }
                return nullptr;
            }

            /** @brief Names a function that decodes the box of one slot in a cell. */
            using SlotDecodeFunction = SlotBox (*)(const float* slot, const SlotFrame& frame,
                                                   std::size_t row, std::size_t col,
                                                   AnchorSize anchor) noexcept;

            /** @brief Names a function that turns an objectness or class entry into (0, 1). */
            using ActivationFunction = float (*)(float value) noexcept;

            /** @brief Holds a coding and the two readings of a slot under it. */
            struct CodingRow
            {
                HeadCoding coding;           /**< The coding the row answers for. */
                SlotDecodeFunction decode;   /**< Decodes the box of a slot. */
                ActivationFunction activate; /**< Reads an objectness or a class probability. */
            };

            /** @brief Holds one row per coding. */
            constexpr CodingRow K_CODING_ROWS[] = {
                {HeadCoding::Logit, &Detail::decode_slot<Detail::LogitRule>,
                 &Detail::LogitRule::activate},
                {HeadCoding::Activated, &Detail::decode_slot<Detail::ActivatedRule>,
                 &Detail::ActivatedRule::activate},
            };

            /** @brief Returns the row of @p coding, or null when no row holds it. */
            const CodingRow* find_coding(HeadCoding coding) noexcept
            {
                for (const CodingRow& row : K_CODING_ROWS)
                {
                    if (row.coding == coding)
                        return &row;
                }
                return nullptr;
            }

            /**
             * @brief Names a function that returns the objectness delta of an ignored slot.
             * @param background The delta toward 0 the slot held before it was ignored.
             * @param pull The weighted gap between the slot's best IoU and its objectness.
             */
            using IgnoredDeltaFunction = float (*)(float background, float pull) noexcept;

            /** @brief Names a function that reports whether an assigned slot overwrites @p held. */
            using OverwriteFunction = bool (*)(float held) noexcept;

            /** @brief Returns 0: a slot ignored under the target One is left alone. */
            float leave_ignored(float, float) noexcept
            {
                return 0.0f;
            }

            /** @brief Returns the larger of the two deltas, for the target BestIoU. */
            float pull_ignored(float background, float pull) noexcept
            {
                return std::max(background, pull);
            }

            /** @brief Returns true: under the target One an assigned slot always overwrites. */
            bool overwrite_always(float) noexcept
            {
                return true;
            }

            /** @brief Reports whether @p held is still 0, the overwrite rule of BestIoU. */
            bool overwrite_zero(float held) noexcept
            {
                return held == 0.0f;
            }

            /** @brief Holds an objectness target and the two rules it sets. */
            struct ObjectnessRow
            {
                ObjectnessTarget target;      /**< The target the row answers for. */
                IgnoredDeltaFunction ignored; /**< Gives the delta of an ignored slot. */
                OverwriteFunction overwrites; /**< Decides the write of an assigned slot. */
            };

            /** @brief Holds one row per objectness target. */
            constexpr ObjectnessRow K_OBJECTNESS_ROWS[] = {
                {ObjectnessTarget::One, &leave_ignored, &overwrite_always},
                {ObjectnessTarget::BestIoU, &pull_ignored, &overwrite_zero},
            };

            /** @brief Returns the row of @p target, or null when no row holds it. */
            const ObjectnessRow* find_objectness(ObjectnessTarget target) noexcept
            {
                for (const ObjectnessRow& row : K_OBJECTNESS_ROWS)
                {
                    if (row.target == target)
                        return &row;
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
                SlotFrame frame;         /**< Grid and stretch of the head's slots. */
                const std::vector<AnchorSize>* anchors = nullptr;      /**< Every anchor. */
                const std::vector<std::int64_t>* head_slots = nullptr; /**< The head's anchors. */
                const HeadTrainingOptions* training = nullptr;         /**< Weights and gates. */
                BoxDeltaFunction add_box_delta = nullptr;              /**< The box term. */
                const CodingRow* coding = nullptr;         /**< Readings of a slot. */
                const ObjectnessRow* objectness = nullptr; /**< Rules of the objectness target. */
            };

            /** @brief Returns the centre and size of a decoded slot. */
            CentreBox box_of(const SlotBox& decoded) noexcept
            {
                CentreBox box;
                box.x = decoded.x;
                box.y = decoded.y;
                box.width = decoded.width;
                box.height = decoded.height;
                return box;
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
                    if (plan.coding->activate(classes[index]) > plan.training->class_gate)
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
             *              threshold and has a gated class takes the delta its target gives.
             */
            void write_background_deltas(const float* head, const DeltaPlan& plan,
                                         const std::vector<Truth>& truths, float* delta) noexcept
            {
                const float ignore = plan.training->thresholds.ignore_threshold;
                const float weight = plan.training->object_weight;

                for (std::size_t row = 0; row < plan.height; ++row)
                {
                    for (std::size_t col = 0; col < plan.width; ++col)
                    {
                        for (std::size_t slot = 0; slot < plan.slots; ++slot)
                        {
                            const CentreBox box = box_of(plan.coding->decode(
                                head, plan.frame, row, col, anchor_of_slot(plan, slot)));
                            const float objectness =
                                plan.coding->activate(head[K_OBJECTNESS_ENTRY]);
                            const float best = measure_best_iou(box, truths);
                            const float background = weight * (0.0f - objectness);

                            delta[K_OBJECTNESS_ENTRY] =
                                best > ignore && has_gated_class(head, plan)
                                    ? plan.objectness->ignored(background,
                                                               weight * (best - objectness))
                                    : background;
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
                    class_delta[target] = 1.0f - plan.coding->activate(classes[target]);
                    return;
                }

                for (std::size_t index = 0; index < plan.classes; ++index)
                {
                    const float wanted = index == target ? 1.0f : 0.0f;
                    class_delta[index] = wanted - plan.coding->activate(classes[index]);
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

            /** @brief Holds the cell a truth's centre falls in and its place within the cell. */
            struct TruthCell
            {
                std::size_t row = 0;   /**< Grid row of the centre. */
                std::size_t col = 0;   /**< Grid column of the centre. */
                float target_x = 0.0f; /**< The centre within the cell, along the columns. */
                float target_y = 0.0f; /**< The centre within the cell, along the rows. */
            };

            /**
             * @brief Returns the cell of a truth's centre.
             * @param truth A truth whose centre lies in [0, 1) on both axes.
             */
            TruthCell locate_cell(const DeltaPlan& plan, const Truth& truth) noexcept
            {
                const float grid_x = truth.box.x * static_cast<float>(plan.width);
                const float grid_y = truth.box.y * static_cast<float>(plan.height);

                // A product that rounds up to the grid's extent still lies in the last cell.
                TruthCell cell;
                cell.col = std::min(static_cast<std::size_t>(grid_x), plan.width - 1);
                cell.row = std::min(static_cast<std::size_t>(grid_y), plan.height - 1);
                cell.target_x = grid_x - static_cast<float>(cell.col);
                cell.target_y = grid_y - static_cast<float>(cell.row);
                return cell;
            }

            /**
             * @brief Writes the box, objectness and class deltas of one slot toward a truth.
             * @param head The image's head values.
             * @param sink Receives the deltas, and one assignment with its IoU in the totals.
             *             The objectness delta is written where the target's rule allows it.
             * @return The index of the slot in the image's `[H, W, A]`.
             */
            std::size_t write_assigned_slot(const float* head, const DeltaPlan& plan,
                                            const Truth& truth, const TruthCell& cell,
                                            std::size_t slot, const AssignmentSink& sink) noexcept
            {
                const std::size_t position =
                    (cell.row * plan.width + cell.col) * plan.slots + slot;
                const float* values = head + position * plan.entries;
                float* delta = sink.delta + position * plan.entries;
                const AnchorSize anchor = anchor_of_slot(plan, slot);
                const SlotBox decoded =
                    plan.coding->decode(values, plan.frame, cell.row, cell.col, anchor);
                const CentreBox box = box_of(decoded);

                BoxDeltaTerms terms;
                terms.predicted = box;
                terms.truth = truth.box;
                terms.anchor = anchor;
                terms.offset_x = decoded.offset_x;
                terms.offset_y = decoded.offset_y;
                terms.raw_width = values[K_WIDTH_ENTRY];
                terms.raw_height = values[K_HEIGHT_ENTRY];
                terms.target_x = cell.target_x;
                terms.target_y = cell.target_y;
                terms.box_weight = plan.training->box_weight;
                terms.max_delta = plan.training->max_delta;
                terms.gradient = plan.training->box_gradient;
                plan.add_box_delta(terms, delta);

                if (plan.objectness->overwrites(delta[K_OBJECTNESS_ENTRY]))
                {
                    delta[K_OBJECTNESS_ENTRY] =
                        plan.training->object_weight *
                        (1.0f - plan.coding->activate(values[K_OBJECTNESS_ENTRY]));
                }
                write_class_deltas(values, plan, truth.label, delta);

                sink.totals->assignments += 1;
                sink.totals->iou_sum += static_cast<double>(measure_box_iou(box, truth.box));
                return position;
            }

            /**
             * @brief Assigns one truth to the slots of the head that take it and writes them.
             * @param head The image's head values.
             * @param sink The slot of the truth's best anchor records the truth's row and
             *             class. Below a match threshold of 1 every other slot whose anchor's
             *             shape IoU with the truth exceeds the threshold is written too.
             */
            void assign_truth(const float* head, const DeltaPlan& plan, const Truth& truth,
                              const AssignmentSink& sink) noexcept
            {
                const TruthCell cell = locate_cell(plan, truth);
                const AnchorSize size{truth.box.width, truth.box.height};
                const std::int64_t best = find_best_anchor(*plan.anchors, size);
                const std::int64_t found = find_slot(*plan.head_slots, best);
                if (found >= 0)
                {
                    const std::size_t position = write_assigned_slot(
                        head, plan, truth, cell, static_cast<std::size_t>(found), sink);
                    sink.assigned_row[position] = truth.row;
                    sink.assigned_class[position] = truth.label;
                }

                const float threshold = plan.training->thresholds.match_threshold;
                if (!(threshold < 1.0f))
                    return;

                for (std::size_t slot = 0; slot < plan.slots; ++slot)
                {
                    if ((*plan.head_slots)[slot] == best)
                        continue;
                    if (shape_iou(anchor_of_slot(plan, slot), size) > threshold)
                        write_assigned_slot(head, plan, truth, cell, slot, sink);
                }
            }

            /**
             * @brief Divides the box deltas of each slot by its count of positive class deltas.
             * @param delta The image's deltas. A slot whose objectness delta is 0, or that has
             *              no positive class delta, keeps its box deltas.
             */
            void share_box_deltas(const DeltaPlan& plan, float* delta) noexcept
            {
                const std::size_t positions = plan.height * plan.width * plan.slots;
                for (std::size_t position = 0; position < positions; ++position)
                {
                    float* slot = delta + position * plan.entries;
                    if (slot[K_OBJECTNESS_ENTRY] == 0.0f)
                        continue;

                    const float* classes = slot + K_FIRST_CLASS_ENTRY;
                    std::size_t positive = 0;
                    for (std::size_t index = 0; index < plan.classes; ++index)
                    {
                        if (classes[index] > 0.0f)
                            ++positive;
                    }
                    if (positive == 0)
                        continue;

                    const float share = static_cast<float>(positive);
                    slot[K_X_ENTRY] /= share;
                    slot[K_Y_ENTRY] /= share;
                    slot[K_WIDTH_ENTRY] /= share;
                    slot[K_HEIGHT_ENTRY] /= share;
                }
            }

            /** @brief Collects the valid truths of image @p image as fractions of the input. */
            void collect_truths(const Detect::GroundTruthBatch& truth, std::size_t image,
                                NetworkSize network, std::vector<Truth>& truths)
            {
                const std::size_t rows = static_cast<std::size_t>(truth.max_boxes);

                truths.clear();
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const std::size_t index = image * rows + row;
                    if (truth.valid[index] == 0)
                        continue;

                    Truth collected;
                    collected.box = Detail::read_truth_box(
                        truth.boxes.data() + index * K_BOX_VALUES, network);
                    collected.label = truth.labels[index];
                    collected.row = static_cast<std::int64_t>(row);
                    truths.push_back(collected);
                }
            }
        } // namespace

        Status compute_head_delta(SushiRuntime::span<const float> head, const HeadShape& shape,
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

            DeltaPlan plan;
            plan.height = static_cast<std::size_t>(shape.height);
            plan.width = static_cast<std::size_t>(shape.width);
            plan.anchors = &layout.anchors;
            plan.head_slots = &layout.head_anchors[static_cast<std::size_t>(head_index)];
            plan.slots = plan.head_slots->size();
            plan.classes = static_cast<std::size_t>(options.num_classes);
            plan.entries = static_cast<std::size_t>(K_FIRST_CLASS_ENTRY) + plan.classes;
            plan.frame = Detail::make_slot_frame(plan.height, plan.width, options.scale_xy);
            plan.training = &training;
            plan.coding = find_coding(options.coding);
            SA_FATAL_IF(plan.coding == nullptr,
                        "compute_head_delta was given a coding outside HeadCoding: {}",
                        static_cast<int>(options.coding));
            plan.objectness = find_objectness(training.objectness_target);
            SA_FATAL_IF(plan.objectness == nullptr,
                        "compute_head_delta was given a target outside ObjectnessTarget: {}",
                        static_cast<int>(training.objectness_target));
            plan.add_box_delta = find_box_delta(training.box_loss, options.coding);
            SA_FATAL_IF(plan.add_box_delta == nullptr,
                        "compute_head_delta has no box delta for box loss {} under coding {}",
                        static_cast<int>(training.box_loss), static_cast<int>(options.coding));

            const std::size_t images = static_cast<std::size_t>(shape.batch);
            const std::size_t image_slots = plan.height * plan.width * plan.slots;
            const std::size_t image_values = image_slots * plan.entries;
            SA_FATAL_IF(head.size() != images * image_values,
                        "compute_head_delta over {} images of {} slots needs {} head values; "
                        "got {}",
                        images, image_slots, images * image_values, head.size());

            // Every check of the caller's arguments is above; the truth is content.
            SA_TRY(validate_head_truth(truth, shape.batch, options.num_classes, network));

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
                if (training.thresholds.match_threshold < 1.0f)
                    share_box_deltas(plan, sink.delta);
            }
            return Status();
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
