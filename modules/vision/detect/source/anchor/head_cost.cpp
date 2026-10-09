/**************************************************************************/
/* head_cost.cpp                                                          */
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
 * @file head_cost.cpp
 * @brief Implements the reported cost of an anchor head's training delta.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_cost.hpp>

#include <cstddef>

#include <SushiAI/core/check.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Holds the number of box entries at the start of every slot. */
            constexpr std::size_t K_BOX_ENTRIES = static_cast<std::size_t>(K_OBJECTNESS_ENTRY);

            /** @brief Holds the squared sums of a delta, split into box and non-box entries. */
            struct EntrySums
            {
                double box = 0.0;     /**< Sum of squared box entries, undivided. */
                double non_box = 0.0; /**< Sum of squared objectness and class entries. */
            };

            /** @brief Returns the stored cost of a squared-error head: every squared delta. */
            double stored_squared_error(const EntrySums& sums, const HeadDelta&,
                                        const HeadTrainingOptions&) noexcept
            {
                return sums.box + sums.non_box;
            }

            /**
             * @brief Returns the stored cost of a CompleteIou head.
             * @return box_weight times the mean of 1 - IoU over the assignments, 0 without
             *         one, plus object_weight times the non-box sum.
             */
            double stored_complete_iou(const EntrySums& sums, const HeadDelta& delta,
                                       const HeadTrainingOptions& training) noexcept
            {
                double iou_term = 0.0;
                if (delta.assignments > 0)
                {
                    const double mean_one_minus_iou =
                        1.0 - delta.iou_sum / static_cast<double>(delta.assignments);
                    iou_term = static_cast<double>(training.box_weight) * mean_one_minus_iou;
                }
                return iou_term + static_cast<double>(training.object_weight) * sums.non_box;
            }

            /** @brief Holds the signature of a function that returns a stored cost. */
            using StoredCostFunction = double (*)(const EntrySums&, const HeadDelta&,
                                                  const HeadTrainingOptions&) noexcept;

            /** @brief Holds a box loss and the function that returns its stored cost. */
            struct StoredCostRow
            {
                BoxLoss loss;               /**< The loss the row answers for. */
                StoredCostFunction stored; /**< The stored cost of that loss. */
            };

            /** @brief Lists every box loss with the function of its stored cost. */
            constexpr StoredCostRow K_STORED_COST_ROWS[] = {
                {BoxLoss::SquaredError, &stored_squared_error},
                {BoxLoss::CompleteIou, &stored_complete_iou},
            };

            /** @brief Sums the squared entries of every slot, split into box and non-box. */
            EntrySums sum_entries(const std::vector<float>& values, std::size_t entries) noexcept
            {
                EntrySums sums;
                for (std::size_t first = 0; first < values.size(); first += entries)
                {
                    for (std::size_t index = 0; index < entries; ++index)
                    {
                        const double value = static_cast<double>(values[first + index]);
                        if (index < K_BOX_ENTRIES)
                            sums.box += value * value;
                        else
                            sums.non_box += value * value;
                    }
                }
                return sums;
            }

        } // namespace

        HeadCost measure_head_cost(const HeadDelta& delta, const HeadShape& shape,
                                   const HeadOptions& options,
                                   const HeadTrainingOptions& training) noexcept
        {
            SA_FATAL_IF(shape.batch < 1 || shape.height < 1 || shape.width < 1,
                        "measure_head_cost needs a positive batch and grid; got batch {}, "
                        "height {}, width {}",
                        shape.batch, shape.height, shape.width);
            SA_FATAL_IF(options.num_classes < 1,
                        "measure_head_cost needs at least one class; got {}",
                        options.num_classes);

            const std::size_t entries =
                static_cast<std::size_t>(options.num_classes + K_FIRST_CLASS_ENTRY);
            const std::size_t rows = static_cast<std::size_t>(shape.batch) *
                                     static_cast<std::size_t>(shape.height) *
                                     static_cast<std::size_t>(shape.width);
            SA_FATAL_IF(delta.delta.empty() || delta.delta.size() % (rows * entries) != 0,
                        "a delta of {} values does not hold whole slots of {} entries over "
                        "{} grid cells",
                        delta.delta.size(), entries, rows);

            StoredCostFunction stored = nullptr;
            for (const StoredCostRow& row : K_STORED_COST_ROWS)
            {
                if (row.loss == training.box_loss)
                    stored = row.stored;
            }
            SA_FATAL_IF(stored == nullptr, "no stored cost is defined for box loss {}",
                        static_cast<int>(training.box_loss));

            const EntrySums sums = sum_entries(delta.delta, entries);
            const double batch = static_cast<double>(shape.batch);

            HeadCost cost;
            cost.total = (sums.box + sums.non_box) / batch;
            cost.non_box = static_cast<double>(training.object_weight) * sums.non_box / batch;
            cost.box = cost.total - cost.non_box;
            cost.stored = stored(sums, delta, training);
            return cost;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
