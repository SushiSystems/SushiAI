/**************************************************************************/
/* detection_run_check.hpp                                                */
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

/**
 * @file detection_run_check.hpp
 * @brief Declares the check of a dataset and a run plan that both detection training loops run.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/optim/rate_schedule.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief States what the model being trained fixes about a run. */
    struct DetectionRunSubject
    {
        std::string_view run_name;     /**< Opens a message, such as "An anchor run". */
        std::string_view batch_origin; /**< Says what fixed the batch: "heads were declared for". */
        std::int64_t batch = 0;        /**< Images per step the model was built for. */
        std::int64_t classes = 0;      /**< Classes the model holds. */
    };

    /** @brief States how a run walks the dataset and sets the rate. */
    struct DetectionRunPlan
    {
        std::int64_t epochs = 1;       /**< Passes over the dataset; at least 1. */
        std::int64_t first_epoch = 0;  /**< Epoch index the first pass reads, from 0. */
        std::size_t first_batch = 0;   /**< Batch the first pass starts at. */
        /// @brief Points at the schedule of the rate; null keeps the optimizer's own rate.
        const Optim::RateSchedule* schedule = nullptr;
    };

    /**
     * @brief Copies the fields of the loop options of one engine into a plan.
     * @tparam Options Declares epochs, first_epoch, first_batch and schedule.
     */
    template <typename Options>
    [[nodiscard]] DetectionRunPlan make_detection_run_plan(const Options& options) noexcept
    {
        DetectionRunPlan plan;
        plan.epochs = options.epochs;
        plan.first_epoch = options.first_epoch;
        plan.first_batch = options.first_batch;
        plan.schedule = options.schedule;
        return plan;
    }

    /**
     * @brief Checks that @p dataset can feed the model of @p subject for the run @p plan states.
     * @param dataset Publishes the boxes, classes and valid roles of a detection step.
     * @param subject The batch and the classes the model fixes.
     * @param plan The epochs, the start and the schedule of the run.
     * @return Success; InvalidDescription or Incompatible for a plan or a dataset no run can
     *         use; or the failure of the schedule check or of the role match.
     */
    [[nodiscard]] Status check_detection_run(const Data::Dataset& dataset,
                                             const DetectionRunSubject& subject,
                                             const DetectionRunPlan& plan) noexcept;

} // namespace SushiAI::Vision::Training
