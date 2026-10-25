/**************************************************************************/
/* free_training_loop.hpp                                                 */
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
 * @file free_training_loop.hpp
 * @brief Declares the loop that drives an anchor-free training step over a dataset.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds what one update of an anchor-free run produced. */
    struct FreeProgress
    {
        std::int64_t epoch = 0;     /**< Pass over the dataset, from 1. */
        std::size_t batch = 0;      /**< Batch of the pass, from 0. */
        std::int64_t iteration = 0; /**< Updates the optimizer has applied, this one counted. */
        double rate = 0.0;          /**< Learning rate the update was given. */
        FreeLossTerms terms;        /**< Loss terms of the leaves the update was fed. */
    };

    /** @brief Receives the progress of one update; a throw ends the run as a failure. */
    using FreeProgressCallback = std::function<void(const FreeProgress&)>;

    /** @brief States how an anchor-free run is driven. */
    struct FreeLoopOptions
    {
        std::int64_t epochs = 1; /**< Passes over the dataset; at least 1. */
        /// @brief Points at the schedule of the rate; null keeps the optimizer's own rate.
        const Optim::RateSchedule* schedule = nullptr;
        FreeProgressCallback on_update; /**< Called after each update; may be empty. */
        std::int64_t first_epoch = 0;   /**< Epoch index the first pass reads, from 0. */
        std::size_t first_batch = 0;    /**< Batch the first pass starts at; below the count. */
    };

    /** @brief Holds what a whole anchor-free run produced. */
    struct FreeRunReport
    {
        std::size_t updates = 0; /**< Batches trained on. */
        FreeLossTerms first;     /**< Terms of the first update. */
        FreeLossTerms last;      /**< Terms of the last update. */
        double last_rate = 0.0;  /**< Learning rate of the last update. */
    };

    /**
     * @brief Trains @p step on every batch of @p dataset, once per epoch of @p options.
     * @param dataset Publishes the boxes, classes and valid roles in the step's batch.
     * @param options Its first pass reads epoch @c first_epoch from batch @c first_batch on.
     * @return The report; InvalidDescription or Incompatible for a run the options or the
     *         dataset cannot give; or the failure of a read, an update or the callback.
     */
    [[nodiscard]] Result<FreeRunReport> run_free_training(FreeTrainingStep& step,
                                                          const Data::Dataset& dataset,
                                                          const FreeLoopOptions& options = {});

} // namespace SushiAI::Vision::Training
