/**************************************************************************/
/* epoch_loop.hpp                                                         */
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
 * @file epoch_loop.hpp
 * @brief Declares run_epochs, the epoch, schedule and report loop of the training runs.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <SushiAI/failure/status.hpp>

#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds how many epochs pass between two progress lines. */
        inline constexpr std::int64_t K_REPORT_INTERVAL = 5;

        /** @brief Returns whether epoch @p epoch of @p epochs prints a progress line. */
        [[nodiscard]] constexpr bool is_reported_epoch(std::int64_t epoch,
                                                       std::int64_t epochs) noexcept
        {
            return epoch == 1 || epoch % K_REPORT_INTERVAL == 0 || epoch == epochs;
        }

        /**
         * @struct EpochSteps
         * @brief Holds what one run does at each point of the loop run_epochs() drives.
         */
        struct EpochSteps
        {
            /** @brief Enters an epoch index of the data; empty for a run that reads none. */
            std::function<void(std::int64_t epoch_index)> enter_epoch;

            /** @brief Runs one update over a batch index, after the scheduled rate is set. */
            std::function<Status(std::size_t batch_index)> run_step;

            /** @brief Prints the progress line of an epoch, as epoch of epochs. */
            std::function<void(std::int64_t epoch, std::int64_t epochs)> report;
        };

        /**
         * @brief Runs every epoch of @p run from the batch its start names, setting the
         *        scheduled rate before each update and reporting as is_reported_epoch() says.
         * @param steps Its run_step and report are set.
         * @return Success, or the first failure of an update.
         */
        [[nodiscard]] Status run_epochs(const PreparedRun& run, const EpochSteps& steps);

    } // namespace Apps
} // namespace SushiAI
