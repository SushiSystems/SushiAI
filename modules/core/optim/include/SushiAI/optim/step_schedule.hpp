/**************************************************************************/
/* step_schedule.hpp                                                      */
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
 * @file step_schedule.hpp
 * @brief Declares the learning-rate schedule of a training run: a warm-up, then steps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @brief Holds the learning rate of every update: a warm-up ramp, then steps.
         *
         * Updates are numbered from 1; iteration 0 is the state before any update.
         */
        struct StepSchedule
        {
            /** @brief Holds the rate once the warm-up is over and before any step applies. */
            double base_rate = 0.0;

            /** @brief Holds the iterations the warm-up lasts; 0 means there is no warm-up. */
            std::int64_t warmup_iterations = 0;

            /** @brief Holds the exponent of the warm-up ramp. */
            double warmup_power = 4.0;

            /** @brief Holds the iterations at which a step applies, strictly ascending. */
            std::vector<std::int64_t> steps;

            /** @brief Holds the factor each step multiplies the rate by; as long as steps. */
            std::vector<double> scales;

            /**
             * @brief Returns the learning rate of update @p iteration.
             * @param iteration Update number from 0; the first update passes 1.
             * @return The warm-up ramp while iteration is below warmup_iterations; otherwise
             *         base_rate times every scale whose step is at or before iteration.
             * @pre validate_step_schedule succeeds on this schedule; a negative iteration
             *      or lists of different lengths are fatal.
             */
            [[nodiscard]] double rate_at(std::int64_t iteration) const noexcept;
        };

        /**
         * @brief Checks that a schedule holds values rate_at can use.
         * @return Success, or InvalidDescription for steps and scales of different lengths,
         *         a negative warm-up length, a warm-up power that is negative or not a
         *         number, or steps that do not strictly ascend.
         */
        [[nodiscard]] Status validate_step_schedule(const StepSchedule& schedule);

    } // namespace Optim
} // namespace SushiAI
