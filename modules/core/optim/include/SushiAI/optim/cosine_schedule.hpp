/**************************************************************************/
/* cosine_schedule.hpp                                                    */
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
 * @file cosine_schedule.hpp
 * @brief Declares the learning rate that warms up linearly and then follows half a cosine.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @brief Holds the learning rate of every update: a linear warm-up, then a cosine decay.
         *
         * Updates are numbered from 1; iteration 0 is the state before any update.
         */
        struct CosineSchedule
        {
            /** @brief Holds the rate the warm-up ends at and the decay starts from. */
            double base_rate = 0.0;

            /** @brief Holds the iterations the warm-up lasts; 0 means there is no warm-up. */
            std::int64_t warmup_iterations = 0;

            /** @brief Holds the iteration the decay ends at; the rate is final_rate from it on. */
            std::int64_t total_iterations = 0;

            /** @brief Holds the rate the decay ends at and every later update uses. */
            double final_rate = 0.0;

            /**
             * @brief Returns the learning rate of update @p iteration.
             * @param iteration Update number from 0; the first update passes 1.
             * @return base_rate * iteration / warmup_iterations below warmup_iterations;
             *         final_rate from total_iterations on; between the two, final_rate plus
             *         (base_rate - final_rate) * (1 + cos(pi * progress)) / 2.
             * @pre validate_cosine_schedule succeeds; a negative iteration is fatal.
             */
            [[nodiscard]] double rate_at(std::int64_t iteration) const noexcept;
        };

        /**
         * @brief Checks that a schedule holds values rate_at can use.
         * @return Success, or InvalidDescription for a negative warm-up length, a decay that
         *         ends before the warm-up does, or a final rate that is negative, not a
         *         number or above base_rate.
         */
        [[nodiscard]] Status validate_cosine_schedule(const CosineSchedule& schedule) noexcept;

    } // namespace Optim
} // namespace SushiAI
