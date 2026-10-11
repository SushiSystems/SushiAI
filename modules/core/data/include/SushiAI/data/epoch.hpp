/**************************************************************************/
/* epoch.hpp                                                              */
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
 * @file epoch.hpp
 * @brief Declares the interface of a dataset whose batches depend on the epoch.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiAI
{
    namespace Data
    {
        class Dataset;

        /**
         * @class EpochAware
         * @brief The interface of a dataset whose batches depend on the epoch they are read in.
         * @see modules/core/data/README.md, section "The epoch of a read".
         */
        class EpochAware
        {
            public:
                virtual ~EpochAware() = default;

                /**
                 * @brief Makes every read that follows return a batch of @p epoch.
                 * @param epoch Counted from 0; a negative one stops the process.
                 * @pre No other thread is reading the dataset or calling this.
                 */
                virtual void begin_epoch(std::int64_t epoch) const noexcept = 0;
        };

        /**
         * @brief Calls EpochAware::begin_epoch() on a dataset that implements it.
         *
         * A dataset that does not implement EpochAware is left as it is.
         * @pre What EpochAware::begin_epoch() states.
         */
        void begin_epoch(const Dataset& dataset, std::int64_t epoch) noexcept;

    } // namespace Data
} // namespace SushiAI
