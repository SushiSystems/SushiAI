/**************************************************************************/
/* epoch.cpp                                                              */
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
 * @file epoch.cpp
 * @brief Implements the call that tells a dataset which epoch its reads belong to.
 * @author Mustafa Garip
 */

#include <SushiAI/data/epoch.hpp>

#include <SushiAI/data/dataset.hpp>

namespace SushiAI
{
    namespace Data
    {
        void begin_epoch(const Dataset& dataset, std::int64_t epoch) noexcept
        {
            const EpochAware* const aware = dynamic_cast<const EpochAware*>(&dataset);
            if (aware != nullptr)
                aware->begin_epoch(epoch);
        }

    } // namespace Data
} // namespace SushiAI
