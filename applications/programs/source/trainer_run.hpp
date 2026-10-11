/**************************************************************************/
/* trainer_run.hpp                                                        */
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
 * @file trainer_run.hpp
 * @brief Declares the function that adds the run of the layers tier to a run registry.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/status.hpp>

#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Sets the fallback run, which trains through Train::Trainer every
         *        objective that fills its own leaves from a batch.
         * @param registry The registry the run joins.
         * @return Success, or InvalidDescription when a fallback is already set.
         */
        [[nodiscard]] Status register_runs(RunRegistry& registry);

    } // namespace Apps
} // namespace SushiAI
