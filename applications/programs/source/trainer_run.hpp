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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file trainer_run.hpp
 * @brief Declares the function that adds the run of the layers tier to a run registry.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/status.hpp>

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
