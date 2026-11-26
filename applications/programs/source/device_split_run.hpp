/**************************************************************************/
/* device_split_run.hpp                                                   */
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
 * @file device_split_run.hpp
 * @brief Declares run_device_split(), which trains with each batch split across devices.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/execution/runtime_backend.hpp>
#include <SushiAI/failure/result.hpp>

#include "registries.hpp"
#include "training_run.hpp"
#include "training_setup.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Trains @p run with each batch split by rows across its file's "device_split".
         * @param setup Traced over the first share's rows; that share runs the update here.
         * @return The compile count, the progress and one ShareRun per share; InvalidDescription
         *         for rows that miss the batch size, Unsupported for an objective with its own
         *         run or a missing device class, or the first failure of a build, feed or copy.
         * @see applications/programs/README.md, section "Splitting a step across devices".
         */
        [[nodiscard]] Result<RunSummary> run_device_split(const Registries& registries,
                                                          const Config::ModelSpec& model_spec,
                                                          RuntimeBackend& backend,
                                                          TrainingSetup& setup, PreparedRun& run);
    } // namespace Apps
} // namespace SushiAI
