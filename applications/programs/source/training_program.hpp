/**************************************************************************/
/* training_program.hpp                                                   */
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
 * @file training_program.hpp
 * @brief Declares the assembly `sushiai train` runs once it holds a parsed training file.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "training_program.hpp".
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/execution/runtime_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct TrainingOutcome
         * @brief Holds what a finished training run reports to its caller.
         */
        struct TrainingOutcome
        {
            /** @brief What the row of the run table returned. */
            RunSummary summary;

            /** @brief The updates the optimizer has taken. */
            std::uint64_t optimizer_steps = 0;
        };

        /**
         * @brief Trains a parsed training file and saves its checkpoint, on a host backend.
         * @param origin What to call the file in a diagnostic and in the header.
         * @param spec The parsed file, its paths already resolved.
         * @param backend The backend of the run.
         * @return The outcome, or the first failure; Unsupported for a file with a cluster
         *         section, which needs a runtime.
         */
        [[nodiscard]] Result<TrainingOutcome> run_training_file(std::string_view origin,
                                                                const Config::TrainSpec& spec,
                                                                ExecutionBackend& backend);

        /**
         * @brief Trains a parsed training file and saves its checkpoint, on a runtime backend.
         * @param origin What to call the file in a diagnostic and in the header.
         * @param spec The parsed file, its paths already resolved.
         * @param backend The backend of the run, and the runtime a cluster attaches to.
         * @return The outcome, or the first failure.
         */
        [[nodiscard]] Result<TrainingOutcome> run_training_file(std::string_view origin,
                                                                const Config::TrainSpec& spec,
                                                                RuntimeBackend& backend);
    } // namespace Apps
} // namespace SushiAI
