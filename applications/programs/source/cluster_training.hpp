/**************************************************************************/
/* cluster_training.hpp                                                   */
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
 * @file cluster_training.hpp
 * @brief Declares what `sushiai train` does with a training file's cluster section.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/execution/runtime_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/optimizer.hpp>

#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Returns the configuration of the runtime `sushiai train` builds for @p spec.
         *
         * A build without the distributed layer links cluster_training_none.cpp, whose
         * version refuses a file with a cluster section as Unsupported.
         * @return The default configuration for a file without a cluster section; for one
         *         with it, the master's; or the refusal of the section.
         */
        [[nodiscard]] Result<SushiRuntime::RuntimeConfig> training_runtime_config(
            const Config::TrainSpec& spec);

        /**
         * @brief Attaches the master the cluster section of @p run describes and trains it.
         * @param backend The backend of the run; it runs on the runtime built from
         *        training_runtime_config(), which the master attaches its cluster to.
         * @param run What the run trains.
         * @param parameters Each parameter with its gradient, as bind_parameters() returned.
         * @return What the run returns, a refusal, or Unsupported without the distributed layer.
         */
        [[nodiscard]] Result<RunSummary> run_cluster_training(
            RuntimeBackend& backend, PreparedRun& run,
            SushiRuntime::span<const Optim::ParameterSlot> parameters);

    } // namespace Apps
} // namespace SushiAI
