/**************************************************************************/
/* vision_training_runs.hpp                                               */
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
 * @file vision_training_runs.hpp
 * @brief Declares the run row of each detector family `sushiai train` runs.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The run table".
 */

#pragma once

#include <SushiAI/failure/result.hpp>

#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Trains a model with anchor heads through AnchorTrainingStep, first starting
         *        the objectness and class biases when no checkpoint or weights are read.
         * @return The compile count of the training pass and where the run stands;
         *         InvalidDescription when the forward-only pass was not compiled exactly
         *         once; or the first failure of the objective, of the bias start, of the
         *         step, of the start or of run_anchor_training.
         */
        [[nodiscard]] Result<RunSummary> run_anchor(ExecutionBackend& backend, PreparedRun& run);

        /**
         * @brief Trains an anchor-free detector through FreeTrainingStep, first starting
         *        the last biases from the strides when no checkpoint or weights are read.
         * @return The compile count and where the run stands; InvalidDescription when the
         *         part before the ports was not compiled once; or the first failure of the
         *         objective, the bias start, the step or run_free_training.
         */
        [[nodiscard]] Result<RunSummary> run_anchor_free(ExecutionBackend& backend,
                                                         PreparedRun& run);

    } // namespace Apps
} // namespace SushiAI
