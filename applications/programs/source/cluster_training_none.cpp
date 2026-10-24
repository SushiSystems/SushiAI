/**************************************************************************/
/* cluster_training_none.cpp                                              */
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
 * @file cluster_training_none.cpp
 * @brief Implements cluster_training.hpp for a build without the distributed layer.
 * @author Mustafa Garip
 */

#include "cluster_training.hpp"

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Result<SushiRuntime::RuntimeConfig> training_runtime_config(const Config::TrainSpec& spec)
        {
            SA_FAIL_IF(!spec.cluster.is_null(), StatusCode::Unsupported,
                       "the training file has a 'cluster' section, but this build has no "
                       "distributed layer; build it with `sa build --distributed`");
            return SushiRuntime::RuntimeConfig();
        }

        Result<RunSummary> run_cluster_training(PreparedRun&,
                                                SushiRuntime::span<const Optim::ParameterSlot>)
        {
            SA_FAIL_IF(true, StatusCode::Unsupported,
                       "this build has no distributed layer; build it with "
                       "`sa build --distributed`");
            return RunSummary();
        }

    } // namespace Apps
} // namespace SushiAI
