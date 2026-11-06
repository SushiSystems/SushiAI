/**************************************************************************/
/* training_cluster_section.hpp                                           */
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
 * @file training_cluster_section.hpp
 * @brief Declares read_training_cluster, the cluster section read with "deterministic".
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <SushiAI/config/train_config.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Reads the cluster section of @p spec, refusing "if_absent": "local" while
         *        "deterministic" is true.
         *
         * A local fallback runs on the master's device, whose bits differ. ClusterSection
         * does not know "deterministic", so this function checks the pair.
         * @return The section, or InvalidDescription naming both keys and the replica.
         */
        [[nodiscard]] Result<Distribution::ClusterSection> read_training_cluster(
            const Config::TrainSpec& spec);

    } // namespace Apps
} // namespace SushiAI
