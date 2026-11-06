/**************************************************************************/
/* training_cluster_section.cpp                                           */
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
 * @file training_cluster_section.cpp
 * @brief Implements read_training_cluster.
 * @author Mustafa Garip
 */

#include "training_cluster_section.hpp"

#include <cstddef>
#include <SushiRuntime/placement/placement_rule.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Result<Distribution::ClusterSection> read_training_cluster(const Config::TrainSpec& spec)
        {
            SA_TRY_ASSIGN(Distribution::ClusterSection section,
                          Distribution::ClusterSection::read(spec.cluster, spec.base));
            if (!spec.deterministic)
                return section;

            const std::size_t count = section.plan().replica_count();
            for (std::size_t replica = 0; replica < count; ++replica)
            {
                SA_FAIL_IF(section.absent_policy_of(replica).action
                               == SushiRuntime::AbsentAction::Local,
                           StatusCode::InvalidDescription,
                           "'cluster' replica {} states 'if_absent': 'local', which runs a lost "
                           "worker's replica on the master's device, and 'deterministic' is true; "
                           "state 'deterministic': false in a version 3 file, or use 'wait' or "
                           "'fail'",
                           replica);
            }
            return section;
        }

    } // namespace Apps
} // namespace SushiAI
