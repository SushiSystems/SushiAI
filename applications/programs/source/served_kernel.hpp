/**************************************************************************/
/* served_kernel.hpp                                                      */
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
 * @file served_kernel.hpp
 * @brief Declares make_served_kernel(), the replica kernel one machine of a cluster serves.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/failure/result.hpp>

#include "training_replica_model.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Builds the kernel of the replica the section places on @p machine: the
         *        master's local replica 0, or the replica a worker's name picks.
         * @param source Not null; the files of the run, whose run index the draws take.
         * @param config The configuration of the runtime the replica's step drives.
         * @return The kernel, InvalidDescription when no replica names @p machine, or the
         *         refusal of the seed or of the replica's step.
         */
        [[nodiscard]] Result<std::unique_ptr<Distribution::ReplicaKernel>> make_served_kernel(
            const std::shared_ptr<const ReplicaSource>& source,
            const Distribution::ClusterSection& section, std::string_view machine,
            const SushiRuntime::RuntimeConfig& config = Distribution::replica_runtime_config());

    } // namespace Apps
} // namespace SushiAI
