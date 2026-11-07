/**************************************************************************/
/* cluster_worker_watch.hpp                                              */
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
 * @file cluster_worker_watch.hpp
 * @brief Declares ClusterWorkerWatch, the WorkerWatch that reads a real cluster and the steady
 *        clock.
 * @author Mustafa Garip
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>
#include <SushiRuntime/distributed/cluster.hpp>

#include "master_cluster.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Reads a real cluster and the steady clock. */
        class ClusterWorkerWatch final : public WorkerWatch
        {
            public:
                /** @brief Watches @p cluster, which outlives the watch. */
                explicit ClusterWorkerWatch(SushiRuntime::Distributed::Cluster& cluster) noexcept
                    : cluster_(&cluster)
                {

                }

                /** @brief Returns the names the cluster's statistics list. */
                [[nodiscard]] std::vector<std::string> attached_names() override
                {
                    std::vector<std::string> names;
                    for (const SushiRuntime::Distributed::WorkerSnapshot& worker :
                         cluster_->statistics().workers)
                        names.push_back(worker.name);
                    return names;
                }

                /** @brief Waits on the cluster. */
                [[nodiscard]] bool wait_for_count(std::size_t count,
                                                  std::chrono::milliseconds bound) override
                {
                    return cluster_->wait_for_workers(count, bound);
                }

                /** @brief Returns the steady clock's time. */
                [[nodiscard]] std::chrono::steady_clock::time_point now() override
                {
                    return std::chrono::steady_clock::now();
                }

            private:
                /// @brief The watched cluster; never null.
                SushiRuntime::Distributed::Cluster* cluster_;
        };
    } // namespace Apps
} // namespace SushiAI
