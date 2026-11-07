/**************************************************************************/
/* cluster_fabric.hpp                                              */
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
 * @file cluster_fabric.hpp
 * @brief Declares Fabric, a master and the workers a test starts
 *        over one in-process fabric.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/in_process_transport.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiRuntime/distributed/worker_agent.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace ClusterTest
        {
            /**
             * @brief Holds a master and the workers a case starts, joined over one
             *        InProcessFabric.
             */
            class Fabric final
            {
                public:
                    /** @brief Attaches a master with no worker yet. */
                    Fabric()
                        : master_runtime_(SushiRuntime::Runtime::create().value()),
                          cluster_(SushiRuntime::Distributed::Cluster::attach(
                                       master_runtime_,
                                       SushiRuntime::Distributed::InProcessTransport::make_master(
                                           fabric_)).value())
                    {

                    }

                    Fabric(const Fabric&) = delete;
                    Fabric& operator=(const Fabric&) = delete;
                    Fabric(Fabric&&) = delete;
                    Fabric& operator=(Fabric&&) = delete;
                    /** @brief Stops the workers before the master. */
                    ~Fabric() = default;

                    /** @brief Starts a worker whose HELLO names @p name. */
                    void start_worker(const std::string& name)
                    {
                        worker_runtimes_.push_back(std::make_unique<SushiRuntime::Runtime>(
                            SushiRuntime::Runtime::create().value()));
                        SushiRuntime::Distributed::WorkerAgentOptions options;
                        options.name = name;
                        agents_.push_back(std::make_unique<SushiRuntime::Distributed::WorkerAgent>(
                            *worker_runtimes_.back(),
                            SushiRuntime::Distributed::InProcessTransport::make_worker(fabric_),
                            SushiRuntime::make_sushi<SushiRuntime::Distributed::KernelRegistry>(),
                            std::move(options)));
                        ASSERT_TRUE(agents_.back()->start().is_ok());
                    }

                    /** @brief Returns the master's cluster handle. */
                    SushiRuntime::Distributed::Cluster& cluster() noexcept
                    {
                        return cluster_;
                    }

                private:
                    /// @brief The fabric every machine talks over; it dies last.
                    SushiRuntime::Distributed::InProcessFabric fabric_;

                    /// @brief The master's runtime, with the cluster attached.
                    SushiRuntime::Runtime master_runtime_;

                    /// @brief The master.
                    SushiRuntime::Distributed::Cluster cluster_;

                    /// @brief The runtime of each worker, in start order.
                    std::vector<std::unique_ptr<SushiRuntime::Runtime>> worker_runtimes_;

                    /// @brief The workers, in start order; they die first.
                    std::vector<std::unique_ptr<SushiRuntime::Distributed::WorkerAgent>> agents_;
            };
        } // namespace ClusterTest
    } // namespace Apps
} // namespace SushiAI
