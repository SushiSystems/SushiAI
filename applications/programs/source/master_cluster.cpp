/**************************************************************************/
/* master_cluster.cpp                                                     */
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
 * @file master_cluster.cpp
 * @brief Implements the master side of a cluster run: its runtime's name, rules and transport.
 * @author Mustafa Garip
 */

#include "master_cluster.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/cluster_load/cluster_statistics.hpp>
#include <SushiRuntime/cluster_load/offload_policy.hpp>
#include <SushiRuntime/distributed/shared_secret.hpp>
#include <SushiRuntime/distributed/tcp_transport.hpp>
#include <SushiRuntime/placement/placement_rule.hpp>
#include <SushiRuntime/placement/placement_rules.hpp>
#include <SushiRuntime/placement/placement_target_matching.hpp>
#include <SushiRuntime/transport/endpoint.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Holds which awaited workers have not attached, read from one snapshot. */
            struct Attendance
            {
                /// @brief The workers alive when the snapshot was taken.
                std::size_t alive = 0;

                /// @brief Each awaited worker that was not among them, in order.
                std::vector<AwaitedWorker> absent;
            };

            /** @brief Compares the workers @p cluster lists by name with the awaited ones. */
            [[nodiscard]] Attendance attendance(const SushiRuntime::Distributed::Cluster& cluster,
                                                const std::vector<AwaitedWorker>& awaited)
            {
                const std::vector<SushiRuntime::Distributed::WorkerSnapshot> workers =
                    cluster.statistics().workers;

                Attendance seen;
                seen.alive = workers.size();
                for (const AwaitedWorker& expected : awaited)
                {
                    const bool attached = std::any_of(
                        workers.begin(), workers.end(),
                        [&](const SushiRuntime::Distributed::WorkerSnapshot& worker)
                        {
                            return SushiRuntime::machine_names_match(worker.name, expected.name);
                        });
                    if (!attached)
                        seen.absent.push_back(expected);
                }
                return seen;
            }
        } // namespace

        SushiRuntime::RuntimeConfig master_runtime_config(
            const Distribution::ClusterSection& section)
        {
            SushiRuntime::RuntimeConfig config;
            config.node_name = section.worker_of(0);
            return config;
        }

        Result<SushiRuntime::Distributed::Cluster> attach_master_cluster(
            SushiRuntime::Runtime& runtime, const Distribution::ClusterSection& section)
        {
            namespace Distributed = SushiRuntime::Distributed;

            const Distributed::EndpointParseResult listen =
                Distributed::Endpoint::parse(section.listen_address());
            SA_FAIL_IF(!listen.is_ok(), StatusCode::InvalidDescription,
                       "cluster 'listen': {}", listen.message);

            SA_TRY_ASSIGN(SushiRuntime::PlacementRules rules, section.placement_rules());

            using Detail::SeamPhase;
            const SushiRuntime::Result<void> ruled = runtime.set_placement_rules(std::move(rules));
            if (!ruled)
                return Detail::status_of_error(ruled.error(), SeamPhase::Execute);

            SushiRuntime::Result<Distributed::SharedSecret> secret =
                Distributed::SharedSecret::from_file(section.secret_file());
            if (!secret)
                return Detail::status_of_error(secret.error(), SeamPhase::Allocate);

            try
            {
                Distributed::TCPMasterOptions transport;
                transport.bind = listen.endpoint;

                Distributed::ClusterOptions options;
                options.secret = std::move(secret.value());
                // The rules choose each replica's machine, so a task of any size may leave.
                options.policy = std::make_shared<Distributed::LeastLoadedOffloadPolicy>(0);

                SushiRuntime::Result<Distributed::Cluster> attached = Distributed::Cluster::attach(
                    runtime, Distributed::TCPTransport::make_master(transport), std::move(options));
                if (!attached)
                    return Detail::status_of_error(attached.error(), SeamPhase::Allocate);

                return std::move(attached.value());
            }
            catch (...)
            {
                return Detail::status_of_current_exception(SeamPhase::Allocate);
            }
        }

        std::chrono::milliseconds attach_bound(const Distribution::ClusterSection& section)
        {
            std::chrono::milliseconds bound = section.attach_timeout();
            for (std::size_t replica = 1; replica < section.plan().replica_count(); ++replica)
            {
                const SushiRuntime::AbsentPolicy& policy = section.absent_policy_of(replica);
                if (policy.action == SushiRuntime::AbsentAction::Wait)
                    bound = std::max(bound, policy.bound);
            }
            return bound;
        }

        std::vector<AwaitedWorker> awaited_workers(const Distribution::ClusterSection& section)
        {
            std::vector<AwaitedWorker> awaited;
            for (std::size_t replica = 1; replica < section.plan().replica_count(); ++replica)
            {
                const std::string& name = section.worker_of(replica);
                if (SushiRuntime::machine_names_match(name, section.worker_of(0)))
                    continue;

                const auto known = std::find_if(
                    awaited.begin(), awaited.end(),
                    [&](const AwaitedWorker& worker)
                    {
                        return SushiRuntime::machine_names_match(worker.name, name);
                    });
                if (known == awaited.end())
                    awaited.push_back(AwaitedWorker{name, {replica}});
                else
                    known->replicas.push_back(replica);
            }
            return awaited;
        }

        Result<std::vector<AwaitedWorker>> wait_for_workers(
            SushiRuntime::Distributed::Cluster& cluster,
            const Distribution::ClusterSection& section)
        {
            using Clock = std::chrono::steady_clock;
            const std::chrono::milliseconds bound = attach_bound(section);
            const Clock::time_point deadline = Clock::now() + bound;

            const std::vector<AwaitedWorker> awaited = awaited_workers(section);
            Attendance seen = attendance(cluster, awaited);
            while (!seen.absent.empty())
            {
                const Clock::time_point now = Clock::now();
                if (now >= deadline)
                    break;

                // One worker more than the snapshot held returns at the next attach.
                const bool attached = cluster.wait_for_workers(
                    seen.alive + 1,
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
                seen = attendance(cluster, awaited);
                if (!attached)
                    break;
            }

            std::string missing;
            bool fails = false;
            for (const AwaitedWorker& worker : seen.absent)
            {
                missing += (missing.empty() ? "'" : ", '") + worker.name + "'";
                for (std::size_t replica : worker.replicas)
                {
                    if (section.absent_policy_of(replica).action
                        == SushiRuntime::AbsentAction::Fail)
                        fails = true;
                }
            }
            SA_FAIL_IF(fails, StatusCode::DeviceFailed,
                       "The workers {} did not attach within {} ms, and the 'if_absent' of a "
                       "replica on them is 'fail'",
                       missing, bound.count());
            return std::move(seen.absent);
        }

    } // namespace Apps
} // namespace SushiAI
