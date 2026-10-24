/**************************************************************************/
/* cluster_training.cpp                                                   */
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
 * @file cluster_training.cpp
 * @brief Implements cluster_training.hpp for a build with the distributed layer.
 * @author Mustafa Garip
 */

#include "cluster_training.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/failure/status.hpp>

#include "cluster_run.hpp"
#include "master_cluster.hpp"
#include "served_kernel.hpp"
#include "training_replica_model.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Returns the checked cluster section of @p spec. */
            [[nodiscard]] Result<Distribution::ClusterSection> read_section(
                const Config::TrainSpec& spec)
            {
                return Distribution::ClusterSection::read(spec.cluster, spec.base);
            }

            /** @brief Returns what each replica on @p worker does while the worker is absent. */
            [[nodiscard]] std::string describe_absent_replicas(
                const Distribution::ClusterSection& section, const AwaitedWorker& worker)
            {
                std::string text;
                for (std::size_t replica : worker.replicas)
                {
                    const SushiRuntime::AbsentPolicy& policy = section.absent_policy_of(replica);
                    std::string action = "runs locally";
                    if (policy.action == SushiRuntime::AbsentAction::Wait)
                        action = SushiLog::format_text("waits up to {} ms for it",
                                                       policy.bound.count());
                    text += SushiLog::format_text("{}replica {} {}", text.empty() ? "" : "; ",
                                                  replica, action);
                }
                return text;
            }
        } // namespace

        Result<SushiRuntime::RuntimeConfig> training_runtime_config(const Config::TrainSpec& spec)
        {
            if (spec.cluster.is_null())
                return SushiRuntime::RuntimeConfig();

            SA_TRY_ASSIGN(const Distribution::ClusterSection section, read_section(spec));
            return master_runtime_config(section);
        }

        Result<RunSummary> run_cluster_training(
            PreparedRun& run, SushiRuntime::span<const Optim::ParameterSlot> parameters)
        {
            SA_TRY_ASSIGN(const Distribution::ClusterSection section, read_section(run.spec));
            SA_TRY_ASSIGN(const std::shared_ptr<const ReplicaSource> source,
                          ReplicaSource::load(run.spec));
            SA_TRY_ASSIGN(const std::unique_ptr<Distribution::ReplicaKernel> local,
                          make_served_kernel(source, section, section.worker_of(0)));

            SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> kernels =
                SushiRuntime::make_sushi<SushiRuntime::Distributed::KernelRegistry>();
            local->register_with(*kernels);

            SA_TRY_ASSIGN(SushiRuntime::Distributed::Cluster cluster,
                          attach_master_cluster(run.engine.runtime(), section));
            SushiBLAS::ClusterBinding binding(cluster, kernels);

            const std::size_t workers = awaited_workers(section).size();
            std::cout << SushiLog::format_text(
                "Cluster    : {} replicas, listening on {}, waiting for {} workers\n",
                section.plan().replica_count(), section.listen_address(), workers);
            Result<std::vector<AwaitedWorker>> waited = wait_for_workers(cluster, section);
            if (!waited.has_value())
            {
                cluster.detach();
                return waited.status();
            }
            for (const AwaitedWorker& missing : waited.value())
                std::cerr << SushiLog::format_text(
                    "Worker '{}' did not join; {}\n", missing.name,
                    describe_absent_replicas(section, missing));

            ClusterRun cluster_run{run, parameters, section, binding, *local,
                                   source->run_index};
            Result<RunSummary> summary = run_on_cluster(cluster_run);
            cluster.detach();
            return summary;
        }

    } // namespace Apps
} // namespace SushiAI
