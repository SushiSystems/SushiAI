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

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/failure/status.hpp>

#include "cluster_run.hpp"
#include "master_cluster.hpp"
#include "registries.hpp"
#include "served_kernel.hpp"
#include "training_replica_model.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Holds how long the master waits for its workers before the first step. */
            constexpr std::chrono::seconds K_WORKER_WAIT{60};

            /** @brief Holds the run index of every run of a training file. */
            constexpr std::uint64_t K_RUN_INDEX = 0;

            /** @brief Returns the checked cluster section of @p spec. */
            [[nodiscard]] Result<Distribution::ClusterSection> read_section(
                const Config::TrainSpec& spec)
            {
                return Distribution::ClusterSection::read(spec.cluster, spec.base);
            }

            /**
             * @brief Builds the files every replica is built from.
             * @return The source, or InvalidDescription for a file that does not read.
             */
            [[nodiscard]] Result<std::shared_ptr<const ReplicaSource>> read_source(
                const Config::TrainSpec& spec)
            {
                Config::ModelSpec model_spec;
                try
                {
                    model_spec = Config::load_model(spec.model_path);
                }
                catch (const std::exception& error)
                {
                    SA_FAIL_IF(true, StatusCode::InvalidDescription, "{}", error.what());
                }

                SA_TRY_ASSIGN(Registries registries, compose_registries());
                return ReplicaSource::make(std::move(registries), spec, std::move(model_spec));
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
                          read_source(run.spec));
            SA_TRY_ASSIGN(const std::unique_ptr<Distribution::ReplicaKernel> local,
                          make_served_kernel(source, section, section.worker_of(0),
                                             K_RUN_INDEX));

            SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> kernels =
                SushiRuntime::make_sushi<SushiRuntime::Distributed::KernelRegistry>();
            local->register_with(*kernels);

            SA_TRY_ASSIGN(SushiRuntime::Distributed::Cluster cluster,
                          attach_master_cluster(run.engine.runtime(), section));
            SushiBLAS::ClusterBinding binding(cluster, kernels);

            const std::size_t workers = section.plan().replica_count() - 1;
            std::cout << SushiLog::format_text(
                "Cluster    : {} replicas, listening on {}, waiting for {} workers\n",
                section.plan().replica_count(), section.listen_address(), workers);
            if (!cluster.wait_for_workers(workers, K_WORKER_WAIT))
                std::cerr << SushiLog::format_text(
                    "{} of {} workers joined within {} s; each absent worker's replica goes "
                    "as its 'if_absent' says\n",
                    cluster.alive_workers(), workers, K_WORKER_WAIT.count());

            ClusterRun cluster_run{run, parameters, section, binding, *local, K_RUN_INDEX};
            Result<RunSummary> summary = run_on_cluster(cluster_run);
            cluster.detach();
            return summary;
        }

    } // namespace Apps
} // namespace SushiAI
