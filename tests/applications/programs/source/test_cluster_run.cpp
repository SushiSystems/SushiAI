/**************************************************************************/
/* test_cluster_run.cpp                                                   */
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
 * @file test_cluster_run.cpp
 * @brief Tests one step of `sushiai train` over a master and a worker in one process.
 * @author Mustafa Garip
 */

#include <chrono>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/cluster_load/offload_policy.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/in_process_transport.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiRuntime/distributed/worker_agent.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>

#include "cluster_run.hpp"
#include "cluster_test_files.hpp"
#include "master_cluster.hpp"
#include "registries.hpp"
#include "run_start.hpp"
#include "served_kernel.hpp"
#include "training_replica_model.hpp"
#include "training_run.hpp"
#include "training_setup.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;
    namespace Distributed = SushiRuntime::Distributed;

    /** @brief Holds how long the master waits for the worker. */
    constexpr std::chrono::seconds K_WAIT{10};

    /** @brief Returns the bytes of the parameter of every slot of @p slots, in order. */
    std::vector<std::vector<unsigned char>> bytes_of(
        const std::vector<Optim::ParameterSlot>& slots)
    {
        std::vector<std::vector<unsigned char>> bytes;
        for (const Optim::ParameterSlot& slot : slots)
        {
            const std::size_t size = static_cast<std::size_t>(slot.parameter.num_elements)
                                     * dtype_size(slot.parameter.dtype);
            const unsigned char* const first =
                static_cast<const unsigned char*>(slot.parameter.data());
            bytes.emplace_back(first, first + size);
        }
        return bytes;
    }

    /** @brief Returns the options of a worker whose machine reports K_WORKER_NAME. */
    Distributed::WorkerAgentOptions worker_options()
    {
        Distributed::WorkerAgentOptions options;
        options.name = K_WORKER_NAME;
        return options;
    }
} // namespace

/**
 * @brief Tests that the master's cluster run trains one step from a training file, with
 *        replica 1 served by the worker's kernel over an InProcessFabric.
 */
TEST(Medium_ClusterRun, OneStepFromAFileRunsReplicaOneOnTheWorker)
{
    const ClusterTestFiles files("sushiai_cluster_run_model.json");
    const Config::TrainSpec& spec = files.spec;
    const Distribution::ClusterSection section = files.section();

    const std::shared_ptr<const Apps::ReplicaSource> source =
        Apps::ReplicaSource::make(Apps::compose_registries().value(), spec, files.model_spec)
            .value();
    std::unique_ptr<Distribution::ReplicaKernel> worker_kernel =
        Apps::make_served_kernel(source, section, K_WORKER_NAME, 0).value();
    std::unique_ptr<Distribution::ReplicaKernel> master_kernel =
        Apps::make_served_kernel(source, section, section.worker_of(0), 0).value();

    // The fabric outlives both ends; the worker stops before it goes.
    Distributed::InProcessFabric fabric;
    SushiRuntime::sushi_ptr<Distributed::KernelRegistry> worker_kernels =
        SushiRuntime::make_sushi<Distributed::KernelRegistry>();
    worker_kernel->register_with(*worker_kernels);
    SushiRuntime::RuntimeConfig worker_config;
    worker_config.node_name = K_WORKER_NAME;
    SushiRuntime::Runtime worker_runtime = SushiRuntime::Runtime::create(worker_config).value();
    Distributed::WorkerAgent agent(worker_runtime,
                                   Distributed::InProcessTransport::make_worker(fabric),
                                   worker_kernels, worker_options());

    SushiRuntime::Runtime master_runtime =
        Detail::create_runtime(Apps::master_runtime_config(section)).value();
    ASSERT_TRUE(master_runtime.set_placement_rules(section.placement_rules().value())
                    .has_value());
    Distributed::ClusterOptions cluster_options;
    cluster_options.policy = std::make_shared<Distributed::LeastLoadedOffloadPolicy>(0);
    Distributed::Cluster cluster =
        Distributed::Cluster::attach(master_runtime,
                                     Distributed::InProcessTransport::make_master(fabric),
                                     std::move(cluster_options))
            .value();
    SushiRuntime::sushi_ptr<Distributed::KernelRegistry> master_kernels =
        SushiRuntime::make_sushi<Distributed::KernelRegistry>();
    master_kernel->register_with(*master_kernels);
    SushiBLAS::ClusterBinding binding(cluster, master_kernels);

    ASSERT_TRUE(agent.start().is_ok());
    ASSERT_TRUE(cluster.wait_for_workers(1, K_WAIT));

    SushiBLAS::Engine engine(master_runtime);
    const Apps::Registries registries = Apps::compose_registries().value();
    std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, spec, files.model_spec, engine, K_SEED).value();
    std::unique_ptr<Optim::Optimizer> optimizer =
        Optim::make_optimizer(engine, spec.optimizer_type, setup->parameters,
                              spec.optimizer_attributes, "the optimizer block")
            .value();
    IO::RunProgress start;
    SA_EXPECT_OK(Apps::start_state(spec, engine, *setup->state, *optimizer, &start));
    const std::vector<std::vector<unsigned char>> before = bytes_of(setup->parameters);

    Apps::PreparedRun prepared{
        engine, spec, *setup->dataset, setup->given, *setup->objective, setup->built,
        *setup->state, *setup->pool, *optimizer, std::move(setup->updates), nullptr, start,
    };
    Apps::ClusterRun run{prepared,
                         SushiRuntime::span<const Optim::ParameterSlot>(
                             setup->parameters.data(), setup->parameters.size()),
                         section, binding, *master_kernel, 0};
    const Result<Apps::RunSummary> summary = Apps::run_on_cluster(run);

    ASSERT_TRUE(summary.has_value()) << summary.status().message();
    EXPECT_EQ(summary.value().compile_count, 1u);
    EXPECT_EQ(summary.value().progress.step_runs, 1u);
    EXPECT_EQ(cluster.offloaded(), 1u) << "replica 1 ran somewhere other than the worker";
    EXPECT_NE(bytes_of(setup->parameters), before) << "the step did not update a parameter";

    agent.stop();
}
