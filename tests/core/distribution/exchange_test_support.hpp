/**************************************************************************/
/* exchange_test_support.hpp                                              */
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
 * @file exchange_test_support.hpp
 * @brief Declares the master and the worker on one InProcessFabric that the exchange tests use.
 * @author Mustafa Garip
 */

#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/cluster_load/offload_policy.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/in_process_transport.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiRuntime/distributed/worker_agent.hpp>
#include <SushiRuntime/placement/placement_rules.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_step.hpp>

/** @brief Holds what the tests of the gradient exchange share. */
namespace SushiAI::Distribution::ExchangeSupport
{
    /** @brief Holds how long a case waits for the worker. */
    inline constexpr std::chrono::seconds K_WAIT{10};

    /** @brief Holds the name the master's runtime reports for its machine. */
    inline constexpr const char* K_MASTER_NAME = "master";

    /** @brief Holds the name the worker's runtime reports for its machine. */
    inline constexpr const char* K_WORKER_NAME = "worker";

    /**
     * @brief Holds a cluster section of two replicas of two rows and one micro-batch each:
     *        replica 0 on the master, replica 1 on the worker, and no fallback for either.
     */
    inline constexpr const char* K_TWO_REPLICAS = R"({
        "listen": "0.0.0.0:7700",
        "secret_file": "cluster.secret",
        "replicas": [
            {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
            {"worker": "worker", "rows": 2, "micro_batch": 2, "if_absent": "fail"}
        ]
    })";

    /** @brief Returns K_TWO_REPLICAS read as a cluster section; a refusal fails the test. */
    inline ClusterSection two_replica_section()
    {
        const Config::Value value =
            Test::require_value(Config::parse_json(K_TWO_REPLICAS, "<test>"));
        return Test::require_value(ClusterSection::read(value, std::filesystem::path(".")));
    }

    /** @brief Returns a runtime whose machine reports @p name, on the devices @p devices takes. */
    inline SushiRuntime::Runtime named_runtime(const char* name,
                                               SushiRuntime::DeviceFilter devices = {})
    {
        SushiRuntime::RuntimeConfig config;
        config.node_name = name;
        config.devices = devices;
        return SushiRuntime::Runtime::create(config).value();
    }

    /** @brief Returns the devices of the in-process worker: the CPU alone, as KI-17 requires. */
    inline SushiRuntime::DeviceFilter worker_devices()
    {
        SushiRuntime::DeviceFilter devices;
        devices.include_gpu = false;
        devices.include_accelerator = false;
        return devices;
    }

    /**
     * @brief Returns the runtime configuration of a replica the in-process worker serves: the
     *        CPU alone, so each machine of the one process drives hardware of its own.
     *
     * Two runtimes of one process on one CUDA GPU on Windows fault a host copy; see KI-17 in
     * docs/reference/KNOWN_ISSUES.md.
     */
    inline SushiRuntime::RuntimeConfig worker_replica_config()
    {
        SushiRuntime::RuntimeConfig config = replica_runtime_config();
        config.devices = worker_devices();
        return config;
    }

    /** @brief Fills a registry with the kernels one machine serves. */
    using KernelSetup = std::function<void(SushiRuntime::Distributed::KernelRegistry&)>;

    /**
     * @brief Holds a master and one worker joined over an InProcessFabric, with the rules of
     *        @p section in force on the master.
     *
     * The master's registry is the binding's, so the kernels it holds run the tasks the
     * rules keep on the master.
     */
    class TwoMachines final
    {
        public:
            /** @brief Starts the worker, attaches the master and waits for the worker. */
            TwoMachines(const ClusterSection& section, const KernelSetup& master_kernels,
                        const KernelSetup& worker_kernels)
                : worker_kernels_(registry(worker_kernels)),
                  worker_runtime_(named_runtime(K_WORKER_NAME, worker_devices())),
                  agent_(worker_runtime_,
                         SushiRuntime::Distributed::InProcessTransport::make_worker(fabric_),
                         worker_kernels_, worker_options()),
                  master_runtime_(named_runtime(K_MASTER_NAME)),
                  cluster_(SushiRuntime::Distributed::Cluster::attach(
                               master_runtime_,
                               SushiRuntime::Distributed::InProcessTransport::make_master(
                                   fabric_),
                               cluster_options()).value()),
                  binding_(cluster_, registry(master_kernels))
            {
                EXPECT_TRUE(master_runtime_.set_placement_rules(
                    section.placement_rules().value()).has_value());
                EXPECT_TRUE(agent_.start().is_ok());
                EXPECT_TRUE(cluster_.wait_for_workers(1, K_WAIT));
            }

            TwoMachines(const TwoMachines&) = delete;
            TwoMachines& operator=(const TwoMachines&) = delete;
            TwoMachines(TwoMachines&&) = delete;
            TwoMachines& operator=(TwoMachines&&) = delete;

            /** @brief Stops the worker before the fabric goes. */
            ~TwoMachines()
            {
                agent_.stop();
            }

            /** @brief Returns the master's runtime. */
            SushiRuntime::Runtime& master_runtime() noexcept
            {
                return master_runtime_;
            }

            /** @brief Returns the master's cluster handle. */
            SushiRuntime::Distributed::Cluster& cluster() noexcept
            {
                return cluster_;
            }

            /** @brief Returns the binding the master's packs are shared through. */
            SushiBLAS::ClusterBinding& binding() noexcept
            {
                return binding_;
            }

        private:
            /** @brief Returns a registry that @p setup filled. */
            static SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> registry(
                const KernelSetup& setup)
            {
                SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> kernels =
                    SushiRuntime::make_sushi<SushiRuntime::Distributed::KernelRegistry>();
                setup(*kernels);
                return kernels;
            }

            /** @brief Returns the options of the worker, which names its machine. */
            static SushiRuntime::Distributed::WorkerAgentOptions worker_options()
            {
                SushiRuntime::Distributed::WorkerAgentOptions options;
                options.name = K_WORKER_NAME;
                return options;
            }

            /** @brief Returns options whose policy would send a task of any size to a worker. */
            static SushiRuntime::Distributed::ClusterOptions cluster_options()
            {
                SushiRuntime::Distributed::ClusterOptions options;
                options.policy =
                    std::make_shared<SushiRuntime::Distributed::LeastLoadedOffloadPolicy>(0);
                return options;
            }

            /// @brief The fabric both machines talk over; it dies last.
            SushiRuntime::Distributed::InProcessFabric fabric_;

            /// @brief The kernels the worker advertises.
            SushiRuntime::sushi_ptr<SushiRuntime::Distributed::KernelRegistry> worker_kernels_;

            /// @brief The worker's runtime.
            SushiRuntime::Runtime worker_runtime_;

            /// @brief The worker, serving over fabric_.
            SushiRuntime::Distributed::WorkerAgent agent_;

            /// @brief The master's runtime, with the cluster attached.
            SushiRuntime::Runtime master_runtime_;

            /// @brief The master.
            SushiRuntime::Distributed::Cluster cluster_;

            /// @brief The master's cluster and the kernels its local tasks run.
            SushiBLAS::ClusterBinding binding_;
    };
} // namespace SushiAI::Distribution::ExchangeSupport
