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

#include <memory>
#include <utility>
#include <SushiRuntime/cluster_load/offload_policy.hpp>
#include <SushiRuntime/distributed/shared_secret.hpp>
#include <SushiRuntime/distributed/tcp_transport.hpp>
#include <SushiRuntime/placement/placement_rules.hpp>
#include <SushiRuntime/transport/endpoint.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Apps
    {
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

    } // namespace Apps
} // namespace SushiAI
