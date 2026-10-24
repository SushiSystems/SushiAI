/**************************************************************************/
/* master_cluster.hpp                                                     */
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
 * @file master_cluster.hpp
 * @brief Declares how `sushiai train` becomes the master of the cluster its file names.
 * @author Mustafa Garip
 */

#pragma once

#include <chrono>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Returns the configuration of the master's runtime.
         *
         * Replica 0 is the master, so its runtime reports the machine name replica 0 names
         * and the rule of replica 0 keeps that replica on it.
         */
        [[nodiscard]] SushiRuntime::RuntimeConfig master_runtime_config(
            const Distribution::ClusterSection& section);

        /**
         * @brief Sets the section's placement rules on @p runtime and attaches a TCP master
         *        that listens where the section says, under the secret its file holds.
         * @param runtime Was built from master_runtime_config(); outlives the cluster.
         * @return The cluster, or InvalidDescription for a listen address that does not parse,
         *         or the refusal of the rules, of the secret file or of the attach.
         */
        [[nodiscard]] Result<SushiRuntime::Distributed::Cluster> attach_master_cluster(
            SushiRuntime::Runtime& runtime, const Distribution::ClusterSection& section);

        /**
         * @brief Returns how long the master waits for its workers before the first step.
         * @return The section's attach_timeout(), raised to the largest "wait_ms" of the rules
         *         of replicas 1 and above.
         */
        [[nodiscard]] std::chrono::milliseconds attach_bound(
            const Distribution::ClusterSection& section);

        /**
         * @brief Waits until every worker replicas 1 and above name has attached, or until
         *        attach_bound() passes, whichever comes first.
         * @return Success, also when a worker whose rule is "local" or "wait" is missing; or
         *         DeviceFailed naming every missing worker when the rule of one is "fail".
         */
        [[nodiscard]] Status wait_for_workers(SushiRuntime::Distributed::Cluster& cluster,
                                              const Distribution::ClusterSection& section);

    } // namespace Apps
} // namespace SushiAI
