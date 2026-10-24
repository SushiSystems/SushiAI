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

#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/failure/result.hpp>

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

    } // namespace Apps
} // namespace SushiAI
