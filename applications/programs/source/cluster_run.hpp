/**************************************************************************/
/* cluster_run.hpp                                                        */
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
 * @file cluster_run.hpp
 * @brief Declares the training run of `sushiai train` over the replicas of a cluster section.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/optimizer.hpp>

#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct ClusterRun
         * @brief Holds what the master built before a cluster run starts; the run owns none.
         */
        struct ClusterRun
        {
            /** @brief What every run of `sushiai train` takes; the step moves its updates. */
            PreparedRun& prepared;

            /** @brief Each parameter with its gradient, as bind_parameters() returned them. */
            SushiRuntime::span<const Optim::ParameterSlot> parameters;

            /** @brief The cluster section of the training file. */
            const Distribution::ClusterSection& section;

            /** @brief The master's cluster and the registry its local tasks run from. */
            SushiBLAS::ClusterBinding& binding;

            /** @brief The kernel registered in that registry, which runs a replica locally. */
            const Distribution::ReplicaKernel& local;

            /** @brief The run index every replica's draws take. */
            std::uint64_t run_index = 0;
        };

        /**
         * @brief Trains every batch of every epoch over the section's replicas; the step holds
         *        the gradient exchange, the optimizer and the parameter publish, and no graph.
         * @return The compile count and where the run stands; InvalidDescription when the
         *         section's rows do not sum to the data's batch; Incompatible for a start the
         *         data lacks; or the first failure of a bind, a compile or a step.
         */
        [[nodiscard]] Result<RunSummary> run_on_cluster(ClusterRun& run);

    } // namespace Apps
} // namespace SushiAI
