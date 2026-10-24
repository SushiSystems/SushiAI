/**************************************************************************/
/* served_kernel.hpp                                                      */
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
 * @file served_kernel.hpp
 * @brief Declares make_served_kernel(), the replica kernel one machine of a cluster serves.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/failure/result.hpp>

#include "training_replica_model.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Builds the kernel of the replica the section places on @p machine: the
         *        master's local replica 0, or the replica a worker's name picks.
         * @param source Not null; the files of the run.
         * @param run_index The run index of the replica's draws.
         * @return The kernel, InvalidDescription when no replica names @p machine, or the
         *         refusal of the seed or of the replica's step.
         */
        [[nodiscard]] Result<std::unique_ptr<Distribution::ReplicaKernel>> make_served_kernel(
            const std::shared_ptr<const ReplicaSource>& source,
            const Distribution::ClusterSection& section, std::string_view machine,
            std::uint64_t run_index);

    } // namespace Apps
} // namespace SushiAI
