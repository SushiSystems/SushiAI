/**************************************************************************/
/* served_kernel.cpp                                                      */
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
 * @file served_kernel.cpp
 * @brief Implements make_served_kernel(), which picks a machine's replica and builds its kernel.
 * @author Mustafa Garip
 */

#include "served_kernel.hpp"

#include <cstddef>
#include <string>
#include <SushiRuntime/placement/placement_target_matching.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Result<std::unique_ptr<Distribution::ReplicaKernel>> make_served_kernel(
            const std::shared_ptr<const ReplicaSource>& source,
            const Distribution::ClusterSection& section, std::string_view machine)
        {
            SA_FAIL_IF(source == nullptr, StatusCode::InvalidDescription,
                       "a served kernel needs the files of its run");

            const std::size_t replicas = section.plan().replica_count();
            std::size_t replica = 0;
            while (replica < replicas &&
                   !SushiRuntime::machine_names_match(section.worker_of(replica), machine))
                ++replica;

            SA_FAIL_IF(replica == replicas, StatusCode::InvalidDescription,
                       "no replica of the cluster section names the machine '{}'; each entry "
                       "of 'replicas' names its machine under 'worker'",
                       machine);

            SA_TRY_ASSIGN(const std::uint64_t seed,
                          Config::read_run_seed(source->spec, "the training file"));

            Distribution::ReplicaDraws draws;
            draws.seed = seed;
            draws.replica_index = static_cast<std::uint64_t>(replica);
            draws.run_index = source->run_index;
            return Distribution::ReplicaKernel::make(section.plan(),
                                                     TrainingReplicaModel::factory(source),
                                                     draws);
        }

    } // namespace Apps
} // namespace SushiAI
