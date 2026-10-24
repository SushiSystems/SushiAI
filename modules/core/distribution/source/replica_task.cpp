/**************************************************************************/
/* replica_task.cpp                                                       */
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
 * @file replica_task.cpp
 * @brief Implements the encoding of ReplicaTask into a node's parameters.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/replica_task.hpp>

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        std::array<std::uint64_t, K_REPLICA_TASK_WORDS> ReplicaTask::encode() const noexcept
        {
            return {epoch, batch_index, replica_index, replica_count, run_index, seed};
        }

        Result<ReplicaTask> ReplicaTask::decode(SushiRuntime::span<const std::uint64_t> words)
        {
            SA_FAIL_IF(words.size() != K_REPLICA_TASK_WORDS, StatusCode::InvalidDescription,
                       "A replica node carries {} parameters, not {}", K_REPLICA_TASK_WORDS,
                       words.size());

            ReplicaTask task;
            task.epoch = words[0];
            task.batch_index = words[1];
            task.replica_index = words[2];
            task.replica_count = words[3];
            task.run_index = words[4];
            task.seed = words[5];

            SA_FAIL_IF(task.replica_count == 0, StatusCode::InvalidDescription,
                       "A replica node names a step of 0 replicas");
            SA_FAIL_IF(task.replica_index >= task.replica_count, StatusCode::InvalidDescription,
                       "A replica node names replica {} of a step of {}", task.replica_index,
                       task.replica_count);
            return task;
        }

    } // namespace Distribution
} // namespace SushiAI
