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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
            return {replica_index, replica_count, run_index, seed};
        }

        Result<ReplicaTask> ReplicaTask::decode(SushiRuntime::span<const std::uint64_t> words) noexcept
        {
            SA_FAIL_IF(words.size() != K_REPLICA_TASK_WORDS, StatusCode::InvalidDescription,
                       "A replica node carries {} parameters, not {}", words.size(),
                       K_REPLICA_TASK_WORDS);

            ReplicaTask task;
            task.replica_index = words[0];
            task.replica_count = words[1];
            task.run_index = words[2];
            task.seed = words[3];

            SA_FAIL_IF(task.replica_count == 0, StatusCode::InvalidDescription,
                       "A replica node names a step of 0 replicas");
            SA_FAIL_IF(task.replica_index >= task.replica_count, StatusCode::InvalidDescription,
                       "A replica node names replica {} of a step of {}", task.replica_index,
                       task.replica_count);
            return task;
        }

    } // namespace Distribution
} // namespace SushiAI
