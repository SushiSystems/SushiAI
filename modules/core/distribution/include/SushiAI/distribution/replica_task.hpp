/**************************************************************************/
/* replica_task.hpp                                                       */
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
 * @file replica_task.hpp
 * @brief Declares ReplicaTask, the parameters a replica node carries.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief Holds how many parameters a replica node carries. */
        inline constexpr std::size_t K_REPLICA_TASK_WORDS = 4;

        /**
         * @brief Names the replica a node runs, as the node's parameters carry it.
         *
         * The words are, in order: replica index, replica count, run index, seed. The epoch
         * and the batch come from the node's position pack, since a replay keeps the words.
         */
        struct ReplicaTask
        {
            /** @brief The replica that runs the task, k. */
            std::uint64_t replica_index = 0;

            /** @brief How many replicas the step has. */
            std::uint64_t replica_count = 1;

            /** @brief The index of the run. */
            std::uint64_t run_index = 0;

            /** @brief The run's seed. */
            std::uint64_t seed = 0;

            /** @brief Returns the words a node carries, in the order of the struct. */
            [[nodiscard]] std::array<std::uint64_t, K_REPLICA_TASK_WORDS> encode() const noexcept;

            /**
             * @brief Reads a task back from the words of a node.
             * @return The task, or InvalidDescription for a count other than
             *         K_REPLICA_TASK_WORDS, a replica count of 0, or a replica index at or
             *         above the replica count.
             */
            [[nodiscard]] static Result<ReplicaTask> decode(
                SushiRuntime::span<const std::uint64_t> words) noexcept;

            /** @brief Returns the seed, the replica and the run of the task. */
            [[nodiscard]] ReplicaDraws draws() const noexcept
            {
                return ReplicaDraws{seed, replica_index, run_index};
            }
        };

    } // namespace Distribution
} // namespace SushiAI
