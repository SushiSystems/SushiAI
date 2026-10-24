/**************************************************************************/
/* replica_position.hpp                                                   */
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
 * @file replica_position.hpp
 * @brief Declares ReplicaPosition, the epoch and batch a replica node reads from a pack.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiRuntime/distributed/buffer_view.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief Holds the bytes of a position pack: two 64-bit words, little-endian. */
        inline constexpr std::size_t K_REPLICA_POSITION_BYTES = 2 * sizeof(std::uint64_t);

        /**
         * @brief Names the batch a replica node of a compiled step serves on one replay.
         *
         * A node's parameters are fixed when it is recorded, so a node replayed every step
         * reads its epoch and batch from a position pack the master rewrites before each replay.
         */
        struct ReplicaPosition
        {
            /** @brief The epoch, counted from 0. */
            std::uint64_t epoch = 0;

            /** @brief The batch of the epoch, counted from 0. */
            std::uint64_t batch_index = 0;

            /**
             * @brief Returns the position of the step counted @p step from the first.
             * @pre @p batches_per_epoch is at least 1.
             */
            [[nodiscard]] static ReplicaPosition at_step(std::uint64_t step,
                                                         std::uint64_t batches_per_epoch);

            /**
             * @brief Reads a position from the first K_REPLICA_POSITION_BYTES of @p view.
             * @return The position, or InvalidDescription for a view with no memory or one
             *         smaller than K_REPLICA_POSITION_BYTES.
             */
            [[nodiscard]] static Result<ReplicaPosition> read(
                const SushiRuntime::Distributed::BufferView& view);
        };

    } // namespace Distribution
} // namespace SushiAI
