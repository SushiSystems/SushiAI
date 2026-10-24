/**************************************************************************/
/* replica_position.cpp                                                   */
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
 * @file replica_position.cpp
 * @brief Implements ReplicaPosition: the position of a step and its read from a pack.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/replica_position.hpp>

#include <cstring>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        ReplicaPosition ReplicaPosition::at_step(std::uint64_t step,
                                                 std::uint64_t batches_per_epoch)
        {
            SA_FATAL_IF(batches_per_epoch == 0,
                        "ReplicaPosition::at_step: an epoch holds at least one batch");

            ReplicaPosition position;
            position.epoch = step / batches_per_epoch;
            position.batch_index = step % batches_per_epoch;
            position.step = step;
            return position;
        }

        std::uint64_t ReplicaPosition::draw_index(std::uint64_t micro_batches,
                                                  std::uint64_t micro_batch) const
        {
            SA_FATAL_IF(micro_batch >= micro_batches,
                        "ReplicaPosition::draw_index: micro-batch {} of a step of {}",
                        micro_batch, micro_batches);

            return step * micro_batches + micro_batch;
        }

        Result<ReplicaPosition> ReplicaPosition::read(
            const SushiRuntime::Distributed::BufferView& view)
        {
            SA_FAIL_IF(view.data == nullptr, StatusCode::InvalidDescription,
                       "A replica node's position pack names no memory");
            SA_FAIL_IF(view.size_bytes < K_REPLICA_POSITION_BYTES, StatusCode::InvalidDescription,
                       "A replica node's position pack holds {} bytes; a position takes {}",
                       view.size_bytes, K_REPLICA_POSITION_BYTES);

            // Every machine of a cluster is little-endian; pack_layout.cpp refuses any other.
            std::uint64_t words[3] = {0, 0, 0};
            std::memcpy(words, view.data, K_REPLICA_POSITION_BYTES);

            ReplicaPosition position;
            position.epoch = words[0];
            position.batch_index = words[1];
            position.step = words[2];
            return position;
        }

    } // namespace Distribution
} // namespace SushiAI
