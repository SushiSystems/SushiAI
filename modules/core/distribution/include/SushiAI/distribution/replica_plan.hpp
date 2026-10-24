/**************************************************************************/
/* replica_plan.hpp                                                       */
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
 * @file replica_plan.hpp
 * @brief Declares ReplicaPlan, which says which rows of a batch each replica takes.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief The placement class of a replica task; its key is the replica's index. */
        inline constexpr std::string_view K_REPLICA_PLACEMENT_CLASS = "sushiai.replica";

        /** @brief States how many rows one replica takes of a batch and how it cuts them. */
        struct ReplicaShare
        {
            /** @brief Rows per step, r_k; zero lets the replica sit out. */
            std::int64_t rows = 0;

            /** @brief Rows per micro-batch, m_k; at least one, and it divides rows. */
            std::int64_t micro_batch = 1;
        };

        /** @brief A run of consecutive positions in the epoch's sample order. */
        struct RowPositions
        {
            /** @brief The position of the first row. */
            std::uint64_t first = 0;

            /** @brief How many rows follow, the first included. */
            std::uint64_t count = 0;
        };

        /**
         * @brief Splits every batch of a run into contiguous row ranges, one per replica.
         *
         * Replica k takes r_k rows of each batch of B = sum of r_k rows, in replica order, and
         * weighs r_k / B in the gradient sum. The plan reads nothing but its shares, so every
         * machine that holds them finds the same rows.
         */
        class ReplicaPlan final
        {
            public:
                /**
                 * @brief Checks @p shares and builds the plan they describe.
                 * @return The plan, or InvalidDescription naming the replica and the key for
                 *         no replica, a negative row count, a micro-batch below one, a
                 *         micro-batch that does not divide its rows, rows that sum above the
                 *         largest int64, or rows that sum to zero.
                 */
                [[nodiscard]] static Result<ReplicaPlan> make(std::vector<ReplicaShare> shares);

                /** @brief Returns how many replicas the plan has; at least one. */
                [[nodiscard]] std::size_t replica_count() const noexcept;

                /** @brief Returns B, the rows of one batch; above zero. */
                [[nodiscard]] std::int64_t batch_rows() const noexcept;

                /**
                 * @brief Returns r_k, the rows replica @p replica takes per batch.
                 * @pre @p replica is below replica_count().
                 */
                [[nodiscard]] std::int64_t rows_of(std::size_t replica) const;

                /**
                 * @brief Returns m_k, the rows per micro-batch of replica @p replica.
                 * @pre @p replica is below replica_count().
                 */
                [[nodiscard]] std::int64_t micro_batch_of(std::size_t replica) const;

                /**
                 * @brief Returns r_k / m_k, how many micro-batches replica @p replica runs.
                 * @pre @p replica is below replica_count().
                 */
                [[nodiscard]] std::int64_t micro_batch_count_of(std::size_t replica) const;

                /**
                 * @brief Returns r_k / B, the weight of replica @p replica in the gradient sum.
                 * @pre @p replica is below replica_count().
                 */
                [[nodiscard]] double weight_of(std::size_t replica) const;

                /**
                 * @brief Returns the rows replica @p replica takes of batch @p batch_index.
                 *
                 * The range starts at batch_index * B plus the rows of the replicas before it
                 * and holds r_k rows.
                 * @pre @p replica is below replica_count(), and the end of the batch,
                 *      (batch_index + 1) * B, fits in 64 bits.
                 */
                [[nodiscard]] RowPositions rows(std::size_t replica,
                                                std::uint64_t batch_index) const;

                /**
                 * @brief Returns micro-batch @p micro_batch_index of the rows of a replica.
                 * @pre The preconditions of rows(), and @p micro_batch_index is below
                 *      micro_batch_count_of(@p replica).
                 */
                [[nodiscard]] RowPositions micro_batch(std::size_t replica,
                                                       std::uint64_t batch_index,
                                                       std::uint64_t micro_batch_index) const;

            private:
                /** @brief Takes checked @p shares whose rows sum to @p batch_rows. */
                ReplicaPlan(std::vector<ReplicaShare> shares, std::int64_t batch_rows);

                /// @brief The shares, in replica order.
                std::vector<ReplicaShare> shares_;

                /// @brief Entry k holds the rows of the replicas before replica k.
                std::vector<std::int64_t> offsets_;

                /// @brief The rows of one batch.
                std::int64_t batch_rows_ = 0;
        };

    } // namespace Distribution
} // namespace SushiAI
