/**************************************************************************/
/* replica_plan.cpp                                                       */
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
 * @file replica_plan.cpp
 * @brief Implements ReplicaPlan.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#include <SushiAI/distribution/replica_plan.hpp>

#include <limits>
#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        Result<ReplicaPlan> ReplicaPlan::make(std::vector<ReplicaShare> shares)
        {
            SA_FAIL_IF(shares.empty(), StatusCode::InvalidDescription,
                       "A plan needs at least one replica; it was given none");

            std::int64_t total = 0;
            for (std::size_t replica = 0; replica < shares.size(); ++replica)
            {
                const ReplicaShare& share = shares[replica];
                SA_FAIL_IF(share.rows < 0, StatusCode::InvalidDescription,
                           "replica {} 'rows' is {}; a replica takes zero rows or more", replica,
                           share.rows);
                SA_FAIL_IF(share.micro_batch < 1, StatusCode::InvalidDescription,
                           "replica {} 'micro_batch' is {}; a micro-batch holds at least one row",
                           replica, share.micro_batch);
                SA_FAIL_IF(share.rows % share.micro_batch != 0, StatusCode::InvalidDescription,
                           "replica {} 'micro_batch' {} does not divide its 'rows' {}", replica,
                           share.micro_batch, share.rows);
                SA_FAIL_IF(share.rows > std::numeric_limits<std::int64_t>::max() - total,
                           StatusCode::InvalidDescription,
                           "the 'rows' of the replicas sum to more than {}, at replica {}",
                           std::numeric_limits<std::int64_t>::max(), replica);
                total += share.rows;
            }

            SA_FAIL_IF(total == 0, StatusCode::InvalidDescription,
                       "the 'rows' of the {} replicas sum to 0; the weight r_k / B has no batch "
                       "to divide",
                       shares.size());

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return ReplicaPlan(std::move(shares), total);
            });
        }

        ReplicaPlan::ReplicaPlan(std::vector<ReplicaShare> shares, std::int64_t batch_rows)
            : shares_(std::move(shares)), batch_rows_(batch_rows)
        {
            offsets_.reserve(shares_.size());
            std::int64_t before = 0;
            for (const ReplicaShare& share : shares_)
            {
                offsets_.push_back(before);
                before += share.rows;
            }
        }

        std::size_t ReplicaPlan::replica_count() const noexcept
        {
            return shares_.size();
        }

        std::int64_t ReplicaPlan::batch_rows() const noexcept
        {
            return batch_rows_;
        }

        std::int64_t ReplicaPlan::rows_of(std::size_t replica) const
        {
            SA_FATAL_IF(replica >= shares_.size(), "Replica {} was asked of a plan that has {}",
                        replica, shares_.size());
            return shares_[replica].rows;
        }

        std::int64_t ReplicaPlan::micro_batch_of(std::size_t replica) const
        {
            SA_FATAL_IF(replica >= shares_.size(), "Replica {} was asked of a plan that has {}",
                        replica, shares_.size());
            return shares_[replica].micro_batch;
        }

        std::int64_t ReplicaPlan::micro_batch_count_of(std::size_t replica) const
        {
            return rows_of(replica) / micro_batch_of(replica);
        }

        double ReplicaPlan::weight_of(std::size_t replica) const
        {
            return static_cast<double>(rows_of(replica)) / static_cast<double>(batch_rows_);
        }

        RowRange ReplicaPlan::rows(std::size_t replica, std::uint64_t batch_index) const
        {
            const std::uint64_t count = static_cast<std::uint64_t>(rows_of(replica));
            const std::uint64_t batch = static_cast<std::uint64_t>(batch_rows_);
            SA_FATAL_IF(batch_index >= std::numeric_limits<std::uint64_t>::max() / batch,
                        "Batch {} of {} rows ends past the 64-bit row positions", batch_index,
                        batch);

            RowRange range;
            range.first = batch_index * batch + static_cast<std::uint64_t>(offsets_[replica]);
            range.count = count;
            return range;
        }

        RowRange ReplicaPlan::micro_batch(std::size_t replica, std::uint64_t batch_index,
                                          std::uint64_t micro_batch_index) const
        {
            const std::uint64_t cuts = static_cast<std::uint64_t>(micro_batch_count_of(replica));
            SA_FATAL_IF(micro_batch_index >= cuts,
                        "Micro-batch {} was asked of replica {}, which runs {}",
                        micro_batch_index, replica, cuts);

            const std::uint64_t size = static_cast<std::uint64_t>(micro_batch_of(replica));
            RowRange range = rows(replica, batch_index);
            range.first += micro_batch_index * size;
            range.count = size;
            return range;
        }

    } // namespace Distribution
} // namespace SushiAI
