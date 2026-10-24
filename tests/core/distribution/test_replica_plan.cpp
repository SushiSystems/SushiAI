/**************************************************************************/
/* test_replica_plan.cpp                                                  */
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
 * @file test_replica_plan.cpp
 * @brief Tests that ReplicaPlan splits a batch into rows, micro-batches and weights.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Distribution::ReplicaPlan;
    using SushiAI::Distribution::ReplicaShare;
    using SushiAI::Distribution::RowRange;

    /** @brief Holds the rows of the five replicas of the main case; they sum to 40. */
    const std::vector<std::int64_t> K_ROWS = {12, 8, 8, 6, 6};

    /** @brief Holds the micro-batch of each of the five replicas; each divides its rows. */
    const std::vector<std::int64_t> K_MICRO_BATCHES = {4, 4, 2, 3, 2};

    /** @brief Builds the shares of five replicas from K_ROWS and K_MICRO_BATCHES. */
    std::vector<ReplicaShare> five_shares()
    {
        std::vector<ReplicaShare> shares;
        for (std::size_t replica = 0; replica < K_ROWS.size(); ++replica)
            shares.push_back({K_ROWS[replica], K_MICRO_BATCHES[replica]});
        return shares;
    }
} // namespace

/** @brief Tests that five replicas take contiguous rows in replica order from batch 0. */
TEST(Small_ReplicaPlan, FiveReplicasTakeContiguousRowsInReplicaOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    ASSERT_EQ(plan.replica_count(), 5u);
    EXPECT_EQ(plan.batch_rows(), 40);

    const std::uint64_t firsts[] = {0, 12, 20, 28, 34};
    for (std::size_t replica = 0; replica < 5; ++replica)
    {
        const RowRange range = plan.rows(replica, 0);
        EXPECT_EQ(range.first, firsts[replica]) << "replica " << replica;
        EXPECT_EQ(range.count, static_cast<std::uint64_t>(K_ROWS[replica]))
            << "replica " << replica;
    }
}

/** @brief Tests that a later batch starts one batch of rows further on. */
TEST(Small_ReplicaPlan, ALaterBatchShiftsEveryReplicaByWholeBatches)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    for (std::size_t replica = 0; replica < 5; ++replica)
    {
        const RowRange first_batch = plan.rows(replica, 0);
        const RowRange third_batch = plan.rows(replica, 2);
        EXPECT_EQ(third_batch.first, first_batch.first + 2 * 40u) << "replica " << replica;
        EXPECT_EQ(third_batch.count, first_batch.count) << "replica " << replica;
    }
}

/** @brief Tests that the ranges of one batch tile it with no gap and no overlap. */
TEST(Small_ReplicaPlan, TheRangesOfOneBatchTileIt)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    constexpr std::uint64_t K_BATCH = 7;
    std::uint64_t next = K_BATCH * 40;
    for (std::size_t replica = 0; replica < 5; ++replica)
    {
        const RowRange range = plan.rows(replica, K_BATCH);
        EXPECT_EQ(range.first, next) << "replica " << replica;
        next = range.first + range.count;
    }

    EXPECT_EQ(next, (K_BATCH + 1) * 40);
}

/** @brief Tests that the micro-batches of a replica cut its rows into equal consecutive runs. */
TEST(Small_ReplicaPlan, MicroBatchesCutTheRowsOfAReplica)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_EQ(plan.micro_batch_of(0), 4);
    ASSERT_EQ(plan.micro_batch_count_of(0), 3);

    const RowRange rows = plan.rows(0, 1);
    for (std::uint64_t cut = 0; cut < 3; ++cut)
    {
        const RowRange micro = plan.micro_batch(0, 1, cut);
        EXPECT_EQ(micro.first, rows.first + cut * 4) << "micro-batch " << cut;
        EXPECT_EQ(micro.count, 4u) << "micro-batch " << cut;
    }

    EXPECT_EQ(plan.micro_batch_count_of(1), 2);
    EXPECT_EQ(plan.micro_batch_count_of(2), 4);
    EXPECT_EQ(plan.micro_batch_count_of(3), 2);
    EXPECT_EQ(plan.micro_batch_count_of(4), 3);
}

/** @brief Tests that a replica's weight is its rows over the rows of the batch. */
TEST(Small_ReplicaPlan, TheWeightOfAReplicaIsItsRowsOverTheBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    const double wanted[] = {0.3, 0.2, 0.2, 0.15, 0.15};
    double sum = 0.0;
    for (std::size_t replica = 0; replica < 5; ++replica)
    {
        EXPECT_NEAR(plan.weight_of(replica), wanted[replica], 1e-15) << "replica " << replica;
        sum += plan.weight_of(replica);
    }

    EXPECT_NEAR(sum, 1.0, 1e-12);
}

/** @brief Tests that a replica of zero rows takes none and leaves the others contiguous. */
TEST(Small_ReplicaPlan, AReplicaOfZeroRowsSitsOut)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan,
                            ReplicaPlan::make({{0, 1}, {6, 3}, {2, 2}}));

    EXPECT_EQ(plan.batch_rows(), 8);
    EXPECT_EQ(plan.rows(0, 3).count, 0u);
    EXPECT_EQ(plan.micro_batch_count_of(0), 0);
    EXPECT_NEAR(plan.weight_of(0), 0.0, 1e-15);
    EXPECT_EQ(plan.rows(1, 3).first, 24u);
    EXPECT_EQ(plan.rows(2, 3).first, 30u);
}

/** @brief Tests that a single replica takes the whole batch with weight one. */
TEST(Small_ReplicaPlan, ASingleReplicaTakesTheWholeBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make({{16, 8}}));

    EXPECT_NEAR(plan.weight_of(0), 1.0, 1e-15);
    EXPECT_EQ(plan.rows(0, 5).first, 80u);
    EXPECT_EQ(plan.rows(0, 5).count, 16u);
}

/** @brief Tests that no replica is refused. */
TEST(Small_ReplicaPlan, NoReplicaIsRefused)
{
    SA_EXPECT_FAILURE(ReplicaPlan::make({}), StatusCode::InvalidDescription,
                      "at least one replica");
}

/** @brief Tests that a micro-batch that does not divide its rows is refused by replica and key. */
TEST(Small_ReplicaPlan, AMicroBatchThatDoesNotDivideItsRowsIsRefused)
{
    SA_EXPECT_FAILURE(ReplicaPlan::make({{8, 4}, {10, 4}, {8, 4}}),
                      StatusCode::InvalidDescription, "replica 1", "'micro_batch' 4",
                      "does not divide", "'rows' 10");
}

/** @brief Tests that a micro-batch below one is refused. */
TEST(Small_ReplicaPlan, AMicroBatchBelowOneIsRefused)
{
    SA_EXPECT_FAILURE(ReplicaPlan::make({{8, 4}, {8, 0}}), StatusCode::InvalidDescription,
                      "replica 1", "'micro_batch' is 0");
}

/** @brief Tests that a negative row count is refused. */
TEST(Small_ReplicaPlan, ANegativeRowCountIsRefused)
{
    SA_EXPECT_FAILURE(ReplicaPlan::make({{8, 4}, {-4, 4}}), StatusCode::InvalidDescription,
                      "replica 1", "'rows' is -4");
}

/** @brief Tests that rows summing to zero are refused, since no weight can be formed. */
TEST(Small_ReplicaPlan, RowsThatSumToZeroAreRefused)
{
    SA_EXPECT_FAILURE(ReplicaPlan::make({{0, 1}, {0, 2}}), StatusCode::InvalidDescription,
                      "'rows'", "sum to 0", "weight");
}

/** @brief Tests that rows summing past the largest int64 are refused. */
TEST(Small_ReplicaPlan, RowsThatSumPastInt64AreRefused)
{
    const std::int64_t most = std::numeric_limits<std::int64_t>::max();
    SA_EXPECT_FAILURE(ReplicaPlan::make({{most, 1}, {1, 1}}), StatusCode::InvalidDescription,
                      "sum to more than", "replica 1");
}

/** @brief Tests that a replica past the last stops the process. */
TEST(Medium_ReplicaPlanPreconditions, AReplicaPastTheLastIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    SA_EXPECT_FATAL(static_cast<void>(plan.rows(5, 0)), "Replica 5", "plan that has 5");
}

/** @brief Tests that a micro-batch past the last of a replica stops the process. */
TEST(Medium_ReplicaPlanPreconditions, AMicroBatchPastTheLastIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    SA_EXPECT_FATAL(static_cast<void>(plan.micro_batch(0, 0, 3)), "Micro-batch 3", "replica 0",
                    "runs 3");
}
