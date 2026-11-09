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
 * @file test_replica_plan.cpp
 * @brief Tests that ReplicaPlan splits a batch into rows, micro-batches and weights.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/tolerance.hpp>
#include <support/test_names.hpp>

namespace
{
    /**
     * @brief Bounds the absolute error of a double result; 16384 unit roundoffs of double, the
     * smallest power of two that covers 1e-12.
     */
    constexpr double K_DOUBLE_TOLERANCE = 16384.0 * TestSupport::unit_roundoff<double>();

    /**
     * @brief Bounds the absolute error of a double result; 16 unit roundoffs of double, the
     * smallest power of two that covers 1e-15.
     */
    constexpr double K_DOUBLE_TIGHT_TOLERANCE = 16.0 * TestSupport::unit_roundoff<double>();

    using SushiAI::StatusCode;
    using SushiAI::Distribution::ReplicaPlan;
    using SushiAI::Distribution::ReplicaShare;
    using SushiAI::Distribution::RowPositions;

    /** @brief Holds one replica of the main case and what the plan makes of it. */
    struct ReplicaRow
    {
        /** @brief Names the replica in the test report. */
        const char* name;

        /** @brief Holds the replica's index. */
        std::size_t replica;

        /** @brief Holds the rows the replica takes of each batch. */
        std::int64_t rows;

        /** @brief Holds the rows of one micro-batch; it divides the rows. */
        std::int64_t micro_batch;

        /** @brief Holds the micro-batches the rows cut into. */
        std::int64_t micro_batches;

        /** @brief Holds the first row position of the replica in batch 0. */
        std::uint64_t first;

        /** @brief Holds the replica's rows over the 40 rows of the batch. */
        double weight;
    };

    /** @brief Lists the five replicas of the main case; their rows sum to 40. */
    std::vector<ReplicaRow> replica_rows()
    {
        return {
            {"replica_0", 0, 12, 4, 3, 0, 0.3},
            {"replica_1", 1, 8, 4, 2, 12, 0.2},
            {"replica_2", 2, 8, 2, 4, 20, 0.2},
            {"replica_3", 3, 6, 3, 2, 28, 0.15},
            {"replica_4", 4, 6, 2, 3, 34, 0.15},
        };
    }

    /** @brief Names a replica row after its replica. */
    std::string replica_row_name(const ::testing::TestParamInfo<ReplicaRow>& info)
    {
        return info.param.name;
    }

    /** @brief Builds the shares of five replicas from replica_rows(). */
    std::vector<ReplicaShare> five_shares()
    {
        std::vector<ReplicaShare> shares;
        for (const ReplicaRow& row : replica_rows())
            shares.push_back({row.rows, row.micro_batch});
        return shares;
    }

    /** @brief Adds the weights of every replica of @p plan. */
    double sum_of_weights(const ReplicaPlan& plan)
    {
        double sum = 0.0;
        for (std::size_t replica = 0; replica < plan.replica_count(); ++replica)
            sum += plan.weight_of(replica);
        return sum;
    }

    /** @brief Holds shares the plan refuses and the words its message must name. */
    struct RefusalCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Holds the shares of the replicas. */
        std::vector<ReplicaShare> shares;

        /** @brief Holds the substrings the message must contain. */
        std::vector<std::string_view> needles;
    };

    /** @brief Lists every share list the plan refuses, with what the message names. */
    std::vector<RefusalCase> refusal_cases()
    {
        const std::int64_t most = std::numeric_limits<std::int64_t>::max();
        return {
            {"NoReplica", {}, {"at least one replica"}},
            {"AMicroBatchThatDoesNotDivideItsRows",
             {{8, 4}, {10, 4}, {8, 4}},
             {"replica 1", "'micro_batch' 4", "does not divide", "'rows' 10"}},
            {"AMicroBatchBelowOne", {{8, 4}, {8, 0}}, {"replica 1", "'micro_batch' is 0"}},
            {"ANegativeRowCount", {{8, 4}, {-4, 4}}, {"replica 1", "'rows' is -4"}},
            {"RowsThatSumToZero", {{0, 1}, {0, 2}}, {"'rows'", "sum to 0", "weight"}},
            {"RowsThatSumPastInt64", {{most, 1}, {1, 1}}, {"sum to more than", "replica 1"}},
        };
    }

    /** @brief Asks @p plan for the rows of the replica past its last. */
    void ask_for_a_replica_past_the_last(const ReplicaPlan& plan)
    {
        static_cast<void>(plan.rows(5, 0));
    }

    /** @brief Asks @p plan for the micro-batch past the last of replica 0. */
    void ask_for_a_micro_batch_past_the_last(const ReplicaPlan& plan)
    {
        static_cast<void>(plan.micro_batch(0, 0, 3));
    }

    /** @brief Holds a call that breaks a precondition of the plan and the report it ends with. */
    struct PreconditionCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Holds the call that breaks the precondition. */
        void (*call)(const ReplicaPlan&);

        /** @brief Holds the pattern the fatal report must match. */
        std::string pattern;
    };

    /** @brief Lists the calls that break a precondition of the plan. */
    std::vector<PreconditionCase> precondition_cases()
    {
        return {
            {"AReplicaPastTheLast", &ask_for_a_replica_past_the_last,
             SushiAI::Test::death_pattern({"Replica 5", "plan that has 5"})},
            {"AMicroBatchPastTheLast", &ask_for_a_micro_batch_past_the_last,
             SushiAI::Test::death_pattern({"Micro-batch 3", "replica 0", "runs 3"})},
        };
    }

    /** @brief Runs each replica of the main case. */
    class Small_ReplicaPlanReplica : public ::testing::TestWithParam<ReplicaRow>
    {

    };

    /** @brief Runs each micro-batch of replica 0. */
    class Small_ReplicaPlanMicroBatch : public ::testing::TestWithParam<std::uint64_t>
    {

    };

    /** @brief Runs each share list the plan refuses. */
    class Small_ReplicaPlanRefusal : public ::testing::TestWithParam<RefusalCase>
    {

    };

    /** @brief Runs each call that breaks a precondition of the plan. */
    class Medium_ReplicaPlanPreconditions : public ::testing::TestWithParam<PreconditionCase>
    {

    };
} // namespace

/** @brief Tests that five replicas make a plan of five replicas. */
TEST(Small_ReplicaPlan, FiveReplicasMakeAPlanOfFiveReplicas)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_EQ(plan.replica_count(), 5u);
}

/** @brief Tests that the batch of five replicas holds the sum of their rows. */
TEST(Small_ReplicaPlan, TheBatchHoldsTheSumOfTheReplicaRows)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_EQ(plan.batch_rows(), 40);
}

/** @brief Tests that a replica takes contiguous rows after the earlier replicas, from batch 0. */
TEST_P(Small_ReplicaPlanReplica, AReplicaTakesContiguousRowsInReplicaOrder)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    const RowPositions range = plan.rows(row.replica, 0);

    EXPECT_EQ(range.first, row.first);
    EXPECT_EQ(range.count, static_cast<std::uint64_t>(row.rows));
}

/** @brief Tests that a later batch starts a replica whole batches of rows further on. */
TEST_P(Small_ReplicaPlanReplica, ALaterBatchShiftsAReplicaByWholeBatches)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    const RowPositions first_batch = plan.rows(row.replica, 0);
    const RowPositions third_batch = plan.rows(row.replica, 2);

    EXPECT_EQ(third_batch.first, first_batch.first + 2 * 40u);
    EXPECT_EQ(third_batch.count, first_batch.count);
}

/** @brief Tests that a replica's range in batch 7 starts where the batch's earlier ranges end. */
TEST_P(Small_ReplicaPlanReplica, AReplicaStartsWhereTheEarlierRangesOfItsBatchEnd)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));
    constexpr std::uint64_t K_BATCH = 7;

    EXPECT_EQ(plan.rows(row.replica, K_BATCH).first, K_BATCH * 40 + row.first);
}

/** @brief Tests that a replica takes the micro-batch size its share states. */
TEST_P(Small_ReplicaPlanReplica, AReplicaTakesTheMicroBatchSizeItsShareStates)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_EQ(plan.micro_batch_of(row.replica), row.micro_batch);
}

/** @brief Tests that a replica's rows cut into as many micro-batches as its size divides them. */
TEST_P(Small_ReplicaPlanReplica, AReplicaCountsItsMicroBatches)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_EQ(plan.micro_batch_count_of(row.replica), row.micro_batches);
}

/** @brief Tests that a replica's weight is its rows over the rows of the batch. */
TEST_P(Small_ReplicaPlanReplica, TheWeightOfAReplicaIsItsRowsOverTheBatch)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_NEAR(plan.weight_of(row.replica), row.weight, K_DOUBLE_TIGHT_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ReplicaPlanReplica, ::testing::ValuesIn(replica_rows()),
                         replica_row_name);

/** @brief Tests that the last range of a batch ends where the next batch starts. */
TEST(Small_ReplicaPlan, TheLastRangeOfABatchEndsWhereTheNextBatchStarts)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));
    constexpr std::uint64_t K_BATCH = 7;

    const RowPositions last = plan.rows(4, K_BATCH);

    EXPECT_EQ(last.first + last.count, (K_BATCH + 1) * 40);
}

/** @brief Tests that the micro-batches of the first replica are consecutive runs of its rows. */
TEST_P(Small_ReplicaPlanMicroBatch, AMicroBatchStartsAfterTheEarlierOnesOfItsReplica)
{
    const std::uint64_t cut = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    const RowPositions rows = plan.rows(0, 1);
    const RowPositions micro = plan.micro_batch(0, 1, cut);

    EXPECT_EQ(micro.first, rows.first + cut * 4);
    EXPECT_EQ(micro.count, 4u);
}

INSTANTIATE_TEST_SUITE_P(Cuts, Small_ReplicaPlanMicroBatch, ::testing::Values(0u, 1u, 2u));

/** @brief Tests that the weights of the five replicas sum to one. */
TEST(Small_ReplicaPlan, TheWeightsOfAPlanSumToOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_NEAR(sum_of_weights(plan), 1.0, K_DOUBLE_TOLERANCE);
}

/** @brief Tests that a replica of zero rows takes no rows. */
TEST(Small_ReplicaPlan, AReplicaOfZeroRowsTakesNoRows)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan,
                            ReplicaPlan::make({{0, 1}, {6, 3}, {2, 2}}));

    EXPECT_EQ(plan.rows(0, 3).count, 0u);
}

/** @brief Tests that a replica of zero rows runs no micro-batch. */
TEST(Small_ReplicaPlan, AReplicaOfZeroRowsRunsNoMicroBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan,
                            ReplicaPlan::make({{0, 1}, {6, 3}, {2, 2}}));

    EXPECT_EQ(plan.micro_batch_count_of(0), 0);
}

/** @brief Tests that a replica of zero rows weighs exactly zero, since 0 over 8 is exact. */
TEST(Small_ReplicaPlan, AReplicaOfZeroRowsWeighsExactlyZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan,
                            ReplicaPlan::make({{0, 1}, {6, 3}, {2, 2}}));

    EXPECT_EQ(plan.weight_of(0), 0.0);
}

/** @brief Tests that the batch of a plan with a zero-row replica holds the rows of the others. */
TEST(Small_ReplicaPlan, ABatchWithAZeroRowReplicaHoldsTheRowsOfTheOthers)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan,
                            ReplicaPlan::make({{0, 1}, {6, 3}, {2, 2}}));

    EXPECT_EQ(plan.batch_rows(), 8);
}

/** @brief Tests that the replicas after a zero-row replica stay contiguous. */
TEST(Small_ReplicaPlan, TheReplicasAfterAZeroRowReplicaStayContiguous)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan,
                            ReplicaPlan::make({{0, 1}, {6, 3}, {2, 2}}));

    EXPECT_EQ(plan.rows(1, 3).first, 24u);
    EXPECT_EQ(plan.rows(2, 3).first, 30u);
}

/** @brief Tests that a single replica weighs exactly one, since 16 over 16 is exact. */
TEST(Small_ReplicaPlan, ASingleReplicaWeighsExactlyOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make({{16, 8}}));

    EXPECT_EQ(plan.weight_of(0), 1.0);
}

/** @brief Tests that a single replica takes the whole batch. */
TEST(Small_ReplicaPlan, ASingleReplicaTakesTheWholeBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make({{16, 8}}));

    EXPECT_EQ(plan.rows(0, 5).first, 80u);
    EXPECT_EQ(plan.rows(0, 5).count, 16u);
}

/** @brief Tests that a share list that breaks a rule of the plan is refused naming the rule. */
TEST_P(Small_ReplicaPlanRefusal, AShareListThatBreaksARuleIsRefusedNamingIt)
{
    const RefusalCase& row = GetParam();

    const SushiAI::Result<ReplicaPlan> outcome = ReplicaPlan::make(row.shares);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
    for (std::string_view needle : row.needles)
    {
        EXPECT_NE(outcome.status().message().find(needle), std::string_view::npos)
            << "the diagnostic does not name \"" << needle << "\": " << outcome.status().message();
    }
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ReplicaPlanRefusal, ::testing::ValuesIn(refusal_cases()),
                         [](const ::testing::TestParamInfo<RefusalCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });

/** @brief Tests that a call that breaks a precondition of the plan stops the process. */
TEST_P(Medium_ReplicaPlanPreconditions, ACallThatBreaksAPreconditionIsFatal)
{
    const PreconditionCase& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPlan plan, ReplicaPlan::make(five_shares()));

    EXPECT_DEATH(row.call(plan), row.pattern);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_ReplicaPlanPreconditions,
                         ::testing::ValuesIn(precondition_cases()),
                         [](const ::testing::TestParamInfo<PreconditionCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
