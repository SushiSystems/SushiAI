/**************************************************************************/
/* test_cluster_section.cpp                                               */
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
 * @file test_cluster_section.cpp
 * @brief Tests that ClusterSection reads the cluster section and turns it into placement rules.
 * @author Mustafa Garip
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiRuntime/core/placement_class.hpp>
#include <SushiRuntime/placement/placement_rule.hpp>
#include <SushiRuntime/placement/placement_rules.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/tolerance.hpp>

namespace
{
    /**
     * @brief Bounds the absolute error of a double result; 16 unit roundoffs of double, the
     * smallest power of two that covers 1e-15.
     */
    constexpr double K_DOUBLE_TIGHT_TOLERANCE = 16.0 * TestSupport::unit_roundoff<double>();

    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Distribution::ClusterSection;
    using SushiAI::Distribution::K_REPLICA_PLACEMENT_CLASS;
    using SushiAI::Distribution::ReplicaPlan;
    using SushiAI::Distribution::RowPositions;

    /** @brief Holds the replicas of the main case: five machines, the first one the master. */
    constexpr std::string_view K_FIVE_REPLICAS = R"([
        {"worker": "atlas", "rows": 12, "micro_batch": 4, "if_absent": "local"},
        {"worker": "bravo", "rows": 8, "micro_batch": 4, "priority": 5,
         "if_absent": "wait", "wait_ms": 30000},
        {"worker": "cobalt", "rows": 8, "micro_batch": 2, "if_absent": "fail"},
        {"worker": "delta", "rows": 6, "micro_batch": 3, "if_absent": "local"},
        {"worker": "ember", "rows": 6, "micro_batch": 2, "if_absent": "fail"}
    ])";

    /** @brief Holds one replica of the main case and everything the section makes of it. */
    struct ReplicaRow
    {
        /** @brief Names the replica in the test report; also its worker. */
        const char* worker;

        /** @brief Holds the replica's index in the section. */
        std::size_t replica;

        /** @brief Holds the rows the replica takes of each batch. */
        std::int64_t rows;

        /** @brief Holds the rows of one micro-batch. */
        std::int64_t micro_batch;

        /** @brief Holds the first row position of the replica in batch 2. */
        std::uint64_t first_in_batch_two;

        /** @brief Holds the replica's rows over the 40 rows of the batch. */
        double weight;

        /** @brief Holds the priority of the rule. */
        int priority;

        /** @brief Holds what the rule does when the worker is absent. */
        SushiRuntime::AbsentAction action;

        /** @brief Holds the bound of the absent action, in milliseconds. */
        std::int64_t bound_ms;
    };

    /** @brief Lists the five replicas of K_FIVE_REPLICAS with their expected results. */
    std::vector<ReplicaRow> replica_rows()
    {
        using SushiRuntime::AbsentAction;
        return {
            {"atlas", 0, 12, 4, 80, 0.3, 0, AbsentAction::Local, 0},
            {"bravo", 1, 8, 4, 92, 0.2, 5, AbsentAction::Wait, 30000},
            {"cobalt", 2, 8, 2, 100, 0.2, 0, AbsentAction::Fail, 0},
            {"delta", 3, 6, 3, 108, 0.15, 0, AbsentAction::Local, 0},
            {"ember", 4, 6, 2, 114, 0.15, 0, AbsentAction::Fail, 0},
        };
    }

    /** @brief Names a replica row after its worker. */
    std::string replica_row_name(const ::testing::TestParamInfo<ReplicaRow>& info)
    {
        return info.param.worker;
    }

    /** @brief Returns the text of a section that states a listen address and a secret file. */
    std::string section_text(std::string_view replicas)
    {
        return R"({"listen": "0.0.0.0:7700", "secret_file": "cluster.secret", "replicas": )" +
               std::string(replicas) + "}";
    }

    /** @brief Reads @p text as a cluster section, with the base directory "conf". */
    SushiAI::Result<ClusterSection> read_text(const std::string& text)
    {
        const Value section =
            SushiAI::Test::require_value(SushiAI::Config::parse_json(text, "<test>"));
        return ClusterSection::read(section, std::filesystem::path("conf"));
    }

    /** @brief Reads a section whose replica list is @p replicas. */
    SushiAI::Result<ClusterSection> read_replicas(std::string_view replicas)
    {
        return read_text(section_text(replicas));
    }

    /** @brief Holds a section the reader refuses and the words its message must name. */
    struct RefusalCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Holds the text of the section. */
        std::string text;

        /** @brief Holds the substrings the message must contain. */
        std::vector<std::string_view> needles;
    };

    /** @brief Builds the text of a section whose only replica is @p replica. */
    std::string one_replica_text(std::string_view replica)
    {
        return section_text("[" + std::string(replica) + "]");
    }

    /** @brief Lists every section the reader refuses, with what the message names. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {
            {"AWorkerNamedTwice",
             section_text(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "local"},
        {"worker": "b", "rows": 4, "micro_batch": 2, "if_absent": "local"},
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "local"}])"),
             {"entry 2", "'worker'", "'a'", "entry 0"}},
            {"RegressionTwoNamesOfOneMachine",
             section_text(R"([
        {"worker": "Rack-7", "rows": 4, "micro_batch": 2, "if_absent": "local"},
        {"worker": "rack-7.lab", "rows": 4, "micro_batch": 2, "if_absent": "local"}])"),
             {"entry 1", "'worker'", "entry 0"}},
            {"AMissingListenAddress",
             R"({"secret_file": "s", "replicas": [{"worker": "a", "rows": 4,
                      "micro_batch": 2, "if_absent": "local"}]})",
             {"cluster", "has no 'listen'"}},
            {"AMissingSecretFile",
             R"({"listen": "a:1", "replicas": [{"worker": "a", "rows": 4,
                      "micro_batch": 2, "if_absent": "local"}]})",
             {"cluster", "has no 'secret_file'"}},
            {"AnEmptyListenAddress",
             R"({"listen": "", "secret_file": "s", "replicas": []})",
             {"cluster 'listen'", "empty"}},
            {"AnEmptySecretFile",
             R"({"listen": "a:1", "secret_file": "", "replicas": []})",
             {"cluster 'secret_file'", "empty"}},
            {"ASectionWithoutReplicas",
             R"({"listen": "a:1", "secret_file": "s"})",
             {"cluster", "has no 'replicas'"}},
            {"AnIfAbsentOutsideTheThree",
             one_replica_text(
                 R"({"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "retry"})"),
             {"entry 0 'if_absent'", "'retry'", "'local'", "'wait'", "'fail'"}},
            {"AMissingIfAbsent",
             one_replica_text(R"({"worker": "a", "rows": 4, "micro_batch": 2})"),
             {"entry 0", "has no 'if_absent'"}},
            {"AWaitWithoutABound",
             one_replica_text(
                 R"({"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "wait"})"),
             {"entry 0 'if_absent'", "'wait'", "'wait_ms'"}},
            {"AWaitBoundOfZero",
             one_replica_text(R"({"worker": "a", "rows": 4, "micro_batch": 2,
                 "if_absent": "wait", "wait_ms": 0})"),
             {"entry 0 'wait_ms'", "is 0", "positive"}},
            {"AWaitBoundBelowZero",
             one_replica_text(R"({"worker": "a", "rows": 4, "micro_batch": 2,
                 "if_absent": "wait", "wait_ms": -5})"),
             {"entry 0 'wait_ms'", "is -5", "positive"}},
            {"ABoundBesideLocal",
             one_replica_text(R"({"worker": "a", "rows": 4, "micro_batch": 2,
                 "if_absent": "local", "wait_ms": 10})"),
             {"entry 0 'wait_ms'", "'local'", "only 'wait'"}},
            {"ABoundBesideFail",
             one_replica_text(R"({"worker": "a", "rows": 4, "micro_batch": 2,
                 "if_absent": "fail", "wait_ms": 10})"),
             {"entry 0 'wait_ms'", "'fail'", "only 'wait'"}},
            {"AReplicaWithoutAWorker",
             one_replica_text(R"({"rows": 4, "micro_batch": 2, "if_absent": "local"})"),
             {"entry 0", "has no 'worker'"}},
            {"AReplicaWithoutRows",
             one_replica_text(R"({"worker": "a", "micro_batch": 2, "if_absent": "local"})"),
             {"entry 0", "has no 'rows'"}},
            {"AReplicaWithoutAMicroBatch",
             one_replica_text(R"({"worker": "a", "rows": 4, "if_absent": "local"})"),
             {"entry 0", "has no 'micro_batch'"}},
            {"RowsOfTheWrongKind",
             one_replica_text(
                 R"({"worker": "a", "rows": "4", "micro_batch": 2, "if_absent": "local"})"),
             {"entry 0 'rows'", "whole number"}},
            {"AWorkerOfTheWrongKind",
             one_replica_text(
                 R"({"worker": 7, "rows": 4, "micro_batch": 2, "if_absent": "local"})"),
             {"entry 0 'worker'", "string"}},
            {"AnEmptyWorker",
             one_replica_text(
                 R"({"worker": "", "rows": 4, "micro_batch": 2, "if_absent": "local"})"),
             {"entry 0 'worker'", "empty"}},
            {"AnUnknownReplicaKey",
             one_replica_text(R"({"worker": "a", "rows": 4, "micro_batch": 2,
                 "if_absent": "local", "speed": 3})"),
             {"entry 0", "'speed'", "does not have"}},
            {"AnUnknownSectionKey",
             R"({"listen": "a:1", "secret_file": "s", "replicas": [], "port": 7})",
             {"cluster", "'port'", "does not have"}},
            {"APriorityOutsideTheRangeOfAnInt",
             one_replica_text(R"({"worker": "a", "rows": 4, "micro_batch": 2,
                 "if_absent": "local", "priority": 3000000000})"),
             {"entry 0 'priority'", "3000000000"}},
            {"ASectionThatIsNoObject", "[]", {"cluster", "must be an object"}},
        };
    }

    /** @brief Expects the reading of a section to be refused with InvalidDescription naming. */
    void expect_refusal(const SushiAI::Result<ClusterSection>& outcome,
                        const std::vector<std::string_view>& needles)
    {
        ASSERT_FALSE(outcome.has_value());
        EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
        for (std::string_view needle : needles)
        {
            EXPECT_NE(outcome.status().message().find(needle), std::string_view::npos)
                << "the diagnostic does not name \"" << needle << "\": "
                << outcome.status().message();
        }
    }

    /** @brief Returns an absolute path that exists on no machine, spelt for the platform. */
    std::filesystem::path fixed_absolute_secret_file()
    {
#if defined(_WIN32)
        return std::filesystem::path("C:/elsewhere/cluster.secret");
#else
        return std::filesystem::path("/elsewhere/cluster.secret");
#endif
    }

    /** @brief Runs each replica of the main case. */
    class Small_ClusterSectionReplica : public ::testing::TestWithParam<ReplicaRow>
    {

    };

    /** @brief Runs each section the reader refuses. */
    class Small_ClusterSectionRefusal : public ::testing::TestWithParam<RefusalCase>
    {

    };
} // namespace

/** @brief Tests that a section keeps its listen address. */
TEST(Small_ClusterSection, TheSectionKeepsItsListenAddress)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    EXPECT_EQ(section.listen_address(), "0.0.0.0:7700");
}

/** @brief Tests that a relative secret file is read from under the base directory. */
TEST(Small_ClusterSection, ARelativeSecretFileSitsUnderTheBaseDirectory)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    EXPECT_EQ(section.secret_file(), std::filesystem::path("conf") / "cluster.secret");
}

/** @brief Tests that a five-replica section makes a plan of five replicas. */
TEST(Small_ClusterSection, FiveReplicasMakeAPlanOfFiveReplicas)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    EXPECT_EQ(section.plan().replica_count(), 5u);
}

/** @brief Tests that the batch of the plan holds the sum of the replicas' rows. */
TEST(Small_ClusterSection, TheBatchHoldsTheSumOfTheReplicaRows)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    EXPECT_EQ(section.plan().batch_rows(), 40);
}

/** @brief Tests that a replica takes the rows and the micro-batch its entry states. */
TEST_P(Small_ClusterSectionReplica, AReplicaTakesTheRowsAndTheMicroBatchItStates)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    EXPECT_EQ(section.plan().rows_of(row.replica), row.rows);
    EXPECT_EQ(section.plan().micro_batch_of(row.replica), row.micro_batch);
}

/** @brief Tests that a replica's weight is its rows over the rows of the batch. */
TEST_P(Small_ClusterSectionReplica, AReplicaWeighsItsRowsOverTheBatch)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    EXPECT_NEAR(section.plan().weight_of(row.replica), row.weight, K_DOUBLE_TIGHT_TOLERANCE);
}

/** @brief Tests that a replica's range in batch 2 starts where the earlier replicas end. */
TEST_P(Small_ClusterSectionReplica, AReplicaStartsItsRangeAfterTheEarlierReplicas)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    const RowPositions range = section.plan().rows(row.replica, 2);

    EXPECT_EQ(range.first, row.first_in_batch_two);
    EXPECT_EQ(range.count, static_cast<std::uint64_t>(row.rows));
}

/** @brief Tests that a replica becomes one rule named, keyed, placed and bounded as stated. */
TEST_P(Small_ClusterSectionReplica, AReplicaBecomesOneRule)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::PlacementRules rules, section.placement_rules());
    ASSERT_EQ(rules.size(), 5u);

    const SushiRuntime::PlacementRule& rule = rules.rule(row.replica);

    EXPECT_EQ(rule.name, "replica " + std::to_string(row.replica));
    EXPECT_EQ(rule.match.placement_class, K_REPLICA_PLACEMENT_CLASS);
    ASSERT_TRUE(rule.match.key.has_value());
    EXPECT_EQ(*rule.match.key, row.replica);
    EXPECT_TRUE(rule.match.operation.empty());
    EXPECT_EQ(rule.target.machine_name, row.worker);
    EXPECT_FALSE(rule.target.machine_label.has_value());
    EXPECT_FALSE(rule.target.device_index.has_value());
    EXPECT_FALSE(rule.target.device_kind.has_value());
    EXPECT_FALSE(rule.target.device_min_memory.has_value());
    EXPECT_EQ(rule.priority, row.priority);
    EXPECT_EQ(rule.if_absent.action, row.action);
    EXPECT_EQ(rule.if_absent.bound, std::chrono::milliseconds(row.bound_ms));
    EXPECT_FALSE(rule.deadline.has_value());
}

/** @brief Tests that the rules resolve the node of a replica to the rule of that replica. */
TEST_P(Small_ClusterSectionReplica, TheRulesResolveAReplicaNodeToItsOwnRule)
{
    const ReplicaRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::PlacementRules rules, section.placement_rules());
    const SushiRuntime::PlacementClass replica_class(K_REPLICA_PLACEMENT_CLASS.data(),
                                                     K_REPLICA_PLACEMENT_CLASS.size());

    const std::optional<std::size_t> winner =
        rules.resolve(replica_class, row.replica, SushiRuntime::OperationID());

    ASSERT_TRUE(winner.has_value());
    EXPECT_EQ(*winner, row.replica);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ClusterSectionReplica, ::testing::ValuesIn(replica_rows()),
                         replica_row_name);

/** @brief Tests that the rules resolve a replica past the last to no rule. */
TEST(Small_ClusterSection, TheRulesResolveAReplicaPastTheLastToNoRule)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::PlacementRules rules, section.placement_rules());
    const SushiRuntime::PlacementClass replica_class(K_REPLICA_PLACEMENT_CLASS.data(),
                                                     K_REPLICA_PLACEMENT_CLASS.size());

    EXPECT_FALSE(rules.resolve(replica_class, 5, SushiRuntime::OperationID()).has_value());
}

/** @brief Tests that the rules resolve a node of no placement class to no rule. */
TEST(Small_ClusterSection, TheRulesResolveANodeOfNoClassToNoRule)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::PlacementRules rules, section.placement_rules());

    EXPECT_FALSE(
        rules.resolve(SushiRuntime::PlacementClass(), 0, SushiRuntime::OperationID()).has_value());
}

/** @brief Tests that an absolute secret file keeps its path. */
TEST(Small_ClusterSection, AnAbsoluteSecretFileIsKept)
{
    const std::filesystem::path absolute = fixed_absolute_secret_file();
    const std::string text = R"({"listen": "a:1", "secret_file": ")" +
                             absolute.generic_string() +
                             R"(", "replicas": [{"worker": "w", "rows": 4, "micro_batch": 2,
                                "if_absent": "fail"}]})";
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_text(text));

    EXPECT_EQ(section.secret_file().generic_string(), absolute.generic_string());
}

/** @brief Tests that a section that breaks a rule of the reader is refused naming the rule. */
TEST_P(Small_ClusterSectionRefusal, ASectionThatBreaksARuleIsRefusedNamingIt)
{
    const RefusalCase& row = GetParam();

    expect_refusal(read_text(row.text), row.needles);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ClusterSectionRefusal, ::testing::ValuesIn(refusal_cases()),
                         [](const ::testing::TestParamInfo<RefusalCase>& info)
                         {
                             return std::string(info.param.name);
                         });
