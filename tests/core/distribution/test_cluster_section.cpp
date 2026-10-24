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

namespace
{
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

    /** @brief Returns the text of a section that states a listen address and a secret file. */
    std::string section_text(std::string_view replicas)
    {
        return R"({"listen": "0.0.0.0:7700", "secret_file": "cluster.secret", "replicas": )" +
               std::string(replicas) + "}";
    }

    /** @brief Reads @p text as a cluster section, with the base directory "conf". */
    SushiAI::Result<ClusterSection> read_text(const std::string& text)
    {
        const Value section = SushiAI::Config::parse_json(text, "<test>");
        return ClusterSection::read(section, std::filesystem::path("conf"));
    }

    /** @brief Reads a section whose replica list is @p replicas. */
    SushiAI::Result<ClusterSection> read_replicas(std::string_view replicas)
    {
        return read_text(section_text(replicas));
    }
} // namespace

/** @brief Tests that a five-replica section turns into rows, micro-batches and weights. */
TEST(Small_ClusterSection, FiveReplicasBecomeRowsMicroBatchesAndWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));

    EXPECT_EQ(section.listen_address(), "0.0.0.0:7700");
    EXPECT_EQ(section.secret_file(), std::filesystem::path("conf") / "cluster.secret");

    const ReplicaPlan& plan = section.plan();
    ASSERT_EQ(plan.replica_count(), 5u);
    EXPECT_EQ(plan.batch_rows(), 40);

    const std::int64_t rows[] = {12, 8, 8, 6, 6};
    const std::int64_t micro_batches[] = {4, 4, 2, 3, 2};
    const std::uint64_t firsts[] = {80, 92, 100, 108, 114};
    const double weights[] = {0.3, 0.2, 0.2, 0.15, 0.15};
    for (std::size_t replica = 0; replica < 5; ++replica)
    {
        EXPECT_EQ(plan.rows_of(replica), rows[replica]) << "replica " << replica;
        EXPECT_EQ(plan.micro_batch_of(replica), micro_batches[replica]) << "replica " << replica;
        EXPECT_NEAR(plan.weight_of(replica), weights[replica], 1e-15) << "replica " << replica;

        const RowPositions range = plan.rows(replica, 2);
        EXPECT_EQ(range.first, firsts[replica]) << "replica " << replica;
        EXPECT_EQ(range.count, static_cast<std::uint64_t>(rows[replica]))
            << "replica " << replica;
    }
}

/** @brief Tests that a five-replica section turns into one rule per replica, in order. */
TEST(Small_ClusterSection, FiveReplicasBecomeOneRuleEach)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::PlacementRules rules, section.placement_rules());

    ASSERT_EQ(rules.size(), 5u);
    const char* workers[] = {"atlas", "bravo", "cobalt", "delta", "ember"};
    const int priorities[] = {0, 5, 0, 0, 0};
    const SushiRuntime::AbsentAction actions[] = {
        SushiRuntime::AbsentAction::Local, SushiRuntime::AbsentAction::Wait,
        SushiRuntime::AbsentAction::Fail, SushiRuntime::AbsentAction::Local,
        SushiRuntime::AbsentAction::Fail};
    for (std::size_t replica = 0; replica < 5; ++replica)
    {
        const SushiRuntime::PlacementRule& rule = rules.rule(replica);
        EXPECT_EQ(rule.name, "replica " + std::to_string(replica));
        EXPECT_EQ(rule.match.placement_class, K_REPLICA_PLACEMENT_CLASS);
        ASSERT_TRUE(rule.match.key.has_value()) << "replica " << replica;
        EXPECT_EQ(*rule.match.key, replica);
        EXPECT_TRUE(rule.match.operation.empty()) << "replica " << replica;
        EXPECT_EQ(rule.target.machine_name, workers[replica]) << "replica " << replica;
        EXPECT_FALSE(rule.target.machine_label.has_value()) << "replica " << replica;
        EXPECT_FALSE(rule.target.device_index.has_value()) << "replica " << replica;
        EXPECT_FALSE(rule.target.device_kind.has_value()) << "replica " << replica;
        EXPECT_FALSE(rule.target.device_min_memory.has_value()) << "replica " << replica;
        EXPECT_EQ(rule.priority, priorities[replica]) << "replica " << replica;
        EXPECT_EQ(rule.if_absent.action, actions[replica]) << "replica " << replica;
        EXPECT_FALSE(rule.deadline.has_value()) << "replica " << replica;
    }

    EXPECT_EQ(rules.rule(1).if_absent.bound, std::chrono::milliseconds(30000));
    EXPECT_EQ(rules.rule(0).if_absent.bound, std::chrono::milliseconds(0));
}

/** @brief Tests that the rules the section makes resolve each replica's node to its own rule. */
TEST(Small_ClusterSection, TheRulesResolveEachReplicaNodeToItsOwnRule)
{
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_replicas(K_FIVE_REPLICAS));
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::PlacementRules rules, section.placement_rules());

    const SushiRuntime::PlacementClass replica_class(K_REPLICA_PLACEMENT_CLASS.data(),
                                                     K_REPLICA_PLACEMENT_CLASS.size());
    for (std::uint64_t replica = 0; replica < 5; ++replica)
    {
        const std::optional<std::size_t> winner =
            rules.resolve(replica_class, replica, SushiRuntime::OperationID());
        ASSERT_TRUE(winner.has_value()) << "replica " << replica;
        EXPECT_EQ(*winner, replica);
    }

    EXPECT_FALSE(rules.resolve(replica_class, 5, SushiRuntime::OperationID()).has_value());
    EXPECT_FALSE(
        rules.resolve(SushiRuntime::PlacementClass(), 0, SushiRuntime::OperationID()).has_value());
}

/** @brief Tests that an absolute secret file keeps its path. */
TEST(Small_ClusterSection, AnAbsoluteSecretFileIsKept)
{
    const std::filesystem::path absolute = std::filesystem::absolute("elsewhere.secret");
    const std::string text = R"({"listen": "a:1", "secret_file": ")" +
                             absolute.generic_string() +
                             R"(", "replicas": [{"worker": "w", "rows": 4, "micro_batch": 2,
                                "if_absent": "fail"}]})";
    SA_ASSERT_OK_AND_ASSIGN(const ClusterSection section, read_text(text));

    EXPECT_EQ(section.secret_file().generic_string(), absolute.generic_string());
}

/** @brief Tests that a micro-batch that does not divide its rows is refused. */
TEST(Small_ClusterSection, AMicroBatchThatDoesNotDivideItsRowsIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 8, "micro_batch": 4, "if_absent": "local"},
        {"worker": "b", "rows": 10, "micro_batch": 4, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "cluster 'replicas'", "replica 1",
                      "'micro_batch' 4", "does not divide", "'rows' 10");
}

/** @brief Tests that a section with no replica is refused. */
TEST(Small_ClusterSection, NoReplicaIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas("[]"), StatusCode::InvalidDescription,
                      "cluster 'replicas'", "at least one replica");
    SA_EXPECT_FAILURE(read_text(R"({"listen": "a:1", "secret_file": "s"})"),
                      StatusCode::InvalidDescription, "cluster", "has no 'replicas'");
}

/** @brief Tests that a worker named by two replicas is refused with both entries. */
TEST(Small_ClusterSection, AWorkerNamedTwiceIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "local"},
        {"worker": "b", "rows": 4, "micro_batch": 2, "if_absent": "local"},
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 2", "'worker'", "'a'", "entry 0");
}

/** @brief Tests that two names placement takes for one machine are refused as one worker. */
TEST(Small_ClusterSection, Regression_TwoNamesOfOneMachineAreRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "Rack-7", "rows": 4, "micro_batch": 2, "if_absent": "local"},
        {"worker": "rack-7.lab", "rows": 4, "micro_batch": 2, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 1", "'worker'", "entry 0");
}

/** @brief Tests that a section without a listen address is refused. */
TEST(Small_ClusterSection, AMissingListenAddressIsRefused)
{
    SA_EXPECT_FAILURE(
        read_text(R"({"secret_file": "s", "replicas": [{"worker": "a", "rows": 4,
                      "micro_batch": 2, "if_absent": "local"}]})"),
        StatusCode::InvalidDescription, "cluster", "has no 'listen'");
}

/** @brief Tests that a section without a secret file is refused. */
TEST(Small_ClusterSection, AMissingSecretFileIsRefused)
{
    SA_EXPECT_FAILURE(
        read_text(R"({"listen": "a:1", "replicas": [{"worker": "a", "rows": 4,
                      "micro_batch": 2, "if_absent": "local"}]})"),
        StatusCode::InvalidDescription, "cluster", "has no 'secret_file'");
}

/** @brief Tests that an empty listen address and an empty secret file are refused. */
TEST(Small_ClusterSection, AnEmptyListenAddressOrSecretFileIsRefused)
{
    SA_EXPECT_FAILURE(
        read_text(R"({"listen": "", "secret_file": "s", "replicas": []})"),
        StatusCode::InvalidDescription, "cluster 'listen'", "empty");
    SA_EXPECT_FAILURE(
        read_text(R"({"listen": "a:1", "secret_file": "", "replicas": []})"),
        StatusCode::InvalidDescription, "cluster 'secret_file'", "empty");
}

/** @brief Tests that rows summing to zero are refused, since no weight can be formed. */
TEST(Small_ClusterSection, RowsThatSumToZeroAreRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 0, "micro_batch": 1, "if_absent": "local"},
        {"worker": "b", "rows": 0, "micro_batch": 3, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "cluster 'replicas'", "'rows'", "sum to 0");
}

/** @brief Tests that an "if_absent" outside local, wait and fail is refused. */
TEST(Small_ClusterSection, AnIfAbsentOutsideTheThreeIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "retry"}])"),
                      StatusCode::InvalidDescription, "entry 0 'if_absent'", "'retry'",
                      "'local'", "'wait'", "'fail'");
}

/** @brief Tests that a replica without an "if_absent" is refused. */
TEST(Small_ClusterSection, AMissingIfAbsentIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2}])"),
                      StatusCode::InvalidDescription, "entry 0", "has no 'if_absent'");
}

/** @brief Tests that a wait without a bound is refused. */
TEST(Small_ClusterSection, AWaitWithoutABoundIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "wait"}])"),
                      StatusCode::InvalidDescription, "entry 0 'if_absent'", "'wait'",
                      "'wait_ms'");
}

/** @brief Tests that a wait with a bound of zero or below is refused. */
TEST(Small_ClusterSection, AWaitWithoutAPositiveBoundIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "wait", "wait_ms": 0}])"),
                      StatusCode::InvalidDescription, "entry 0 'wait_ms'", "is 0", "positive");
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "wait", "wait_ms": -5}])"),
                      StatusCode::InvalidDescription, "entry 0 'wait_ms'", "is -5", "positive");
}

/** @brief Tests that a bound beside local or fail is refused. */
TEST(Small_ClusterSection, ABoundBesideLocalOrFailIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "local", "wait_ms": 10}])"),
                      StatusCode::InvalidDescription, "entry 0 'wait_ms'", "'local'",
                      "only 'wait'");
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "fail", "wait_ms": 10}])"),
                      StatusCode::InvalidDescription, "entry 0 'wait_ms'", "'fail'",
                      "only 'wait'");
}

/** @brief Tests that a replica missing a key, or stating one of the wrong kind, is refused. */
TEST(Small_ClusterSection, AReplicaWithAMissingOrMistypedKeyIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"rows": 4, "micro_batch": 2, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 0", "has no 'worker'");
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "micro_batch": 2, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 0", "has no 'rows'");
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 0", "has no 'micro_batch'");
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": "4", "micro_batch": 2, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 0 'rows'", "whole number");
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": 7, "rows": 4, "micro_batch": 2, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 0 'worker'", "string");
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "", "rows": 4, "micro_batch": 2, "if_absent": "local"}])"),
                      StatusCode::InvalidDescription, "entry 0 'worker'", "empty");
}

/** @brief Tests that a key the section or a replica does not have is refused. */
TEST(Small_ClusterSection, AnUnknownKeyIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "local", "speed": 3}])"),
                      StatusCode::InvalidDescription, "entry 0", "'speed'", "does not have");
    SA_EXPECT_FAILURE(read_text(R"({"listen": "a:1", "secret_file": "s", "replicas": [],
                                    "port": 7})"),
                      StatusCode::InvalidDescription, "cluster", "'port'", "does not have");
}

/** @brief Tests that a priority outside the range of an int is refused. */
TEST(Small_ClusterSection, APriorityOutsideTheRangeOfAnIntIsRefused)
{
    SA_EXPECT_FAILURE(read_replicas(R"([
        {"worker": "a", "rows": 4, "micro_batch": 2, "if_absent": "local",
         "priority": 3000000000}])"),
                      StatusCode::InvalidDescription, "entry 0 'priority'", "3000000000");
}

/** @brief Tests that a section that is no object is refused. */
TEST(Small_ClusterSection, ASectionThatIsNoObjectIsRefused)
{
    SA_EXPECT_FAILURE(read_text("[]"), StatusCode::InvalidDescription, "cluster",
                      "must be an object");
}
