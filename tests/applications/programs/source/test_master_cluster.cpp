/**************************************************************************/
/* test_master_cluster.cpp                                                */
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
 * @file test_master_cluster.cpp
 * @brief Tests the master side of a cluster run: its runtime's name and its attach.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "cluster_test_files.hpp"
#include "master_cluster.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;
} // namespace

/** @brief Tests that the master's runtime reports the machine replica 0 names. */
TEST(Medium_MasterCluster, TheRuntimeReportsTheMachineOfReplicaZero)
{
    const ClusterTestFiles files("sushiai_master_cluster_name.json");

    EXPECT_EQ(Apps::master_runtime_config(files.section()).node_name, "master");
}

/** @brief Tests that a listen address that does not parse is refused before anything starts. */
TEST(Medium_MasterCluster, RefusesAListenAddressThatDoesNotParse)
{
    const ClusterTestFiles files("sushiai_master_cluster_listen.json", "no port here");
    SushiRuntime::Runtime runtime =
        Detail::create_runtime(Apps::master_runtime_config(files.section())).value();

    SA_EXPECT_FAILURE(Apps::attach_master_cluster(runtime, files.section()).status(),
                      StatusCode::InvalidDescription);
}

namespace
{
    /** @brief Returns the section of a master and two workers with absent policies @p a, @p b. */
    Distribution::ClusterSection two_workers(const std::string& a, const std::string& b)
    {
        const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret",
            "replicas": [
                {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
                {"worker": "left", "rows": 1, "micro_batch": 1, )" + a + R"(},
                {"worker": "right", "rows": 1, "micro_batch": 1, )" + b + R"(}
            ]})";
        return Distribution::ClusterSection::read(Config::parse_json(text, "<test>"), ".")
            .value();
    }
} // namespace

/** @brief Tests that the master waits up to the largest bound of the workers' wait rules. */
TEST(Medium_MasterCluster, RegressionTheWaitIsTheLargestBoundOfTheWaitRules)
{
    const Apps::WorkerWait wait = Apps::worker_wait(
        two_workers(R"("if_absent": "wait", "wait_ms": 250)",
                    R"("if_absent": "wait", "wait_ms": 4000)"));

    EXPECT_EQ(wait.bound.count(), 4000);
    EXPECT_FALSE(wait.fails_when_short);
}

/** @brief Tests that a section of fail rules waits for nothing and fails when short. */
TEST(Medium_MasterCluster, RegressionEveryFailRuleWaitsForNothingAndFailsWhenShort)
{
    const Apps::WorkerWait wait =
        Apps::worker_wait(two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")"));

    EXPECT_EQ(wait.bound.count(), 0);
    EXPECT_TRUE(wait.fails_when_short);
}

/** @brief Tests that one local rule among fail rules lets a short count go on. */
TEST(Medium_MasterCluster, ALocalRuleLetsAShortCountGoOn)
{
    const Apps::WorkerWait wait =
        Apps::worker_wait(two_workers(R"("if_absent": "fail")", R"("if_absent": "local")"));

    EXPECT_EQ(wait.bound.count(), 0);
    EXPECT_FALSE(wait.fails_when_short);
}
