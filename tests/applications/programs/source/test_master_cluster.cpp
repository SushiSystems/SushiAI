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
 * @file test_master_cluster.cpp
 * @brief Tests the master side of a cluster run: its runtime's name, its attach and its wait.
 * @author Mustafa Garip
 */

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/in_process_transport.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiRuntime/distributed/worker_agent.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/test_names.hpp>

#include "cluster_test_files.hpp"
#include "cluster_fabric.hpp"
#include "in_memory_worker_watch.hpp"
#include "master_cluster.hpp"

namespace
{
    using SushiAI::Test::require_value;
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;
} // namespace

namespace
{
    namespace Distributed = SushiRuntime::Distributed;

    /** @brief Holds an attach bound far above any wait a case may see. */
    constexpr const char* K_LONG_ATTACH = R"("attach_timeout_ms": 600000, )";

    /** @brief Holds an attach bound a case with a missing worker waits out. */
    constexpr const char* K_SHORT_ATTACH = R"("attach_timeout_ms": 200, )";

    /**
     * @brief Returns the section of a master and two workers with absent policies @p a, @p b.
     * @param attach The "attach_timeout_ms" member and its comma, or empty for the default.
     */
    Distribution::ClusterSection two_workers(const std::string& a, const std::string& b,
                                             const std::string& attach = "")
    {
        const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret", )"
                                 + attach + R"("replicas": [
                {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
                {"worker": "left", "rows": 1, "micro_batch": 1, )" + a + R"(},
                {"worker": "right", "rows": 1, "micro_batch": 1, )" + b + R"(}
            ]})";
        return Distribution::ClusterSection::read(require_value(Config::parse_json(text, "<test>")),
                                                  ".")
            .value();
    }

    /** @brief Holds the names two workers announce in a case of the name rows. */
    struct AnnouncedNames
    {
        /** @brief Names the case. */
        const char* name;

        /** @brief What the worker of the section's second label announces. */
        const char* left;

        /** @brief What the worker of the section's third label announces. */
        const char* right;
    };

    /** @brief Runs the rows of the names two workers announce. */
    class Medium_MasterClusterAnnouncedNames : public ::testing::TestWithParam<AnnouncedNames> {};

    /** @brief Returns the section of the master alone, with a short attach bound. */
    Distribution::ClusterSection master_alone()
    {
        const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret",
        "attach_timeout_ms": 200, "replicas": [
            {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"}
        ]})";
        return Distribution::ClusterSection::read(require_value(Config::parse_json(text, "<test>")),
                                                  ".").value();
    }
} // namespace

/** @brief Tests that the master's runtime reports the machine replica 0 names. */
TEST(Small_MasterCluster, TheRuntimeReportsTheMachineOfReplicaZero)
{
    const ClusterTestSpec spec;

    EXPECT_EQ(Apps::master_runtime_config(spec.section()).node_name, "master");
}

/** @brief Tests that a listen address that does not parse is refused before anything starts. */
TEST(Medium_MasterCluster, RefusesAListenAddressThatDoesNotParse)
{
    const ClusterTestSpec spec("no port here");
    SushiRuntime::Runtime runtime =
        SushiRuntime::Runtime::create(Apps::master_runtime_config(spec.section())).value();

    SA_EXPECT_FAILURE(Apps::attach_master_cluster(runtime, spec.section()).status(),
                      StatusCode::InvalidDescription);
}

/** @brief Tests that an absent "attach_timeout_ms" bounds the wait by the default. */
TEST(Small_MasterCluster, TheAttachBoundDefaultsToSixtySeconds)
{
    EXPECT_EQ(Apps::attach_bound(two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")")),
              Distribution::K_DEFAULT_ATTACH_TIMEOUT);
}

/** @brief Tests that a wait rule longer than the attach bound raises it. */
TEST(Small_MasterCluster, RegressionTheLargestWaitRuleRaisesTheAttachBound)
{
    const std::chrono::milliseconds bound = Apps::attach_bound(
        two_workers(R"("if_absent": "wait", "wait_ms": 250)",
                    R"("if_absent": "wait", "wait_ms": 4000)", K_SHORT_ATTACH));

    EXPECT_EQ(bound.count(), 4000);
}

/** @brief Tests that a section whose "attach_timeout_ms" is not positive is refused. */
TEST(Small_MasterCluster, RefusesAnAttachTimeoutThatIsNotPositive)
{
    const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret",
        "attach_timeout_ms": 0,
        "replicas": [{"worker": "master", "rows": 1, "micro_batch": 1, "if_absent": "fail"}]})";

    SA_EXPECT_FAILURE(
        Distribution::ClusterSection::read(
            SushiAI::Test::require_value(Config::parse_json(text, "<test>")), ".").status(),
        StatusCode::InvalidDescription, "attach_timeout_ms");
}

/** @brief Tests that the master goes on at the last attach, long before its bound. */
TEST(Small_MasterCluster, RegressionEveryWorkerAttachedEndsTheWait)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")", K_LONG_ATTACH);
    Apps::InMemoryWorkerWatch watch({{"left", std::chrono::milliseconds(2000)},
                               {"right", std::chrono::milliseconds(5000)}});

    SA_EXPECT_OK(Apps::wait_for_workers(watch, section).status());

    // The wait ended at the last attach, 5 s in, not at the 600 s bound.
    EXPECT_EQ(watch.elapsed(), std::chrono::milliseconds(5000));
}

/** @brief Tests that a missing worker whose rule is "fail" fails the run, by name. */
TEST(Small_MasterCluster, RegressionAMissingWorkerUnderFailFailsNamingIt)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")", K_SHORT_ATTACH);
    Apps::InMemoryWorkerWatch watch({{"left", std::chrono::milliseconds(0)}});

    SA_EXPECT_FAILURE(Apps::wait_for_workers(watch, section).status(),
                      StatusCode::DeviceFailed, "'right'");
    EXPECT_EQ(watch.elapsed(), std::chrono::milliseconds(200));
}

/** @brief Tests that a missing worker whose rule is "local" lets the run go on. */
TEST(Small_MasterCluster, AMissingWorkerUnderLocalRuns)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "local")", K_SHORT_ATTACH);
    Apps::InMemoryWorkerWatch watch({{"left", std::chrono::milliseconds(0)}});

    SA_EXPECT_OK(Apps::wait_for_workers(watch, section).status());
}

/** @brief Tests that the master awaits the worker of each replica above 0, in replica order. */
TEST(Small_MasterCluster, AwaitsTheWorkerOfEachReplicaAboveZeroInOrder)
{
    const std::vector<Apps::AwaitedWorker> awaited = Apps::awaited_workers(
        two_workers(R"("if_absent": "fail")", R"("if_absent": "local")", K_SHORT_ATTACH));

    ASSERT_EQ(awaited.size(), 2u);
    EXPECT_EQ(awaited[0].name, "left");
    EXPECT_EQ(awaited[0].replica, 1u);
    EXPECT_EQ(awaited[1].name, "right");
    EXPECT_EQ(awaited[1].replica, 2u);
}

/** @brief Tests that a worker with an unexpected name does not hide the missing one. */
TEST(Small_MasterCluster, AMissingWorkerIsNamedInWhatTheWaitReturns)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "local")", R"("if_absent": "local")", K_SHORT_ATTACH);
    Apps::InMemoryWorkerWatch watch({{"left", std::chrono::milliseconds(0)},
                               {"stranger", std::chrono::milliseconds(0)}});

    const Result<std::vector<Apps::AwaitedWorker>> missing =
        Apps::wait_for_workers(watch, section);

    SA_EXPECT_OK(missing.status());
    ASSERT_EQ(missing.value().size(), 1u);
    EXPECT_EQ(missing.value().front().name, "right");
    EXPECT_EQ(missing.value().front().replica, 2u);
}

/** @brief Tests that a section of the master alone awaits no worker. */
TEST(Small_MasterCluster, AMasterAloneAwaitsNoWorker)
{
    EXPECT_TRUE(Apps::awaited_workers(master_alone()).empty());
}

/** @brief Tests that the wait of a section of the master alone returns nothing missing. */
TEST(Medium_MasterCluster, TheWaitOfAMasterAloneReturnsNothingMissing)
{
    const Distribution::ClusterSection section = master_alone();
    Fabric fabric;

    const Result<std::vector<Apps::AwaitedWorker>> missing =
        Apps::wait_for_workers(fabric.cluster(), section);

    SA_EXPECT_OK(missing.status());
    EXPECT_TRUE(missing.value().empty());
}

/** @brief Tests that a worker announcing a qualified or recased name satisfies its label. */
TEST_P(Medium_MasterClusterAnnouncedNames, ANameOfAnotherFormSatisfiesTheSectionsLabel)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")", K_SHORT_ATTACH);
    Fabric fabric;
    fabric.start_worker(GetParam().left);
    fabric.start_worker(GetParam().right);

    const Result<std::vector<Apps::AwaitedWorker>> missing =
        Apps::wait_for_workers(fabric.cluster(), section);

    SA_EXPECT_OK(missing.status());
    EXPECT_TRUE(missing.value().empty());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_MasterClusterAnnouncedNames,
    ::testing::Values(AnnouncedNames{"AQualifiedUpperCaseName", "Left.lan", "RIGHT.lab.example"},
                      AnnouncedNames{"TheBareLabels", "left", "right"}),
    [](const ::testing::TestParamInfo<AnnouncedNames>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });
