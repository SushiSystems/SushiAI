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
        return Distribution::ClusterSection::read(Config::parse_json(text, "<test>"), ".")
            .value();
    }

    /** @brief Returns the section of a master and two replicas that both name the worker "left". */
    Distribution::ClusterSection one_worker_two_replicas()
    {
        const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret",
            "attach_timeout_ms": 200, "replicas": [
                {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
                {"worker": "left", "rows": 1, "micro_batch": 1, "if_absent": "local"},
                {"worker": "left", "rows": 1, "micro_batch": 1, "if_absent": "local"}
            ]})";
        return Distribution::ClusterSection::read(Config::parse_json(text, "<test>"), ".")
            .value();
    }

    /** @brief Holds a master and the workers a case starts, joined over one InProcessFabric. */
    class Fabric final
    {
        public:
            /** @brief Attaches a master with no worker yet. */
            Fabric()
                : master_runtime_(SushiRuntime::Runtime::create().value()),
                  cluster_(Distributed::Cluster::attach(
                               master_runtime_, Distributed::InProcessTransport::make_master(
                                                    fabric_)).value())
            {

            }

            Fabric(const Fabric&) = delete;
            Fabric& operator=(const Fabric&) = delete;
            Fabric(Fabric&&) = delete;
            Fabric& operator=(Fabric&&) = delete;
            /** @brief Stops the workers before the master. */
            ~Fabric() = default;

            /** @brief Starts a worker whose HELLO names @p name. */
            void start_worker(const std::string& name)
            {
                worker_runtimes_.push_back(std::make_unique<SushiRuntime::Runtime>(
                    SushiRuntime::Runtime::create().value()));
                Distributed::WorkerAgentOptions options;
                options.name = name;
                agents_.push_back(std::make_unique<Distributed::WorkerAgent>(
                    *worker_runtimes_.back(), Distributed::InProcessTransport::make_worker(fabric_),
                    SushiRuntime::make_sushi<Distributed::KernelRegistry>(), std::move(options)));
                ASSERT_TRUE(agents_.back()->start().is_ok());
            }

            /** @brief Returns the master's cluster handle. */
            Distributed::Cluster& cluster() noexcept
            {
                return cluster_;
            }

        private:
            /// @brief The fabric every machine talks over; it dies last.
            Distributed::InProcessFabric fabric_;

            /// @brief The master's runtime, with the cluster attached.
            SushiRuntime::Runtime master_runtime_;

            /// @brief The master.
            Distributed::Cluster cluster_;

            /// @brief The runtime of each worker, in start order.
            std::vector<std::unique_ptr<SushiRuntime::Runtime>> worker_runtimes_;

            /// @brief The workers, in start order; they die first.
            std::vector<std::unique_ptr<Distributed::WorkerAgent>> agents_;
    };
} // namespace

/** @brief Tests that an absent "attach_timeout_ms" bounds the wait by the default. */
TEST(Medium_MasterCluster, TheAttachBoundDefaultsToSixtySeconds)
{
    EXPECT_EQ(Apps::attach_bound(two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")")),
              Distribution::K_DEFAULT_ATTACH_TIMEOUT);
}

/** @brief Tests that a wait rule longer than the attach bound raises it. */
TEST(Medium_MasterCluster, RegressionTheLargestWaitRuleRaisesTheAttachBound)
{
    const std::chrono::milliseconds bound = Apps::attach_bound(
        two_workers(R"("if_absent": "wait", "wait_ms": 250)",
                    R"("if_absent": "wait", "wait_ms": 4000)", K_SHORT_ATTACH));

    EXPECT_EQ(bound.count(), 4000);
}

/** @brief Tests that a section whose "attach_timeout_ms" is not positive is refused. */
TEST(Medium_MasterCluster, RefusesAnAttachTimeoutThatIsNotPositive)
{
    const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret",
        "attach_timeout_ms": 0,
        "replicas": [{"worker": "master", "rows": 1, "micro_batch": 1, "if_absent": "fail"}]})";

    SA_EXPECT_FAILURE(
        Distribution::ClusterSection::read(Config::parse_json(text, "<test>"), ".").status(),
        StatusCode::InvalidDescription, "attach_timeout_ms");
}

/** @brief Tests that the master goes on at the last attach, long before its bound. */
TEST(Medium_MasterCluster, RegressionEveryWorkerAttachedEndsTheWait)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")", K_LONG_ATTACH);
    Fabric fabric;
    fabric.start_worker("left");
    fabric.start_worker("right");

    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    SA_EXPECT_OK(Apps::wait_for_workers(fabric.cluster(), section).status());

    // Far below the 600 s bound: the wait ended at the attach, not at the bound.
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(60));
}

/** @brief Tests that a missing worker whose rule is "fail" fails the run, by name. */
TEST(Medium_MasterCluster, RegressionAMissingWorkerUnderFailFailsNamingIt)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")", K_SHORT_ATTACH);
    Fabric fabric;
    fabric.start_worker("left");

    SA_EXPECT_FAILURE(Apps::wait_for_workers(fabric.cluster(), section).status(),
                      StatusCode::DeviceFailed, "'right'");
}

/** @brief Tests that a missing worker whose rule is "local" lets the run go on. */
TEST(Medium_MasterCluster, AMissingWorkerUnderLocalRuns)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "local")", K_SHORT_ATTACH);
    Fabric fabric;
    fabric.start_worker("left");

    SA_EXPECT_OK(Apps::wait_for_workers(fabric.cluster(), section).status());
}

/** @brief Tests that two replicas on one worker are one awaited worker. */
TEST(Medium_MasterCluster, TwoReplicasOnOneWorkerCountAsOneAwaitedWorker)
{
    const std::vector<Apps::AwaitedWorker> awaited =
        Apps::awaited_workers(one_worker_two_replicas());

    ASSERT_EQ(awaited.size(), 1u);
    EXPECT_EQ(awaited.front().name, "left");
    EXPECT_EQ(awaited.front().replicas, (std::vector<std::size_t>{1, 2}));
}

/** @brief Tests that the one worker serving two replicas, once attached, leaves nothing missing. */
TEST(Medium_MasterCluster, AWorkerServingTwoReplicasAttachedLeavesNoneMissing)
{
    const Distribution::ClusterSection section = one_worker_two_replicas();
    Fabric fabric;
    fabric.start_worker("left");

    const Result<std::vector<Apps::AwaitedWorker>> missing =
        Apps::wait_for_workers(fabric.cluster(), section);

    SA_EXPECT_OK(missing.status());
    EXPECT_TRUE(missing.value().empty());
}

/** @brief Tests that a worker with an unexpected name does not hide the missing one. */
TEST(Medium_MasterCluster, AMissingWorkerIsNamedInWhatTheWaitReturns)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "local")", R"("if_absent": "local")", K_SHORT_ATTACH);
    Fabric fabric;
    fabric.start_worker("left");
    fabric.start_worker("stranger");

    const Result<std::vector<Apps::AwaitedWorker>> missing =
        Apps::wait_for_workers(fabric.cluster(), section);

    SA_EXPECT_OK(missing.status());
    ASSERT_EQ(missing.value().size(), 1u);
    EXPECT_EQ(missing.value().front().name, "right");
    EXPECT_EQ(missing.value().front().replicas, (std::vector<std::size_t>{2}));
}

/** @brief Tests that a replica on the master's own machine is not awaited and fails nothing. */
TEST(Medium_MasterCluster, AReplicaOnTheMasterIsNotAwaitedAndDoesNotFail)
{
    const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret",
        "attach_timeout_ms": 200, "replicas": [
            {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
            {"worker": "Master.lab.example", "rows": 1, "micro_batch": 1, "if_absent": "fail"}
        ]})";
    const Distribution::ClusterSection section =
        Distribution::ClusterSection::read(Config::parse_json(text, "<test>"), ".").value();
    Fabric fabric;

    EXPECT_TRUE(Apps::awaited_workers(section).empty());

    const Result<std::vector<Apps::AwaitedWorker>> missing =
        Apps::wait_for_workers(fabric.cluster(), section);

    SA_EXPECT_OK(missing.status());
    EXPECT_TRUE(missing.value().empty());
}

/** @brief Tests that a worker announcing a qualified, upper-case name satisfies its first label. */
TEST(Medium_MasterCluster, AWorkerAnnouncingAQualifiedNameSatisfiesTheSectionsFirstLabel)
{
    const Distribution::ClusterSection section =
        two_workers(R"("if_absent": "fail")", R"("if_absent": "fail")", K_SHORT_ATTACH);
    Fabric fabric;
    fabric.start_worker("Left.lan");
    fabric.start_worker("RIGHT.lab.example");

    const Result<std::vector<Apps::AwaitedWorker>> missing =
        Apps::wait_for_workers(fabric.cluster(), section);

    SA_EXPECT_OK(missing.status());
    EXPECT_TRUE(missing.value().empty());
}
