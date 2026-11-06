/**************************************************************************/
/* test_replica_fallback_warning.cpp                                      */
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
 * @file test_replica_fallback_warning.cpp
 * @brief Tests ReplicaFallbackWarning: one WARN per replica and cause, none for the master.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_capture.hpp>
#include <SushiLog/log/log_level.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/log/sushiai_log_project.hpp>

#include "master_cluster.hpp"
#include "replica_fallback_warning.hpp"

namespace
{
    using SushiAI::Test::require_value;
    using namespace SushiAI;

    /** @brief Returns the section of a master and the workers "left" (local), "right" (wait). */
    Distribution::ClusterSection two_workers()
    {
        const std::string text = R"({"listen": "127.0.0.1:0", "secret_file": "cluster.secret",
            "replicas": [
                {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
                {"worker": "left", "rows": 1, "micro_batch": 1, "if_absent": "local"},
                {"worker": "right", "rows": 1, "micro_batch": 1, "if_absent": "wait",
                 "wait_ms": 250}
            ]})";
        return Distribution::ClusterSection::read(require_value(Config::parse_json(text, "<test>")),
                                                  ".")
            .value();
    }

    /** @brief Captures every record through SushiLog's LogCapture with sushiai at Warn while it
     *         lives. */
    class CapturedRecords final
    {
        public:
            /** @brief Sets the sushiai level to Warn. */
            CapturedRecords() : saved_(SushiLog::log_level(K_SUSHIAI_LOG_PROJECT))
            {
                SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, SushiLog::LogLevel::Warn);
            }

            CapturedRecords(const CapturedRecords&) = delete;
            CapturedRecords& operator=(const CapturedRecords&) = delete;
            CapturedRecords(CapturedRecords&&) = delete;
            CapturedRecords& operator=(CapturedRecords&&) = delete;

            /** @brief Restores the saved level. */
            ~CapturedRecords()
            {
                SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, saved_);
            }

            /** @brief Returns the project and module of each record, in order. */
            [[nodiscard]] std::vector<std::string> categories() const
            {
                std::vector<std::string> result;
                for (const SushiLog::CapturedRecord& record : capture_.records())
                    result.push_back(record.project + "." + record.module);
                return result;
            }

            /** @brief Returns the message of each record, in order. */
            [[nodiscard]] std::vector<std::string> messages() const
            {
                std::vector<std::string> result;
                for (const SushiLog::CapturedRecord& record : capture_.records())
                    result.push_back(record.message);
                return result;
            }

        private:
            /// @brief The sushiai level before the capture.
            SushiLog::LogLevel saved_;

            /// @brief The scoped sink of SushiLog.
            SushiLog::LogCapture capture_;
    };

    /** @brief Returns whether @p text holds @p part. */
    bool holds(const std::string& text, const std::string& part)
    {
        return text.find(part) != std::string::npos;
    }
} // namespace

/** @brief Tests that a replica placed on a worker warns once, naming it and the step. */
TEST(Small_ReplicaFallbackWarning, AReplicaThatRanOnTheMasterWarnsOnceWithItsStep)
{
    Apps::ReplicaFallbackWarning warning(two_workers());
    CapturedRecords captured;

    warning.on_served(1, Distribution::ReplicaPosition::at_step(5, 2));
    warning.on_served(1, Distribution::ReplicaPosition::at_step(6, 2));

    ASSERT_EQ(captured.messages().size(), 1u);
    EXPECT_EQ(captured.categories().front(), "sushiai.programs");
    const std::string message = captured.messages().front();
    EXPECT_TRUE(holds(message, "replica 1 ran on the master at step 5")) << message;
    EXPECT_TRUE(holds(message, "epoch 2, batch 1")) << message;
    EXPECT_TRUE(holds(message, "'left'")) << message;
}

/** @brief Tests that replica 0, whose rule names the master, never warns. */
TEST(Small_ReplicaFallbackWarning, AReplicaOnTheMastersMachineNeverWarns)
{
    Apps::ReplicaFallbackWarning warning(two_workers());
    CapturedRecords captured;

    warning.on_served(0, Distribution::ReplicaPosition::at_step(0, 2));
    warning.on_served(7, Distribution::ReplicaPosition::at_step(0, 2));

    EXPECT_TRUE(captured.messages().empty());
}

/** @brief Tests that each absent worker's replica warns once, naming what its policy does. */
TEST(Small_ReplicaFallbackWarning, AnAbsentWorkerWarnsEachOfItsReplicasOnce)
{
    Apps::ReplicaFallbackWarning warning(two_workers());
    CapturedRecords captured;

    warning.warn_absent_at_attach(Apps::AwaitedWorker{"left", 1});
    warning.warn_absent_at_attach(Apps::AwaitedWorker{"left", 1});
    warning.warn_absent_at_attach(Apps::AwaitedWorker{"right", 2});

    ASSERT_EQ(captured.messages().size(), 2u);
    EXPECT_TRUE(holds(captured.messages()[0], "worker 'left' did not attach; replica 1 runs on "
                                            "the master"))
        << captured.messages()[0];
    EXPECT_TRUE(holds(captured.messages()[1], "replica 2 waits up to 250 ms"))
        << captured.messages()[1];
}

/** @brief Tests that a replica warned of at attach still warns when it first runs there. */
TEST(Small_ReplicaFallbackWarning, TheTwoCausesWarnApart)
{
    Apps::ReplicaFallbackWarning warning(two_workers());
    CapturedRecords captured;

    warning.warn_absent_at_attach(Apps::AwaitedWorker{"left", 1});
    warning.on_served(1, Distribution::ReplicaPosition::at_step(0, 2));

    EXPECT_EQ(captured.messages().size(), 2u);
}
