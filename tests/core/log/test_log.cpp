/**************************************************************************/
/* test_log.cpp                                                           */
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
 * @file test_log.cpp
 * @brief Tests log.hpp: where an SA_LOG record lands and what the sushiai level reaches.
 * @author Mustafa Garip
 */

#include <SushiAI/log/log.hpp>

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_capture.hpp>
#include <SushiLog/log/log_category.hpp>
#include <SushiLog/log/log_project.hpp>
#include <SushiRuntime/log/sushiruntime_log_project.hpp>

#include <SushiAI/log/sushiai_log_project.hpp>

namespace
{
    /** @brief Names the records these tests send, written as [sushiai.log_test]. */
    constexpr SushiLog::LogCategory K_LOG_TEST_CATEGORY =
        SushiLog::make_log_category(SushiAI::K_SUSHIAI_LOG_PROJECT, "log_test");
} // namespace

/** @brief Tests that an SA_LOG_WARN record reaches the sink under its project and module. */
TEST(Small_Log, AWarnRecordCarriesItsProjectAndModule)
{
    using SushiAI::K_SUSHIAI_LOG_PROJECT;
    const SushiLog::LogLevel saved = SushiLog::log_level(K_SUSHIAI_LOG_PROJECT);
    std::vector<SushiLog::CapturedRecord> records;
    {
        const SushiLog::LogCapture capture;
        SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, SushiLog::LogLevel::Warn);

        SA_LOG_WARN(K_LOG_TEST_CATEGORY, "extent {} is refused", 7);

        records = capture.records();
    }
    SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, saved);
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records.front().project + "." + records.front().module, "sushiai.log_test");
    EXPECT_EQ(records.front().message, "extent 7 is refused");
}

/** @brief Tests that setting the sushiai project to Off silences SushiAI and leaves the level
 *         of the sushiruntime project as it was. */
TEST(Small_Log, TurningSushiAIOffLeavesSushiRuntimeAtItsLevel)
{
    using SushiAI::K_SUSHIAI_LOG_PROJECT;
    using SushiLog::LogLevel;
    using SushiRuntime::K_SUSHIRUNTIME_LOG_PROJECT;
    const LogLevel saved = SushiLog::log_level(K_SUSHIAI_LOG_PROJECT);
    const LogLevel runtime_before = SushiLog::log_level(K_SUSHIRUNTIME_LOG_PROJECT);
    std::vector<SushiLog::CapturedRecord> records;
    LogLevel runtime_after = runtime_before;
    {
        const SushiLog::LogCapture capture;
        SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, LogLevel::Off);

        SA_LOG_ERROR(K_LOG_TEST_CATEGORY, "never delivered");
        runtime_after = SushiLog::log_level(K_SUSHIRUNTIME_LOG_PROJECT);

        records = capture.records();
    }
    SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, saved);
    EXPECT_TRUE(records.empty());
    EXPECT_EQ(runtime_after, runtime_before);
}
