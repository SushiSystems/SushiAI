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
 * @file test_log.cpp
 * @brief Tests log.hpp: where an SA_LOG record lands and what the sushiai level reaches.
 * @author Mustafa Garip
 */

#include <SushiAI/log/log.hpp>

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_category.hpp>
#include <SushiLog/log/log_project.hpp>
#include <SushiLog/log/log_record.hpp>
#include <SushiRuntime/log/sushiruntime_log_project.hpp>

#include <SushiAI/log/sushiai_log_project.hpp>

namespace
{
    /** @brief Names the records these tests send, written as [sushiai.log_test]. */
    constexpr SushiLog::LogCategory K_LOG_TEST_CATEGORY =
        SushiLog::make_log_category(SushiAI::K_SUSHIAI_LOG_PROJECT, "log_test");

    /** @brief Collects the category and the message of every record it receives. */
    struct CapturedRecords
    {
        std::vector<std::string> categories;
        std::vector<std::string> messages;
    };

    /** @brief Routes every record to @p captured on the calling thread. */
    void capture_into(CapturedRecords& captured)
    {
        SushiLog::set_log_synchronous(true);
        SushiLog::set_log_sink([&captured](const SushiLog::LogRecord& record)
        {
            captured.categories.push_back(std::string(record.project) + "." +
                                          std::string(record.module));
            captured.messages.push_back(std::string(record.message));
        });
    }
} // namespace

/** @brief Tests that an SA_LOG_WARN record reaches the sink under its project and module. */
TEST(Small_Log, AWarnRecordCarriesItsProjectAndModule)
{
    using SushiAI::K_SUSHIAI_LOG_PROJECT;
    CapturedRecords captured;
    capture_into(captured);
    const SushiLog::LogLevel saved = SushiLog::log_level(K_SUSHIAI_LOG_PROJECT);
    SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, SushiLog::LogLevel::Warn);

    SA_LOG_WARN(K_LOG_TEST_CATEGORY, "extent {} is refused", 7);

    SushiLog::set_log_sink(nullptr);
    SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, saved);
    EXPECT_EQ(captured.categories, std::vector<std::string>{"sushiai.log_test"});
    EXPECT_EQ(captured.messages, std::vector<std::string>{"extent 7 is refused"});
}

/** @brief Tests that setting the sushiai project to Off silences SushiAI and leaves the level
 *         of the sushiruntime project as it was. */
TEST(Small_Log, TurningSushiAIOffLeavesSushiRuntimeAtItsLevel)
{
    using SushiAI::K_SUSHIAI_LOG_PROJECT;
    using SushiLog::LogLevel;
    using SushiRuntime::K_SUSHIRUNTIME_LOG_PROJECT;
    CapturedRecords captured;
    capture_into(captured);
    const LogLevel saved = SushiLog::log_level(K_SUSHIAI_LOG_PROJECT);
    const LogLevel runtime_before = SushiLog::log_level(K_SUSHIRUNTIME_LOG_PROJECT);
    SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, LogLevel::Off);

    SA_LOG_ERROR(K_LOG_TEST_CATEGORY, "never delivered");
    const LogLevel runtime_after = SushiLog::log_level(K_SUSHIRUNTIME_LOG_PROJECT);

    SushiLog::set_log_sink(nullptr);
    SushiLog::set_log_level(K_SUSHIAI_LOG_PROJECT, saved);
    EXPECT_TRUE(captured.messages.empty());
    EXPECT_EQ(runtime_after, runtime_before);
}
