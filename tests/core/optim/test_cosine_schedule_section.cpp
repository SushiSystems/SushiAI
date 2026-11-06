/**************************************************************************/
/* test_cosine_schedule_section.cpp                                       */
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
 * @file test_cosine_schedule_section.cpp
 * @brief Tests the type and keys the cosine section writer states and what both sides refuse.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/cosine_schedule_section.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Optim::CosineSchedule;
    using SushiAI::Optim::K_COSINE_SCHEDULE_TYPE;
    using SushiAI::Optim::read_cosine_schedule_section;
    using SushiAI::Optim::write_cosine_schedule_section;

    /** @brief Parses @p text as a section of a training file. */
    Value section_from(std::string_view text)
    {
        return require_value(SushiAI::Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that the writer states the type and every key, each read back as written. */
TEST(Small_CosineScheduleSection, TheWriterStatesTheTypeAndEveryKey)
{
    CosineSchedule schedule;
    schedule.base_rate = 0.5;
    schedule.warmup_iterations = 2;
    schedule.total_iterations = 10;
    schedule.final_rate = 0.25;

    SA_ASSERT_OK_AND_ASSIGN(const Value section, write_cosine_schedule_section(schedule));

    const auto read = [&section](const char* key) -> const Value&
    {
        return require_value(section.at(key, "section"));
    };

    EXPECT_EQ(require_value(read("type").as_string("type")), K_COSINE_SCHEDULE_TYPE);
    EXPECT_DOUBLE_EQ(require_value(read("base_rate").as_real("base_rate")), 0.5);
    EXPECT_EQ(require_value(read("warmup_iterations").as_int("warmup_iterations")), 2);
    EXPECT_EQ(require_value(read("total_iterations").as_int("total_iterations")), 10);
    EXPECT_DOUBLE_EQ(require_value(read("final_rate").as_real("final_rate")), 0.25);
}

/** @brief Tests that a section whose type the caller took out is read like a full one. */
TEST(Small_CosineScheduleSection, ASectionWithoutItsTypeIsRead)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const CosineSchedule read,
        read_cosine_schedule_section(
            section_from(R"({"base_rate": 0.5, "total_iterations": 10, "final_rate": 0.25})")));

    EXPECT_DOUBLE_EQ(read.base_rate, 0.5);
    EXPECT_EQ(read.total_iterations, 10);
    EXPECT_DOUBLE_EQ(read.final_rate, 0.25);
}

/** @brief Tests that the writer refuses a schedule the schedule check refuses. */
TEST(Small_CosineScheduleSection, TheWriterRefusesWhatTheCheckRefuses)
{
    CosineSchedule rising;
    rising.base_rate = 0.1;
    rising.total_iterations = 10;
    rising.final_rate = 0.2;

    SA_EXPECT_FAILURE(write_cosine_schedule_section(rising), StatusCode::InvalidDescription,
                      "above the base rate");
}
