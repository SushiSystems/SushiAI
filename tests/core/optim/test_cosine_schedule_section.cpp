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
 * @file test_cosine_schedule_section.cpp
 * @brief Tests the type and keys the cosine section writer states and what both sides refuse.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/cosine_schedule_section.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Optim::CosineSchedule;
    using SushiAI::Optim::K_COSINE_SCHEDULE_TYPE;
    using SushiAI::Optim::read_cosine_schedule_section;
    using SushiAI::Optim::write_cosine_schedule_section;

    /** @brief Bounds the error of a rate a section states and a double holds as read. */
    constexpr double K_RATE_TOLERANCE = 0.0;

    /** @brief Parses @p text as a section of a training file. */
    Value section_from(std::string_view text)
    {
        return SushiAI::Config::parse_json(text, "<test>");
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

    EXPECT_EQ(section.at("type", "section").as_string("type"), K_COSINE_SCHEDULE_TYPE);
    EXPECT_NEAR(section.at("base_rate", "section").as_real("base_rate"), 0.5, K_RATE_TOLERANCE);
    EXPECT_EQ(section.at("warmup_iterations", "section").as_int("warmup_iterations"), 2);
    EXPECT_EQ(section.at("total_iterations", "section").as_int("total_iterations"), 10);
    EXPECT_NEAR(section.at("final_rate", "section").as_real("final_rate"), 0.25,
                K_RATE_TOLERANCE);
}

/** @brief Tests that a section whose type the caller took out is read like a full one. */
TEST(Small_CosineScheduleSection, ASectionWithoutItsTypeIsRead)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const CosineSchedule read,
        read_cosine_schedule_section(
            section_from(R"({"base_rate": 0.5, "total_iterations": 10, "final_rate": 0.25})")));

    EXPECT_NEAR(read.base_rate, 0.5, K_RATE_TOLERANCE);
    EXPECT_EQ(read.total_iterations, 10);
    EXPECT_NEAR(read.final_rate, 0.25, K_RATE_TOLERANCE);
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
