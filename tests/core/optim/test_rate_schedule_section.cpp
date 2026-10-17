/**************************************************************************/
/* test_rate_schedule_section.cpp                                         */
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
 * @file test_rate_schedule_section.cpp
 * @brief Tests that a schedule section of either type is written under its own type.
 * @author Mustafa Garip
 */

#include <limits>
#include <variant>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/cosine_schedule_section.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/rate_schedule_section.hpp>
#include <SushiAI/optim/step_schedule.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Optim::CosineSchedule;
    using SushiAI::Optim::RateSchedule;
    using SushiAI::Optim::read_rate_schedule_section;
    using SushiAI::Optim::StepSchedule;
    using SushiAI::Optim::write_rate_schedule_section;

    /** @brief Bounds the error of a rate a section states and a double holds as read. */
    constexpr double K_RATE_TOLERANCE = 0.0;
} // namespace

/** @brief Tests that a step schedule is written as "step" and reads back as a step schedule. */
TEST(Small_RateScheduleSection, AStepScheduleIsWrittenAsStep)
{
    StepSchedule schedule;
    schedule.base_rate = 0.5;
    schedule.steps = {3};
    schedule.scales = {0.5};

    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_rate_schedule_section(schedule));
    SA_ASSERT_OK_AND_ASSIGN(const RateSchedule read, read_rate_schedule_section(written));

    EXPECT_EQ(written.at("type", "section").as_string("type"), "step");
    ASSERT_TRUE(std::holds_alternative<StepSchedule>(read));
    EXPECT_NEAR(std::get<StepSchedule>(read).base_rate, 0.5, K_RATE_TOLERANCE);
    EXPECT_EQ(std::get<StepSchedule>(read).steps.size(), 1u);
}

/** @brief Tests that a cosine schedule is written under the cosine type. */
TEST(Small_RateScheduleSection, ACosineScheduleIsWrittenAsCosine)
{
    CosineSchedule schedule;
    schedule.base_rate = 0.25;
    schedule.total_iterations = 10;
    schedule.final_rate = 0.125;

    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_rate_schedule_section(schedule));

    EXPECT_EQ(written.at("type", "section").as_string("type"),
              SushiAI::Optim::K_COSINE_SCHEDULE_TYPE);
}

/** @brief Tests that the writer passes on the refusal of the type's own writer. */
TEST(Small_RateScheduleSection, TheWriterPassesOnTheTypesRefusal)
{
    StepSchedule schedule;
    schedule.base_rate = std::numeric_limits<double>::infinity();

    SA_EXPECT_FAILURE(write_rate_schedule_section(schedule), StatusCode::InvalidDescription,
                      "'base_rate'", "finite");
}
