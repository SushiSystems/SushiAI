/**************************************************************************/
/* test_run_schedule.cpp                                                  */
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
 * @file test_run_schedule.cpp
 * @brief Tests that a run reads its learning rate from one section only.
 * @author Mustafa Garip
 */

#include <optional>
#include <string_view>
#include <variant>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/run_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Optim::RateSchedule;
    using SushiAI::Optim::read_rate_schedule;
    using SushiAI::Optim::StepSchedule;

    /** @brief Bounds the error of a rate a section states and a double holds as read. */
    constexpr double K_RATE_TOLERANCE = 0.0;

    /** @brief Holds a schedule section with a rate, a warm-up and one step. */
    constexpr std::string_view K_SCHEDULE = R"({
        "type": "step",
        "base_rate": 0.00261,
        "warmup_iterations": 1000,
        "steps": [400000],
        "scales": [0.1]
    })";

    /** @brief Parses @p text as a section of a training file. */
    Value section_from(std::string_view text)
    {
        return require_value(SushiAI::Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that a file without a schedule yields none, whatever the optimizer states. */
TEST(Small_RunSchedule, ANullSectionYieldsNoSchedule)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::optional<RateSchedule> schedule,
        read_rate_schedule(Value(), section_from(R"({"learning_rate": 0.01})")));

    EXPECT_FALSE(schedule.has_value());
}

/** @brief Tests that a schedule beside an optimizer without a rate is read whole. */
TEST(Small_RunSchedule, AScheduleIsReadWhenTheOptimizerStatesNoRate)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::optional<RateSchedule> schedule,
        read_rate_schedule(section_from(K_SCHEDULE), section_from(R"({"momentum": 0.9})")));

    ASSERT_TRUE(schedule.has_value());
    ASSERT_TRUE(std::holds_alternative<StepSchedule>(*schedule));
    const StepSchedule& step = std::get<StepSchedule>(*schedule);
    EXPECT_NEAR(step.base_rate, 0.00261, K_RATE_TOLERANCE);
    EXPECT_EQ(step.warmup_iterations, 1000);
    ASSERT_EQ(step.steps.size(), 1u);
    EXPECT_EQ(step.steps[0], 400000);
}

/** @brief Tests that the rate stated in both sections is refused, equal values included. */
TEST(Small_RunSchedule, ARateStatedTwiceIsRefused)
{
    SA_EXPECT_FAILURE(read_rate_schedule(section_from(K_SCHEDULE),
                                        section_from(R"({"learning_rate": 0.00261})")),
                      StatusCode::InvalidDescription, "stated twice", "'learning_rate'",
                      "'base_rate'");
}

/** @brief Tests that a schedule that leaves its rate out is refused, not trained at zero. */
TEST(Small_RunSchedule, AScheduleWithoutABaseRateIsRefused)
{
    SA_EXPECT_FAILURE(read_rate_schedule(section_from(R"({"type": "step"})"),
                                        section_from("{}")),
                      StatusCode::InvalidDescription, "'base_rate'", "above zero");
}

/** @brief Tests that a base rate of zero or below is refused. */
TEST(Small_RunSchedule, ABaseRateNotAboveZeroIsRefused)
{
    SA_EXPECT_FAILURE(read_rate_schedule(section_from(R"({"base_rate": 0.0})"),
                                        section_from("{}")),
                      StatusCode::InvalidDescription, "'base_rate'", "above zero");
    SA_EXPECT_FAILURE(read_rate_schedule(section_from(R"({"base_rate": -0.01})"),
                                        section_from("{}")),
                      StatusCode::InvalidDescription, "'base_rate'", "above zero");
}

/** @brief Tests that a refusal of the section reader comes back unchanged. */
TEST(Small_RunSchedule, WhatTheSectionReaderRefusesIsRefused)
{
    SA_EXPECT_FAILURE(read_rate_schedule(section_from(R"({"base_rate": 0.1, "gamma": 2})"),
                                        section_from("{}")),
                      StatusCode::InvalidDescription, "schedule", "'gamma'");
    SA_EXPECT_FAILURE(read_rate_schedule(section_from(R"("step")"), section_from("{}")),
                      StatusCode::InvalidDescription, "schedule", "object");
}
