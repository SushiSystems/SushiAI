/**************************************************************************/
/* test_rate_schedule.cpp                                                 */
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
 * @file test_rate_schedule.cpp
 * @brief Tests that a schedule section is read by the type it states, and written back.
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
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/rate_schedule_section.hpp>
#include <SushiAI/optim/run_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Optim::CosineSchedule;
    using SushiAI::Optim::RateSchedule;
    using SushiAI::Optim::read_rate_schedule;
    using SushiAI::Optim::read_rate_schedule_section;
    using SushiAI::Optim::StepSchedule;
    using SushiAI::Optim::write_rate_schedule_section;

    /** @brief Holds a step section with a rate and one step. */
    constexpr std::string_view K_STEP_SECTION = R"({
        "type": "step",
        "base_rate": 0.5,
        "steps": [3],
        "scales": [0.5]
    })";

    /** @brief Holds a cosine section with a rate, a warm-up and a decay. */
    constexpr std::string_view K_COSINE_SECTION = R"({
        "type": "cosine",
        "base_rate": 0.25,
        "warmup_iterations": 2,
        "total_iterations": 10,
        "final_rate": 0.125
    })";

    /** @brief Parses @p text as a section of a training file. */
    Value section_from(std::string_view text)
    {
        return require_value(SushiAI::Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that "step" and "cosine" each read as the schedule of that type. */
TEST(Small_RateSchedule, ASectionIsReadByTheTypeItStates)
{
    SA_ASSERT_OK_AND_ASSIGN(const RateSchedule step,
                            read_rate_schedule_section(section_from(K_STEP_SECTION)));
    SA_ASSERT_OK_AND_ASSIGN(const RateSchedule cosine,
                            read_rate_schedule_section(section_from(K_COSINE_SECTION)));

    ASSERT_TRUE(std::holds_alternative<StepSchedule>(step));
    ASSERT_TRUE(std::holds_alternative<CosineSchedule>(cosine));
    EXPECT_EQ(std::get<StepSchedule>(step).base_rate, 0.5);
    EXPECT_EQ(std::get<CosineSchedule>(cosine).total_iterations, 10);
    EXPECT_EQ(std::get<CosineSchedule>(cosine).final_rate, 0.125);
}

/** @brief Tests that a section that states no type is read as a step schedule. */
TEST(Small_RateSchedule, ASectionWithoutATypeIsAStepSchedule)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const RateSchedule read,
        read_rate_schedule_section(section_from(R"({"base_rate": 0.5})")));

    EXPECT_TRUE(std::holds_alternative<StepSchedule>(read));
}

/** @brief Tests that a type no schedule has is refused with every type there is. */
TEST(Small_RateSchedule, AnUnknownTypeIsRefusedWithTheTypesThereAre)
{
    SA_EXPECT_FAILURE(
        read_rate_schedule_section(section_from(R"({"type": "linear", "base_rate": 0.5})")),
        StatusCode::InvalidDescription, "schedule 'type' is 'linear'", "'step'", "'cosine'");
    SA_EXPECT_FAILURE(read_rate_schedule_section(section_from(R"({"type": 3})")),
                      StatusCode::InvalidDescription, "schedule 'type' must be a string");
    SA_EXPECT_FAILURE(read_rate_schedule_section(section_from("7")),
                      StatusCode::InvalidDescription, "schedule must be an object");
}

/** @brief Tests that a key of one type is refused in a section of the other. */
TEST(Small_RateSchedule, AKeyOfAnotherTypeIsRefused)
{
    SA_EXPECT_FAILURE(
        read_rate_schedule_section(section_from(R"({"type": "cosine", "steps": [1]})")),
        StatusCode::InvalidDescription, "states the key 'steps'");
    SA_EXPECT_FAILURE(
        read_rate_schedule_section(section_from(R"({"type": "step", "final_rate": 0.1})")),
        StatusCode::InvalidDescription, "states the key 'final_rate'");
}

/** @brief Tests that the rate and the base rate of a held schedule are that schedule's. */
TEST(Small_RateSchedule, TheRateIsThatOfTheScheduleHeld)
{
    SA_ASSERT_OK_AND_ASSIGN(const RateSchedule step,
                            read_rate_schedule_section(section_from(K_STEP_SECTION)));
    SA_ASSERT_OK_AND_ASSIGN(const RateSchedule cosine,
                            read_rate_schedule_section(section_from(K_COSINE_SECTION)));

    EXPECT_EQ(SushiAI::Optim::rate_at(step, 2), 0.5);
    EXPECT_EQ(SushiAI::Optim::rate_at(step, 3), 0.25);
    EXPECT_EQ(SushiAI::Optim::rate_at(cosine, 1), 0.125);
    EXPECT_EQ(SushiAI::Optim::rate_at(cosine, 10), 0.125);
    EXPECT_EQ(SushiAI::Optim::base_rate_of(step), 0.5);
    EXPECT_EQ(SushiAI::Optim::base_rate_of(cosine), 0.25);
}

/** @brief Tests that a written section states the type of the schedule held and reads back. */
TEST(Small_RateSchedule, AWrittenSectionStatesItsTypeAndReadsBack)
{
    SA_ASSERT_OK_AND_ASSIGN(const RateSchedule cosine,
                            read_rate_schedule_section(section_from(K_COSINE_SECTION)));
    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_rate_schedule_section(cosine));
    SA_ASSERT_OK_AND_ASSIGN(const RateSchedule read, read_rate_schedule_section(written));

    EXPECT_EQ(require_value(require_value(written.at("type", "section")).as_string("type")),
              "cosine");
    ASSERT_TRUE(std::holds_alternative<CosineSchedule>(read));
    EXPECT_EQ(std::get<CosineSchedule>(read).warmup_iterations, 2);
}

/** @brief Tests that a run reads a cosine section as a cosine schedule. */
TEST(Small_RateSchedule, ARunReadsACosineSection)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::optional<RateSchedule> read,
        read_rate_schedule(section_from(K_COSINE_SECTION), section_from("{}")));

    ASSERT_TRUE(read.has_value());
    EXPECT_TRUE(std::holds_alternative<CosineSchedule>(*read));
}

/** @brief Tests that a held schedule is checked by the validation of its own type. */
TEST(Small_RateSchedule, AHeldScheduleIsCheckedByItsOwnValidation)
{
    StepSchedule uneven;
    uneven.base_rate = 0.5;
    uneven.steps = {2, 5};
    uneven.scales = {0.5};
    CosineSchedule cosine;
    cosine.base_rate = 0.5;
    cosine.total_iterations = 10;

    SA_EXPECT_FAILURE(SushiAI::Optim::validate_rate_schedule(RateSchedule(uneven)),
                      StatusCode::InvalidDescription, "2 steps", "1 scales");
    SA_EXPECT_OK(SushiAI::Optim::validate_rate_schedule(RateSchedule(cosine)));
}
