/**************************************************************************/
/* test_step_schedule_section.cpp                                         */
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
 * @file test_step_schedule_section.cpp
 * @brief Tests that a StepSchedule and the schedule section of a training file convert.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/optim/step_schedule_section.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Optim::K_STEP_SCHEDULE_TYPE;
    using SushiAI::Optim::read_step_schedule_section;
    using SushiAI::Optim::StepSchedule;
    using SushiAI::Optim::write_step_schedule_section;

    /** @brief Holds a section that states every key, as the writer orders them. */
    constexpr std::string_view K_FULL_SECTION = R"({
        "type": "step",
        "base_rate": 0.00261,
        "warmup_iterations": 1000,
        "warmup_power": 4.0,
        "steps": [400000, 450000],
        "scales": [0.1, 0.1]
    })";

    /** @brief Parses @p text as a schedule section. */
    Value section_from(std::string_view text)
    {
        return SushiAI::Config::parse_json(text, "<test>");
    }

    /** @brief Parses the section that states the one member @p member, as JSON text. */
    Value section_of(std::string_view member)
    {
        return section_from("{" + std::string(member) + "}");
    }

    /** @brief Returns a warm-up of 1000 updates and two steps, with no default left. */
    StepSchedule make_two_step_schedule()
    {
        StepSchedule schedule;
        schedule.base_rate = 0.00261;
        schedule.warmup_iterations = 1000;
        schedule.warmup_power = 2.5;
        schedule.steps = {400000, 450000};
        schedule.scales = {0.1, 0.3};
        return schedule;
    }

    /**
     * @brief Expects every field of @p found to equal that of @p wanted.
     *
     * A double is written in the shortest form that reads back to it, so the tolerance is
     * zero, absolute and relative.
     */
    void expect_same_schedule(const StepSchedule& found, const StepSchedule& wanted)
    {
        EXPECT_EQ(found.base_rate, wanted.base_rate);
        EXPECT_EQ(found.warmup_iterations, wanted.warmup_iterations);
        EXPECT_EQ(found.warmup_power, wanted.warmup_power);
        EXPECT_EQ(found.steps, wanted.steps);
        EXPECT_EQ(found.scales, wanted.scales);
    }

    /** @brief Expects @p schedule to survive the writer, JSON text, the parser and the reader. */
    void expect_schedule_round_trip(const StepSchedule& schedule)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Value section, write_step_schedule_section(schedule));
        SA_ASSERT_OK_AND_ASSIGN(const std::string text, SushiAI::Config::write_json(section));
        SA_ASSERT_OK_AND_ASSIGN(const StepSchedule read,
                                read_step_schedule_section(section_from(text)));

        expect_same_schedule(read, schedule);
    }
} // namespace

/** @brief Tests that the default schedule reads back from its section unchanged. */
TEST(Unit_StepScheduleSection, TheDefaultsRoundTripThroughTheirSection)
{
    expect_schedule_round_trip(StepSchedule{});
}

/** @brief Tests that a warm-up and two steps read back from their section unchanged. */
TEST(Unit_StepScheduleSection, AWarmupAndTwoStepsRoundTripThroughTheirSection)
{
    expect_schedule_round_trip(make_two_step_schedule());
}

/** @brief Tests that a section stating every key is written back as it was read. */
TEST(Unit_StepScheduleSection, AFullSectionRoundTripsThroughItsSchedule)
{
    const Value section = section_from(K_FULL_SECTION);
    SA_ASSERT_OK_AND_ASSIGN(const StepSchedule schedule, read_step_schedule_section(section));
    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_step_schedule_section(schedule));

    SA_ASSERT_OK_AND_ASSIGN(const std::string found, SushiAI::Config::write_json(written));
    SA_ASSERT_OK_AND_ASSIGN(const std::string wanted, SushiAI::Config::write_json(section));
    EXPECT_EQ(found, wanted);
}

/** @brief Tests that the reader fills every field of the schedule from the section. */
TEST(Unit_StepScheduleSection, ReadsEveryFieldOfASection)
{
    SA_ASSERT_OK_AND_ASSIGN(const StepSchedule schedule,
                            read_step_schedule_section(section_from(K_FULL_SECTION)));

    StepSchedule wanted;
    wanted.base_rate = 0.00261;
    wanted.warmup_iterations = 1000;
    wanted.warmup_power = 4.0;
    wanted.steps = {400000, 450000};
    wanted.scales = {0.1, 0.1};
    expect_same_schedule(schedule, wanted);
}

/** @brief Tests that a section with no key, or with its type alone, gives the defaults. */
TEST(Unit_StepScheduleSection, AKeyLeftOutKeepsTheDefaultOfItsField)
{
    SA_ASSERT_OK_AND_ASSIGN(const StepSchedule empty,
                            read_step_schedule_section(section_from("{}")));
    expect_same_schedule(empty, StepSchedule{});

    SA_ASSERT_OK_AND_ASSIGN(const StepSchedule typed,
                            read_step_schedule_section(section_of(R"("type": "step")")));
    expect_same_schedule(typed, StepSchedule{});

    SA_ASSERT_OK_AND_ASSIGN(const StepSchedule rated,
                            read_step_schedule_section(section_of(R"("base_rate": 0.01)")));
    StepSchedule wanted;
    wanted.base_rate = 0.01;
    expect_same_schedule(rated, wanted);
}

/** @brief Tests that the writer states the type and every key in the order of the fields. */
TEST(Unit_StepScheduleSection, WritesTheTypeAndEveryKeyInOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value section,
                            write_step_schedule_section(make_two_step_schedule()));

    EXPECT_EQ(section.keys(),
              (std::vector<std::string>{"type", "base_rate", "warmup_iterations",
                                        "warmup_power", "steps", "scales"}));
    EXPECT_EQ(section.at("type", "schedule").as_string("type"), K_STEP_SCHEDULE_TYPE);
    EXPECT_EQ(section.at("warmup_iterations", "schedule").as_int("warmup_iterations"), 1000);
    EXPECT_EQ(section.at("steps", "schedule").as_array("steps")[1].as_int("step"), 450000);
}

/** @brief Tests that a whole number is read where the section holds a real. */
TEST(Unit_StepScheduleSection, ReadsAWholeNumberAsARateAPowerAndAScale)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const StepSchedule schedule,
        read_step_schedule_section(section_from(
            R"({"base_rate": 1, "warmup_power": 2, "steps": [10], "scales": [3]})")));

    EXPECT_EQ(schedule.base_rate, 1.0);
    EXPECT_EQ(schedule.warmup_power, 2.0);
    EXPECT_EQ(schedule.scales, (std::vector<double>{3.0}));
}

/** @brief Tests that a section that is no object is refused with the kind it is. */
TEST(Unit_StepScheduleSection, RefusesASectionThatIsNotAnObject)
{
    SA_EXPECT_FAILURE(read_step_schedule_section(section_from("[]")),
                      StatusCode::InvalidDescription, "schedule", "an object");
    SA_EXPECT_FAILURE(read_step_schedule_section(Value()), StatusCode::InvalidDescription,
                      "schedule", "an object");
}

/** @brief Tests that a type of another schedule or of another kind is refused. */
TEST(Unit_StepScheduleSection, RefusesAnotherType)
{
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("type": "cosine")")),
                      StatusCode::InvalidDescription, "schedule 'type'", "'cosine'", "'step'");
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("type": 1)")),
                      StatusCode::InvalidDescription, "schedule 'type'", "a string");
}

/** @brief Tests that a key outside the section is refused with the keys it accepts. */
TEST(Unit_StepScheduleSection, RefusesAnUnknownKey)
{
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("burn_in": 1000)")),
                      StatusCode::InvalidDescription, "schedule", "'burn_in'",
                      "'type', 'base_rate'", "'scales'");
}

/** @brief Tests that a value of another kind is refused with its key and its entry. */
TEST(Unit_StepScheduleSection, RefusesAValueOfAnotherKind)
{
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("base_rate": "fast")")),
                      StatusCode::InvalidDescription, "schedule 'base_rate'", "a number");
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("warmup_iterations": 1.5)")),
                      StatusCode::InvalidDescription, "schedule 'warmup_iterations'",
                      "a whole number");
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("steps": 100)")),
                      StatusCode::InvalidDescription, "schedule 'steps'", "an array");
    SA_EXPECT_FAILURE(
        read_step_schedule_section(section_from(R"({"steps": [10, 20.5], "scales": [1, 1]})")),
        StatusCode::InvalidDescription, "schedule 'steps' entry 1", "a whole number");
    SA_EXPECT_FAILURE(
        read_step_schedule_section(section_from(R"({"steps": [10], "scales": ["half"]})")),
        StatusCode::InvalidDescription, "schedule 'scales' entry 0", "a number");
}

/** @brief Tests that the reader refuses the schedules validate_step_schedule refuses. */
TEST(Unit_StepScheduleSection, RefusesTheSchedulesTheScheduleCheckRefuses)
{
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("steps": [100, 200])")),
                      StatusCode::InvalidDescription, "schedule", "2 steps", "0 scales");
    SA_EXPECT_FAILURE(
        read_step_schedule_section(section_from(R"({"steps": [200, 100], "scales": [1, 1]})")),
        StatusCode::InvalidDescription, "schedule", "strictly ascend");
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("warmup_iterations": -1)")),
                      StatusCode::InvalidDescription, "schedule", "-1", "negative");
    SA_EXPECT_FAILURE(read_step_schedule_section(section_of(R"("warmup_power": -2.0)")),
                      StatusCode::InvalidDescription, "schedule", "-2", "negative");
}

/** @brief Tests that the writer refuses a number JSON cannot hold, with its key. */
TEST(Unit_StepScheduleSection, WriterRefusesANumberThatIsNotFinite)
{
    StepSchedule rated = make_two_step_schedule();
    rated.base_rate = std::numeric_limits<double>::infinity();
    SA_EXPECT_FAILURE(write_step_schedule_section(rated), StatusCode::InvalidDescription,
                      "schedule 'base_rate'", "finite");

    StepSchedule powered = make_two_step_schedule();
    powered.warmup_power = std::numeric_limits<double>::infinity();
    SA_EXPECT_FAILURE(write_step_schedule_section(powered), StatusCode::InvalidDescription,
                      "schedule 'warmup_power'", "finite");

    StepSchedule scaled = make_two_step_schedule();
    scaled.scales[1] = std::numeric_limits<double>::quiet_NaN();
    SA_EXPECT_FAILURE(write_step_schedule_section(scaled), StatusCode::InvalidDescription,
                      "schedule 'scales' entry 1", "finite");
}

/** @brief Tests that the writer refuses the schedules validate_step_schedule refuses. */
TEST(Unit_StepScheduleSection, WriterRefusesTheSchedulesTheScheduleCheckRefuses)
{
    StepSchedule uneven = make_two_step_schedule();
    uneven.scales.pop_back();
    SA_EXPECT_FAILURE(write_step_schedule_section(uneven), StatusCode::InvalidDescription,
                      "schedule", "2 steps", "1 scales");

    StepSchedule descending = make_two_step_schedule();
    descending.steps = {450000, 400000};
    SA_EXPECT_FAILURE(write_step_schedule_section(descending), StatusCode::InvalidDescription,
                      "schedule", "strictly ascend");
}
