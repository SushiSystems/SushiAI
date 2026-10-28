/**************************************************************************/
/* test_cosine_schedule.cpp                                               */
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
 * @file test_cosine_schedule.cpp
 * @brief Tests the rate a CosineSchedule gives each update, its section and its check.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/json_writer.hpp>
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
    using SushiAI::Optim::read_cosine_schedule_section;
    using SushiAI::Optim::validate_cosine_schedule;
    using SushiAI::Optim::write_cosine_schedule_section;

    /** @brief Bounds the error of a rate that is a few products of doubles of order one. */
    constexpr double K_RATE_TOLERANCE = 1e-15;

    /** @brief Returns a rate of 1 after 10 updates of warm-up that decays to 0.1 by 110. */
    CosineSchedule make_schedule()
    {
        CosineSchedule schedule;
        schedule.base_rate = 1.0;
        schedule.warmup_iterations = 10;
        schedule.total_iterations = 110;
        schedule.final_rate = 0.1;
        return schedule;
    }

    /** @brief Parses @p text as a schedule section. */
    Value section_from(std::string_view text)
    {
        return require_value(SushiAI::Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that the warm-up rises linearly from zero to the base rate. */
TEST(Small_CosineSchedule, TheWarmupIsLinear)
{
    const CosineSchedule schedule = make_schedule();

    EXPECT_NEAR(schedule.rate_at(0), 0.0, K_RATE_TOLERANCE);
    EXPECT_NEAR(schedule.rate_at(1), 0.1, K_RATE_TOLERANCE);
    EXPECT_NEAR(schedule.rate_at(5), 0.5, K_RATE_TOLERANCE);
    EXPECT_NEAR(schedule.rate_at(10), 1.0, K_RATE_TOLERANCE);
}

/** @brief Tests the decay at its quarter, half and three quarters, against the cosine. */
TEST(Small_CosineSchedule, TheDecayFollowsHalfACosine)
{
    const CosineSchedule schedule = make_schedule();

    // cos(pi / 4) is sqrt(2) / 2, so the weights are 0.853553..., 0.5 and 0.146446...
    constexpr double K_HALF_ROOT_TWO = 0.70710678118654752440;
    EXPECT_NEAR(schedule.rate_at(35), 0.1 + 0.9 * 0.5 * (1.0 + K_HALF_ROOT_TWO),
                K_RATE_TOLERANCE);
    EXPECT_NEAR(schedule.rate_at(60), 0.55, K_RATE_TOLERANCE);
    EXPECT_NEAR(schedule.rate_at(85), 0.1 + 0.9 * 0.5 * (1.0 - K_HALF_ROOT_TWO),
                K_RATE_TOLERANCE);
}

/** @brief Tests that the rate is the final one from the last iteration of the decay on. */
TEST(Small_CosineSchedule, TheRateStaysAtTheFinalOne)
{
    const CosineSchedule schedule = make_schedule();

    EXPECT_NEAR(schedule.rate_at(110), 0.1, K_RATE_TOLERANCE);
    EXPECT_NEAR(schedule.rate_at(1000000), 0.1, K_RATE_TOLERANCE);
}

/** @brief Tests that a schedule without a warm-up starts at the base rate. */
TEST(Small_CosineSchedule, WithoutAWarmupTheFirstRateIsTheBaseRate)
{
    CosineSchedule schedule = make_schedule();
    schedule.warmup_iterations = 0;

    EXPECT_NEAR(schedule.rate_at(0), 1.0, K_RATE_TOLERANCE);
    EXPECT_LT(schedule.rate_at(1), 1.0);
}

/** @brief Tests that a decay of no length steps from the base rate to the final one. */
TEST(Small_CosineSchedule, ADecayOfNoLengthDividesByNothing)
{
    CosineSchedule schedule = make_schedule();
    schedule.total_iterations = schedule.warmup_iterations;

    SA_EXPECT_OK(validate_cosine_schedule(schedule));
    EXPECT_NEAR(schedule.rate_at(9), 0.9, K_RATE_TOLERANCE);
    EXPECT_NEAR(schedule.rate_at(10), 0.1, K_RATE_TOLERANCE);
}

/** @brief Tests that a negative iteration stops the process. */
TEST(Small_CosineSchedule, ANegativeIterationIsFatal)
{
    const CosineSchedule schedule = make_schedule();

    SA_EXPECT_FATAL(static_cast<void>(schedule.rate_at(-1)), "iteration 0 or later");
}

/** @brief Tests each value the check refuses. */
TEST(Small_CosineSchedule, TheCheckRefusesWhatRateAtCannotUse)
{
    CosineSchedule negative_warmup = make_schedule();
    negative_warmup.warmup_iterations = -1;
    SA_EXPECT_FAILURE(validate_cosine_schedule(negative_warmup),
                      StatusCode::InvalidDescription, "warm-up", "negative");

    CosineSchedule short_decay = make_schedule();
    short_decay.total_iterations = 5;
    SA_EXPECT_FAILURE(validate_cosine_schedule(short_decay), StatusCode::InvalidDescription,
                      "ends at iteration 5", "warm-up of 10");

    CosineSchedule negative_final = make_schedule();
    negative_final.final_rate = -0.1;
    SA_EXPECT_FAILURE(validate_cosine_schedule(negative_final),
                      StatusCode::InvalidDescription, "final rate", "negative");

    CosineSchedule rising = make_schedule();
    rising.final_rate = 2.0;
    SA_EXPECT_FAILURE(validate_cosine_schedule(rising), StatusCode::InvalidDescription,
                      "above the base rate");
}

/** @brief Tests that a schedule survives the writer, JSON text, the parser and the reader. */
TEST(Small_CosineSchedule, ASectionRoundTripsEveryField)
{
    CosineSchedule schedule = make_schedule();
    schedule.base_rate = 0.0006;
    schedule.final_rate = 0.00006;

    SA_ASSERT_OK_AND_ASSIGN(const Value section, write_cosine_schedule_section(schedule));
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, SushiAI::Config::write_json(section));
    SA_ASSERT_OK_AND_ASSIGN(const CosineSchedule read,
                            read_cosine_schedule_section(section_from(text)));

    EXPECT_EQ(require_value(require_value(section.at("type", "section")).as_string("type")),
              "cosine");
    EXPECT_EQ(read.base_rate, schedule.base_rate);
    EXPECT_EQ(read.warmup_iterations, schedule.warmup_iterations);
    EXPECT_EQ(read.total_iterations, schedule.total_iterations);
    EXPECT_EQ(read.final_rate, schedule.final_rate);
}

/** @brief Tests that a key left out keeps the default of its field. */
TEST(Small_CosineSchedule, AnEmptySectionReadsAsTheDefaults)
{
    SA_ASSERT_OK_AND_ASSIGN(const CosineSchedule read,
                            read_cosine_schedule_section(section_from("{}")));

    EXPECT_EQ(read.base_rate, 0.0);
    EXPECT_EQ(read.warmup_iterations, 0);
    EXPECT_EQ(read.total_iterations, 0);
    EXPECT_EQ(read.final_rate, 0.0);
}

/** @brief Tests each section the reader refuses, and that the refusal names the key. */
TEST(Small_CosineSchedule, TheReaderRefusesAndNamesTheKey)
{
    SA_EXPECT_FAILURE(read_cosine_schedule_section(section_from(R"({"type": "step"})")),
                      StatusCode::InvalidDescription, "schedule 'type'", "'step'",
                      "a cosine schedule states 'cosine'");
    SA_EXPECT_FAILURE(read_cosine_schedule_section(section_from(R"({"steps": [1]})")),
                      StatusCode::InvalidDescription, "schedule states the key 'steps'",
                      "'total_iterations'");
    SA_EXPECT_FAILURE(
        read_cosine_schedule_section(section_from(R"({"total_iterations": 1.5})")),
        StatusCode::InvalidDescription, "schedule 'total_iterations'", "whole number");
    SA_EXPECT_FAILURE(
        read_cosine_schedule_section(section_from(R"({"base_rate": 1.0, "final_rate": 2.0})")),
        StatusCode::InvalidDescription, "schedule: ", "above the base rate");
    SA_EXPECT_FAILURE(read_cosine_schedule_section(section_from("[]")),
                      StatusCode::InvalidDescription, "schedule must be an object");
}

/** @brief Tests that the writer refuses a rate JSON cannot hold. */
TEST(Small_CosineSchedule, TheWriterRefusesARateThatIsNotFinite)
{
    CosineSchedule schedule = make_schedule();
    schedule.base_rate = std::numeric_limits<double>::infinity();

    SA_EXPECT_FAILURE(write_cosine_schedule_section(schedule), StatusCode::InvalidDescription,
                      "schedule 'base_rate'", "finite");
}
