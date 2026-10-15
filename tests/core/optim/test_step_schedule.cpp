/**************************************************************************/
/* test_step_schedule.cpp                                                 */
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
 * @file test_step_schedule.cpp
 * @brief Tests the learning-rate formulas of StepSchedule and the checks of its lists.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/step_schedule.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Optim::StepSchedule;
    using SushiAI::Optim::validate_step_schedule;

    /** @brief The base rate every case starts from. */
    constexpr double K_BASE_RATE = 0.001;

    /** @brief The warm-up length of the warm-up cases. */
    constexpr std::int64_t K_WARMUP = 1000;

    /** @brief Absolute part of the tolerance every rate comparison uses. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-15;

    /** @brief Relative part of the tolerance every rate comparison uses. */
    constexpr double K_RELATIVE_TOLERANCE = 1e-12;

    /**
     * @brief Expects the schedule's rate at @p iteration to equal @p expected within tolerance.
     * @param schedule The schedule under test.
     * @param iteration The update number passed to rate_at.
     * @param expected The rate the test computed in double.
     */
    void expect_rate(const StepSchedule& schedule, std::int64_t iteration, double expected)
    {
        const double tolerance = K_ABSOLUTE_TOLERANCE + K_RELATIVE_TOLERANCE * std::fabs(expected);
        EXPECT_NEAR(schedule.rate_at(iteration), expected, tolerance)
            << "at iteration " << iteration;
    }

    /** @brief A schedule with a warm-up of K_WARMUP updates at the default power 4. */
    StepSchedule warmup_schedule()
    {
        StepSchedule schedule;
        schedule.base_rate = K_BASE_RATE;
        schedule.warmup_iterations = K_WARMUP;
        return schedule;
    }

} // namespace

TEST(Small_StepSchedule, ConstantPolicyKeepsBaseRate)
{
    StepSchedule schedule;
    schedule.base_rate = K_BASE_RATE;

    expect_rate(schedule, 0, K_BASE_RATE);
    expect_rate(schedule, 1, K_BASE_RATE);
    expect_rate(schedule, 1000000, K_BASE_RATE);
}

TEST(Small_StepSchedule, WarmupRisesAsPowerOfIterationFraction)
{
    const StepSchedule schedule = warmup_schedule();

    // Iteration 0 is the state before any update, so the warm-up starts at zero.
    expect_rate(schedule, 0, 0.0);

    // The first update sees iteration 1.
    expect_rate(schedule, 1, K_BASE_RATE * std::pow(1.0 / 1000.0, 4.0));

    // Half the warm-up.
    expect_rate(schedule, 500, K_BASE_RATE * std::pow(500.0 / 1000.0, 4.0));

    // The last warm-up iteration.
    expect_rate(schedule, 999, K_BASE_RATE * std::pow(999.0 / 1000.0, 4.0));
}

TEST(Small_StepSchedule, WarmupEndsAtItsLengthWithTheBaseRate)
{
    const StepSchedule schedule = warmup_schedule();

    // The first iteration past the warm-up has no step to apply: the rate is the base rate.
    expect_rate(schedule, 1000, K_BASE_RATE);
    expect_rate(schedule, 1001, K_BASE_RATE);
}

TEST(Small_StepSchedule, WarmupPowerIsTheExponentOfTheRamp)
{
    StepSchedule linear = warmup_schedule();
    linear.warmup_power = 1.0;
    expect_rate(linear, 250, K_BASE_RATE * 0.25);

    StepSchedule quadratic = warmup_schedule();
    quadratic.warmup_power = 2.0;
    expect_rate(quadratic, 500, K_BASE_RATE * 0.25);
}

TEST(Small_StepSchedule, StepAppliesAtItsListedIteration)
{
    StepSchedule schedule;
    schedule.base_rate = K_BASE_RATE;
    schedule.steps = {400, 450};
    schedule.scales = {0.1, 0.1};

    expect_rate(schedule, 399, K_BASE_RATE);
    expect_rate(schedule, 400, K_BASE_RATE * 0.1);
    expect_rate(schedule, 401, K_BASE_RATE * 0.1);
    expect_rate(schedule, 449, K_BASE_RATE * 0.1);
    expect_rate(schedule, 450, K_BASE_RATE * 0.1 * 0.1);
    expect_rate(schedule, 451, K_BASE_RATE * 0.1 * 0.1);
}

TEST(Small_StepSchedule, WarmupReplacesThePolicyUntilItsLength)
{
    StepSchedule schedule = warmup_schedule();
    schedule.steps = {500};
    schedule.scales = {0.1};

    // The step at 500 falls inside the warm-up: the ramp is not multiplied by its scale.
    expect_rate(schedule, 500, K_BASE_RATE * std::pow(500.0 / 1000.0, 4.0));
    expect_rate(schedule, 999, K_BASE_RATE * std::pow(999.0 / 1000.0, 4.0));

    // Past the warm-up the policy resumes, and the step that fell inside it has compounded.
    expect_rate(schedule, 1000, K_BASE_RATE * 0.1);
}

TEST(Small_StepSchedule, StepAtFirstIterationPastWarmupApplies)
{
    StepSchedule schedule = warmup_schedule();
    schedule.steps = {1000};
    schedule.scales = {0.5};

    expect_rate(schedule, 999, K_BASE_RATE * std::pow(999.0 / 1000.0, 4.0));
    expect_rate(schedule, 1000, K_BASE_RATE * 0.5);
}

TEST(Small_StepSchedule, ValidateAcceptsEmptyAndAscendingSchedules)
{
    EXPECT_TRUE(validate_step_schedule(StepSchedule{}).is_ok());

    StepSchedule schedule = warmup_schedule();
    schedule.steps = {400, 450};
    schedule.scales = {0.1, 0.1};
    EXPECT_TRUE(validate_step_schedule(schedule).is_ok());
}

TEST(Small_StepSchedule, ValidateRefusesDescendingSteps)
{
    StepSchedule schedule;
    schedule.base_rate = K_BASE_RATE;
    schedule.steps = {450, 400};
    schedule.scales = {0.1, 0.1};

    EXPECT_EQ(validate_step_schedule(schedule).code(), StatusCode::InvalidDescription);
}

TEST(Small_StepSchedule, ValidateRefusesRepeatedStep)
{
    StepSchedule schedule;
    schedule.base_rate = K_BASE_RATE;
    schedule.steps = {400, 400};
    schedule.scales = {0.1, 0.1};

    EXPECT_EQ(validate_step_schedule(schedule).code(), StatusCode::InvalidDescription);
}

TEST(Small_StepSchedule, ValidateRefusesScalesLongerThanSteps)
{
    StepSchedule schedule;
    schedule.base_rate = K_BASE_RATE;
    schedule.steps = {400};
    schedule.scales = {0.1, 0.1};

    EXPECT_EQ(validate_step_schedule(schedule).code(), StatusCode::InvalidDescription);
}

TEST(Small_StepSchedule, ValidateRefusesStepsLongerThanScales)
{
    StepSchedule schedule;
    schedule.base_rate = K_BASE_RATE;
    schedule.steps = {400, 450};
    schedule.scales = {0.1};

    EXPECT_EQ(validate_step_schedule(schedule).code(), StatusCode::InvalidDescription);
}

TEST(Small_StepSchedule, ValidateRefusesNegativeWarmup)
{
    StepSchedule schedule = warmup_schedule();
    schedule.warmup_iterations = -1;

    EXPECT_EQ(validate_step_schedule(schedule).code(), StatusCode::InvalidDescription);
}

TEST(Small_StepSchedule, ValidateRefusesNegativeWarmupPower)
{
    StepSchedule schedule = warmup_schedule();
    schedule.warmup_power = -1.0;

    EXPECT_EQ(validate_step_schedule(schedule).code(), StatusCode::InvalidDescription);
}
