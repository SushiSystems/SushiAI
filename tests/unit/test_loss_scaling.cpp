/**************************************************************************/
/* test_loss_scaling.cpp                                                  */
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
 * @file test_loss_scaling.cpp
 * @brief Tests the dynamic loss-scaling rule on LossScaleState, with no device involved.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

using Optim::LossScaleOptions;
using Optim::LossScaleState;

namespace
{
    class Unit_LossScaling : public ::testing::Test {};

    /** @brief Options with a growth interval small enough to reach in a test. */
    LossScaleOptions with_interval(std::uint64_t interval, double initial = 1024.0)
    {
        LossScaleOptions options;
        options.initial = initial;
        options.growth_interval = interval;
        return options;
    }

    /** @brief Runs @p steps clean steps through @p state. */
    void run_clean(LossScaleState& state, std::uint64_t steps)
    {
        for (std::uint64_t i = 0; i < steps; ++i) state.observe(false);
    }
}

TEST_F(Unit_LossScaling, TheScaleStartsWhereItWasAsked)
{
    SA_ASSERT_OK_AND_ASSIGN(const LossScaleState state,
                            LossScaleState::make(with_interval(4, 65536.0)));

    EXPECT_EQ(state.scale(), 65536.0);
    EXPECT_EQ(state.inverse_scale(), 1.0 / 65536.0);
    EXPECT_FALSE(state.last_step_skipped());
    EXPECT_EQ(state.skipped_steps(), 0u);
}

TEST_F(Unit_LossScaling, AnOverflowHalvesTheScaleAndSkipsTheStep)
{
    SA_ASSERT_OK_AND_ASSIGN(LossScaleState state, LossScaleState::make(with_interval(4)));

    state.observe(true);

    EXPECT_EQ(state.scale(), 512.0);
    EXPECT_TRUE(state.last_step_skipped());
    EXPECT_EQ(state.skipped_steps(), 1u);
    EXPECT_EQ(state.backoffs(), 1u);
    EXPECT_EQ(state.growths(), 0u);
}

TEST_F(Unit_LossScaling, TheScaleDoublesAfterExactlyTheGrowthInterval)
{
    SA_ASSERT_OK_AND_ASSIGN(LossScaleState state, LossScaleState::make(with_interval(4)));

    run_clean(state, 4);

    EXPECT_EQ(state.scale(), 2048.0);
    EXPECT_EQ(state.growths(), 1u);
    EXPECT_FALSE(state.last_step_skipped());
}

// Controls the case above with one clean step fewer; see tests/unit/README.md.
TEST_F(Unit_LossScaling, OneStepShortOfTheIntervalTheScaleDoesNotMove)
{
    SA_ASSERT_OK_AND_ASSIGN(LossScaleState state, LossScaleState::make(with_interval(4)));

    run_clean(state, 3);

    EXPECT_EQ(state.scale(), 1024.0);
    EXPECT_EQ(state.growths(), 0u);
    EXPECT_EQ(state.consecutive_good_steps(), 3u);
}

TEST_F(Unit_LossScaling, AnOverflowRestartsTheRunOfCleanSteps)
{
    SA_ASSERT_OK_AND_ASSIGN(LossScaleState state, LossScaleState::make(with_interval(4)));

    run_clean(state, 3);
    state.observe(true);   // 1024 -> 512, and the three clean steps are spent
    run_clean(state, 3);

    // Three clean steps are one short of the interval; see tests/unit/README.md.
    EXPECT_EQ(state.scale(), 512.0);
    EXPECT_EQ(state.growths(), 0u);

    state.observe(false);
    EXPECT_EQ(state.scale(), 1024.0);
    EXPECT_EQ(state.growths(), 1u);
}

TEST_F(Unit_LossScaling, GrowthStopsAtTheCeilingWithoutCountingAGrowth)
{
    LossScaleOptions options = with_interval(1, 1024.0);
    options.maximum = 2048.0;
    SA_ASSERT_OK_AND_ASSIGN(LossScaleState state, LossScaleState::make(options));

    state.observe(false);
    EXPECT_EQ(state.scale(), 2048.0);

    state.observe(false);
    EXPECT_EQ(state.scale(), 2048.0);
    EXPECT_EQ(state.growths(), 1u);
}

TEST_F(Unit_LossScaling, BackoffStopsAtTheFloorButStillSkipsTheStep)
{
    LossScaleOptions options = with_interval(4, 2.0);
    options.minimum = 1.0;
    SA_ASSERT_OK_AND_ASSIGN(LossScaleState state, LossScaleState::make(options));

    state.observe(true);
    EXPECT_EQ(state.scale(), 1.0);

    // At the floor the step is still skipped; see tests/unit/README.md.
    state.observe(true);
    EXPECT_EQ(state.scale(), 1.0);
    EXPECT_TRUE(state.last_step_skipped());
    EXPECT_EQ(state.skipped_steps(), 2u);
    EXPECT_EQ(state.backoffs(), 1u);
}

// A constant that is not a power of two is refused; see tests/unit/README.md.
TEST_F(Unit_LossScaling, ANonDyadicConstantIsRefusedByName)
{
    LossScaleOptions options;
    options.initial = 3.0;
    SA_EXPECT_FAILURE(LossScaleState::make(options), StatusCode::InvalidDescription,
                      "power of two");

    options = LossScaleOptions{};
    options.growth = 3.0;
    SA_EXPECT_FAILURE(LossScaleState::make(options), StatusCode::InvalidDescription,
                      "power of two");

    options = LossScaleOptions{};
    options.backoff = 1.0 / 3.0;
    SA_EXPECT_FAILURE(LossScaleState::make(options), StatusCode::InvalidDescription,
                      "power of two");
}

TEST_F(Unit_LossScaling, IncoherentBoundsAreRefused)
{
    LossScaleOptions options;
    options.initial = 65536.0;
    options.maximum = 1024.0;
    SA_EXPECT_FAILURE(LossScaleState::make(options), StatusCode::InvalidDescription,
                      "outside the bounds");

    options = LossScaleOptions{};
    options.growth_interval = 0;
    SA_EXPECT_FAILURE(LossScaleState::make(options), StatusCode::InvalidDescription,
                      "growth interval must be positive");
}

// Every reachable scale is a power of two; see tests/unit/README.md.
TEST_F(Unit_LossScaling, EveryReachableScaleHasAnExactReciprocal)
{
    SA_ASSERT_OK_AND_ASSIGN(LossScaleState state, LossScaleState::make(with_interval(2, 1024.0)));

    for (int i = 0; i < 40; ++i)
    {
        state.observe(i % 5 == 0);
        EXPECT_EQ(state.scale() * state.inverse_scale(), 1.0)
            << "scale " << state.scale() << " does not invert exactly";
    }
}
