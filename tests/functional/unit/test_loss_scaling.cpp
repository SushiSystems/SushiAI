/****************************************************************/
/* test_loss_scaling.cpp                                        */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The dynamic-scaling rule, with no device in the measurement.
//
// LossScaleState is split out of LossScaler precisely so this file can exist:
// "the scale doubles after exactly N clean steps" is a property of the rule,
// not of a kernel, and asserting it through a training run would make every
// one of these cases cost a compile and a replay while proving less.
//
// The control matters as much as the case. A growth test that only ran N clean
// steps and saw a doubling would pass just as happily against an
// implementation that doubled after *every* step, so each growth assertion
// here is paired with the N-1 run that must not double.

#include <cstdint>
#include <gtest/gtest.h>
#include "../../common/test_throws.hpp"
#include "../../common/test_utils.hpp"

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
    const LossScaleState state{with_interval(4, 65536.0)};

    EXPECT_EQ(state.scale(), 65536.0);
    EXPECT_EQ(state.inverse_scale(), 1.0 / 65536.0);
    EXPECT_FALSE(state.last_step_skipped());
    EXPECT_EQ(state.skipped_steps(), 0u);
}

TEST_F(Unit_LossScaling, AnOverflowHalvesTheScaleAndSkipsTheStep)
{
    LossScaleState state{with_interval(4)};

    state.observe(true);

    EXPECT_EQ(state.scale(), 512.0);
    EXPECT_TRUE(state.last_step_skipped());
    EXPECT_EQ(state.skipped_steps(), 1u);
    EXPECT_EQ(state.backoffs(), 1u);
    EXPECT_EQ(state.growths(), 0u);
}

TEST_F(Unit_LossScaling, TheScaleDoublesAfterExactlyTheGrowthInterval)
{
    LossScaleState state{with_interval(4)};

    run_clean(state, 4);

    EXPECT_EQ(state.scale(), 2048.0);
    EXPECT_EQ(state.growths(), 1u);
    EXPECT_FALSE(state.last_step_skipped());
}

// The control for the case above. Without it, "the scale doubled after four
// clean steps" is equally true of an implementation that doubles after one.
TEST_F(Unit_LossScaling, OneStepShortOfTheIntervalTheScaleDoesNotMove)
{
    LossScaleState state{with_interval(4)};

    run_clean(state, 3);

    EXPECT_EQ(state.scale(), 1024.0);
    EXPECT_EQ(state.growths(), 0u);
    EXPECT_EQ(state.consecutive_good_steps(), 3u);
}

TEST_F(Unit_LossScaling, AnOverflowRestartsTheRunOfCleanSteps)
{
    LossScaleState state{with_interval(4)};

    run_clean(state, 3);
    state.observe(true);   // 1024 -> 512, and the three clean steps are spent
    run_clean(state, 3);

    // Three clean steps since the overflow is one short of the interval, so
    // nothing has grown back: the scale is still the backed-off one.
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
    LossScaleState state{options};

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
    LossScaleState state{options};

    state.observe(true);
    EXPECT_EQ(state.scale(), 1.0);

    // At the floor the scale cannot go lower - but the step is still garbage
    // and must still be thrown away, or an overflowed gradient reaches the
    // weights the moment the backoff bottoms out.
    state.observe(true);
    EXPECT_EQ(state.scale(), 1.0);
    EXPECT_TRUE(state.last_step_skipped());
    EXPECT_EQ(state.skipped_steps(), 2u);
    EXPECT_EQ(state.backoffs(), 1u);
}

// Every constant is a power of two so that scaling and unscaling change only
// the exponent field. A scale of 3 would work and would quietly make every
// gradient in the model carry a rounding error, so it is refused rather than
// accepted.
TEST_F(Unit_LossScaling, ANonDyadicConstantIsRefusedByName)
{
    LossScaleOptions options;
    options.initial = 3.0;
    SA_EXPECT_ERROR((LossScaleState{options}), "power of two");

    options = LossScaleOptions{};
    options.growth = 3.0;
    SA_EXPECT_ERROR((LossScaleState{options}), "power of two");

    options = LossScaleOptions{};
    options.backoff = 1.0 / 3.0;
    SA_EXPECT_ERROR((LossScaleState{options}), "power of two");
}

TEST_F(Unit_LossScaling, IncoherentBoundsAreRefused)
{
    LossScaleOptions options;
    options.initial = 65536.0;
    options.maximum = 1024.0;
    SA_EXPECT_ERROR((LossScaleState{options}), "outside the bounds");

    options = LossScaleOptions{};
    options.growth_interval = 0;
    SA_EXPECT_ERROR((LossScaleState{options}), "growth interval must be positive");
}

// The property the whole design is arranged around, asserted on the rule
// itself: whatever sequence of backoffs and growths the machine takes, the
// scale is a power of two, so the reciprocal is exact and the round trip is
// lossless.
TEST_F(Unit_LossScaling, EveryReachableScaleHasAnExactReciprocal)
{
    LossScaleState state{with_interval(2, 1024.0)};

    for (int i = 0; i < 40; ++i)
    {
        state.observe(i % 5 == 0);
        EXPECT_EQ(state.scale() * state.inverse_scale(), 1.0)
            << "scale " << state.scale() << " does not invert exactly";
    }
}
