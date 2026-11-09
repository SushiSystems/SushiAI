/**************************************************************************/
/* test_random.cpp                                                        */
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
 * @file test_random.cpp
 * @brief Tests the imaging generator's SplitMix64 sequence and its float and index mappings.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/random.hpp>
#include <support/test_names.hpp>

using SushiAI::Vision::Imaging::Random;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the seed of every statistical case in this file. */
    constexpr std::uint64_t K_SEED = 2026;

    /** @brief Holds the sample count of the statistical cases, 2^16. */
    constexpr std::int32_t K_SAMPLES = 65536;

    /** @brief Holds 2^24, the number of values uniform() can return. */
    constexpr float K_UNIFORM_VALUES = 16777216.0f;

    /** @brief Holds the chance that a correct run of draws still misses a statistical bound. */
    constexpr double K_FALSE_FAILURE_RATE = 1e-9;

    /**
     * @brief Returns the distance from its expectation that a mean of draws in [0, 1] stays within.
     *
     * Hoeffding's inequality gives P(|mean - expectation| >= t) <= 2 exp(-2 n t^2), so t is
     * sqrt(ln(2 / rate) / (2 n)) for n draws and K_FALSE_FAILURE_RATE. It bounds a mean of
     * uniform draws, a success rate and, times n, a bucket count.
     */
    double mean_tolerance(std::int32_t draws)
    {
        return std::sqrt(std::log(2.0 / K_FALSE_FAILURE_RATE) / (2.0 * static_cast<double>(draws)));
    }

    /**
     * @brief Returns the distance from 1/12 that the variance of K_SAMPLES uniform draws stays
     *        within.
     *
     * The variance is the mean of the squares minus the square of the mean. Both means stay
     * within t of their expectation, and the square of a mean near 1/2 moves by at most
     * t + t^2, so the variance moves by at most 2 t + t^2.
     */
    double variance_tolerance()
    {
        const double t = mean_tolerance(K_SAMPLES);
        return 2.0 * t + t * t;
    }

    /** @brief Holds the mean and the variance of a run of draws. */
    struct Moments
    {
        /** @brief Holds the mean of the draws. */
        double mean;

        /** @brief Holds the variance of the draws. */
        double variance;
    };

    /** @brief Draws K_SAMPLES values of uniform() and returns their mean and variance. */
    Moments measure_uniform_moments(Random& random)
    {
        double sum = 0.0;
        double sum_of_squares = 0.0;
        for (std::int32_t i = 0; i < K_SAMPLES; ++i)
        {
            const double value = static_cast<double>(random.uniform());
            sum += value;
            sum_of_squares += value * value;
        }

        const double mean = sum / K_SAMPLES;
        return Moments{mean, sum_of_squares / K_SAMPLES - mean * mean};
    }

    /** @brief Draws @p draws values of index(7) and returns the hit count of each bucket. */
    std::array<std::int32_t, 7> count_index_hits(Random& random, std::int32_t draws)
    {
        std::array<std::int32_t, 7> hits{};
        for (std::int32_t i = 0; i < draws; ++i)
            ++hits[random.index(static_cast<std::uint32_t>(hits.size()))];

        return hits;
    }

    /** @brief Draws @p draws values of bernoulli(probability) and returns the success count. */
    std::int32_t count_successes(Random& random, float probability, std::int32_t draws)
    {
        std::int32_t successes = 0;
        for (std::int32_t i = 0; i < draws; ++i)
            successes += random.bernoulli(probability) ? 1 : 0;

        return successes;
    }

    /** @brief Draws @p draws values of next_u64 and returns them in draw order. */
    std::vector<std::uint64_t> draw_values(Random& random, std::size_t draws)
    {
        std::vector<std::uint64_t> values;
        for (std::size_t i = 0; i < draws; ++i)
            values.push_back(random.next_u64());

        return values;
    }

    /** @brief Draws @p draws values of index(1) and returns them in draw order. */
    std::vector<std::uint32_t> draw_indices_of_one(Random& random, std::size_t draws)
    {
        std::vector<std::uint32_t> values;
        for (std::size_t i = 0; i < draws; ++i)
            values.push_back(random.index(1));

        return values;
    }

    /** @brief Advances a generator by @p draws values of next_u64. */
    void skip_draws(Random& random, std::int32_t draws)
    {
        for (std::int32_t i = 0; i < draws; ++i)
            static_cast<void>(random.next_u64());
    }

    /** @brief Describes a value and the SplitMix64 finaliser of it, worked out by hand. */
    struct MixRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the value mixed. */
        std::uint64_t value;

        /** @brief Holds the finaliser of the value. */
        std::uint64_t result;
    };

    /** @brief Runs each hand-computed finaliser case. */
    class Small_ImagingRandomMix : public ::testing::TestWithParam<MixRow>
    {
    };

    /** @brief Describes the draw of a half-open interval and the bounds it must stay within. */
    struct IntervalRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Draws one value of the interval. */
        float (*draw)(Random&);

        /** @brief Holds the inclusive lower bound. */
        float low;

        /** @brief Holds the exclusive upper bound. */
        float high;
    };

    /** @brief Runs each half-open interval case. */
    class Small_ImagingRandomInterval : public ::testing::TestWithParam<IntervalRow>
    {
    };

    /** @brief Draws K_SAMPLES values and checks each is in [low, high). */
    void expect_draws_in_interval(const IntervalRow& row)
    {
        Random random(K_SEED);
        for (std::int32_t i = 0; i < K_SAMPLES; ++i)
        {
            const float value = row.draw(random);

            ASSERT_GE(value, row.low) << "draw " << i;
            ASSERT_LT(value, row.high) << "draw " << i;
        }
    }
} // namespace

/**
 * @brief Tests that mix is the SplitMix64 finaliser.
 *
 * Each result is z = (v ^ v >> 30) * K_FIRST_MULTIPLIER, then z = (z ^ z >> 27) *
 * K_SECOND_MULTIPLIER, then z ^ z >> 31, worked out by hand for the row's value.
 */
TEST_P(Small_ImagingRandomMix, MixFollowsTheFinaliser)
{
    const MixRow& row = GetParam();

    EXPECT_EQ(Random::mix(row.value), row.result);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingRandomMix,
    ::testing::Values(MixRow{"TheFirstIncrement", 0x9E3779B97F4A7C15ull, 0xE220A8397B1DCDAFull},
                      MixRow{"TheSecondCounter", 0x3C6EF372FE94F82Aull, 0x6E789E6AA1B965F4ull}),
    [](const ::testing::TestParamInfo<MixRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

TEST(Small_ImagingRandom, SeedZeroReproducesTheHandComputedSequence)
{
    Random random(0);

    // The n-th counter of seed 0 is n * K_INCREMENT modulo 2^64.
    EXPECT_EQ(random.next_u64(), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(random.state(), 0x9E3779B97F4A7C15ull);

    EXPECT_EQ(random.next_u64(), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(random.state(), 0x3C6EF372FE94F82Aull);

    EXPECT_EQ(random.next_u64(), 0x06C45D188009454Full);
    EXPECT_EQ(random.state(), 0xDAA66D2C7DDF743Full);

    EXPECT_EQ(random.next_u64(), 0xF88BB8A8724C81ECull);
}

TEST(Small_ImagingRandom, TheCounterStartsAtTheSeed)
{
    EXPECT_EQ(Random(0).state(), 0u);
    EXPECT_EQ(Random(42).state(), 42u);

    Random random(42);
    EXPECT_EQ(random.next_u64(), Random::mix(42 + Random::K_INCREMENT));
    EXPECT_EQ(random.state(), 42 + Random::K_INCREMENT);
}

TEST(Small_ImagingRandom, NextU32IsTheHighHalfOfOneDraw)
{
    Random random(0);

    EXPECT_EQ(random.next_u32(), 0xE220A839u);
    EXPECT_EQ(random.next_u32(), 0x6E789E6Au);
    EXPECT_EQ(random.state(), 0x3C6EF372FE94F82Aull);
}

TEST(Small_ImagingRandom, UniformIsTheTop24BitsScaledByTwoToTheMinus24)
{
    Random random(0);

    // 0xE220A8397B1DCDAF >> 40 is 0xE220A8, which is 14819496.
    EXPECT_EQ(random.uniform(), 14819496.0f / K_UNIFORM_VALUES);

    // 0x6E789E6AA1B965F4 >> 40 is 0x6E789E, which is 7239838.
    EXPECT_EQ(random.uniform(), 7239838.0f / K_UNIFORM_VALUES);

    // 0x06C45D188009454F >> 40 is 0x06C45D, which is 443485.
    EXPECT_EQ(random.uniform(), 443485.0f / K_UNIFORM_VALUES);

    EXPECT_EQ(random.state(), 0xDAA66D2C7DDF743Full);
}

/** @brief Tests that uniform draws stay in the half-open interval they are asked for. */
TEST_P(Small_ImagingRandomInterval, UniformStaysInItsHalfOpenInterval)
{
    expect_draws_in_interval(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingRandomInterval,
    ::testing::Values(
        IntervalRow{"TheUnitInterval", [](Random& random) { return random.uniform(); }, 0.0f,
                    1.0f},
        IntervalRow{"ARangedInterval", [](Random& random) { return random.uniform(0.4f, 0.6f); },
                    0.4f, 0.6f}),
    [](const ::testing::TestParamInfo<IntervalRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

/** @brief Tests that the mean and variance of uniform draws are those of a uniform variable. */
TEST(Small_ImagingRandom, UniformHasTheMeanAndVarianceOfAUniformVariable)
{
    Random random(K_SEED);

    const Moments moments = measure_uniform_moments(random);

    EXPECT_NEAR(moments.mean, 0.5, mean_tolerance(K_SAMPLES));
    EXPECT_NEAR(moments.variance, 1.0 / 12.0, variance_tolerance());
}

TEST(Small_ImagingRandom, RangedUniformMapsTheUnitDrawLinearly)
{
    Random random(0);

    // -1 + 2 * 14819496 / 2^24 is 12861776 / 2^24, exact in a float.
    EXPECT_EQ(random.uniform(-1.0f, 1.0f), 12861776.0f / K_UNIFORM_VALUES);

    // 0.5 + 1 * 7239838 / 2^24 is 15628446 / 2^24, exact in a float.
    EXPECT_EQ(random.uniform(0.5f, 1.5f), 15628446.0f / K_UNIFORM_VALUES);

    EXPECT_EQ(random.state(), 0x3C6EF372FE94F82Aull);
}

/** @brief Tests that a ranged draw over an interval of one float returns its lower bound. */
TEST(Small_ImagingRandom, RangedUniformNeverReturnsItsUpperBound)
{
    const float low = 1.0f;
    const float high = std::nextafter(1.0f, 2.0f);

    Random random(K_SEED);

    // The interval holds one float, so every sum that rounds up has to come back to low.
    for (std::int32_t i = 0; i < 1024; ++i)
        ASSERT_EQ(random.uniform(low, high), low) << "draw " << i;
}

TEST(Small_ImagingRandom, RangedUniformOverAnEmptyRangeStillDraws)
{
    Random random(K_SEED);
    Random reference(K_SEED);

    EXPECT_EQ(random.uniform(0.25f, 0.25f), 0.25f);

    static_cast<void>(reference.next_u64());
    EXPECT_EQ(random, reference);
}

TEST(Small_ImagingRandom, IndexIsTheHighWordOfTheProduct)
{
    Random random(0);

    // 0xE220A839 * 10 is 0x8_D546923A, whose high word is 8.
    EXPECT_EQ(random.index(10), 8u);

    // 0x6E789E6A * 10 is 0x4_50B63024, whose high word is 4.
    EXPECT_EQ(random.index(10), 4u);

    // 0x06C45D18 * 10 is 0x0_43ABA2F0, whose high word is 0.
    EXPECT_EQ(random.index(10), 0u);

    EXPECT_EQ(random.state(), 0xDAA66D2C7DDF743Full);
}

/** @brief Tests that an index below a count of one is always zero. */
TEST(Small_ImagingRandom, IndexOfOneIsAlwaysZero)
{
    Random random(K_SEED);

    EXPECT_EQ(draw_indices_of_one(random, 64), std::vector<std::uint32_t>(64, 0u));
}

/** @brief Tests that an index below a count of one still takes a draw. */
TEST(Small_ImagingRandom, IndexOfOneStillDraws)
{
    Random random(K_SEED);
    Random reference(K_SEED);

    static_cast<void>(random.index(1));

    static_cast<void>(reference.next_u64());
    EXPECT_EQ(random, reference);
}

/** @brief Tests that every bucket of an index draw is hit about as often as any other. */
TEST(Small_ImagingRandom, IndexStaysBelowTheCountAndFillsEveryBucketEvenly)
{
    constexpr std::int32_t K_DRAWS = 70000;
    Random random(K_SEED);

    const std::array<std::int32_t, 7> hits = count_index_hits(random, K_DRAWS);

    const double expected_hits = K_DRAWS / static_cast<double>(hits.size());
    const double bucket_tolerance = K_DRAWS * mean_tolerance(K_DRAWS);
    for (std::size_t bucket = 0; bucket < hits.size(); ++bucket)
        EXPECT_NEAR(hits[bucket], expected_hits, bucket_tolerance) << "bucket " << bucket;
}

TEST(Small_ImagingRandom, IndexReachesTheLargestCount)
{
    Random random(0);

    // 0xE220A839 * 0xFFFFFFFF is 0xE220A838_1DDF57C7, whose high word is 0xE220A838.
    EXPECT_EQ(random.index(0xFFFFFFFFu), 0xE220A838u);
}

TEST(Small_ImagingRandom, BernoulliComparesOneUniformDrawWithTheProbability)
{
    Random random(0);

    // The first three uniform draws of seed 0 are 0.8833, 0.4315 and 0.0264.
    EXPECT_FALSE(random.bernoulli(0.5f));
    EXPECT_TRUE(random.bernoulli(0.5f));
    EXPECT_TRUE(random.bernoulli(0.5f));

    EXPECT_EQ(random.state(), 0xDAA66D2C7DDF743Full);
}

TEST(Small_ImagingRandom, BernoulliNeverSucceedsAtZeroAndAlwaysSucceedsAtOne)
{
    Random never(K_SEED);
    Random always(K_SEED);
    Random reference(K_SEED);

    EXPECT_EQ(count_successes(never, 0.0f, K_SAMPLES), 0);
    EXPECT_EQ(count_successes(always, 1.0f, K_SAMPLES), K_SAMPLES);

    skip_draws(reference, K_SAMPLES);
    EXPECT_EQ(never, reference);
    EXPECT_EQ(always, reference);
}

TEST(Small_ImagingRandom, AZeroDrawIsTheLowerBoundAndFailsAZeroProbability)
{
    // mix(0) is 0, so a counter that the first increment wraps to 0 draws 64 zero bits.
    const std::uint64_t seed = std::uint64_t{0} - Random::K_INCREMENT;

    Random unit(seed);
    Random gate(seed);
    Random ranged(seed);

    ASSERT_EQ(Random::mix(0), 0u);

    EXPECT_EQ(unit.uniform(), 0.0f);
    EXPECT_EQ(unit.state(), 0u);
    EXPECT_FALSE(gate.bernoulli(0.0f));
    EXPECT_EQ(ranged.uniform(0.4f, 0.6f), 0.4f);
}

/** @brief Tests that the success rate of bernoulli is its probability. */
TEST(Small_ImagingRandom, BernoulliSucceedsAtTheRateOfItsProbability)
{
    Random random(K_SEED);

    const std::int32_t successes = count_successes(random, 0.25f, K_SAMPLES);

    EXPECT_NEAR(static_cast<double>(successes) / K_SAMPLES, 0.25, mean_tolerance(K_SAMPLES));
}

TEST(Small_ImagingRandom, DeriveStartsAtTheMixOfTheStateAndTheKeyedMix)
{
    // mix(0xE220A8397B1DCDAF), worked out by hand, is 0x48218226FF3CD4BF.
    EXPECT_EQ(Random(0).derive(0).state(), 0x48218226FF3CD4BFull);
}

TEST(Small_ImagingRandom, DeriveDependsOnTheKeyAndOnTheParentState)
{
    EXPECT_EQ(Random(0).derive(1).state(), 0xDCE423FC82C0D5B8ull);
    EXPECT_EQ(Random(7).derive(5).state(), 0xDBD5DBE5ABB6FDECull);

    EXPECT_NE(Random(0).derive(0), Random(0).derive(1));
    EXPECT_NE(Random(0).derive(5), Random(7).derive(5));
}

TEST(Small_ImagingRandom, DeriveLeavesTheParentUntouched)
{
    const Random parent(K_SEED);
    const Random child = parent.derive(3);

    EXPECT_EQ(parent.state(), K_SEED);
    EXPECT_NE(child, parent);
    EXPECT_EQ(parent.derive(3), child);
}

TEST(Small_ImagingRandom, ACopyReplaysTheSequenceFromThePointOfTheCopy)
{
    Random random(K_SEED);
    static_cast<void>(random.next_u64());

    Random copy = random;
    EXPECT_EQ(copy, random);

    EXPECT_EQ(draw_values(copy, 16), draw_values(random, 16));
}

TEST(Small_ImagingRandom, EqualityComparesTheCounter)
{
    Random left(K_SEED);
    Random right(K_SEED);

    EXPECT_TRUE(left == right);
    EXPECT_FALSE(left != right);

    static_cast<void>(left.next_u64());

    EXPECT_FALSE(left == right);
    EXPECT_TRUE(left != right);
}
