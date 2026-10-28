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
 * @file test_random.cpp
 * @brief Tests the imaging generator's SplitMix64 sequence and its float and index mappings.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/random.hpp>

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
} // namespace

TEST(Small_ImagingRandom, MixFollowsTheFinaliserStepByStep)
{
    // Each literal is the previous one after the next step of mix(), worked out by hand.
    const std::uint64_t value = 0x9E3779B97F4A7C15ull;
    const std::uint64_t first_xor = 0x9E3779BB07979AF0ull;
    const std::uint64_t first_product = 0x6F68261B57E7A770ull;
    const std::uint64_t second_xor = 0x6F682616BAE3641Aull;
    const std::uint64_t second_product = 0xE220A838BF5C9DDEull;
    const std::uint64_t result = 0xE220A8397B1DCDAFull;

    ASSERT_EQ(value ^ (value >> 30), first_xor);
    ASSERT_EQ(first_xor * Random::K_FIRST_MULTIPLIER, first_product);
    ASSERT_EQ(first_product ^ (first_product >> 27), second_xor);
    ASSERT_EQ(second_xor * Random::K_SECOND_MULTIPLIER, second_product);
    ASSERT_EQ(second_product ^ (second_product >> 31), result);

    EXPECT_EQ(Random::mix(value), result);
}

TEST(Small_ImagingRandom, MixOfTheSecondCounterFollowsTheFinaliserStepByStep)
{
    const std::uint64_t value = 0x3C6EF372FE94F82Aull;
    const std::uint64_t first_xor = 0x3C6EF3720F2F35E1ull;
    const std::uint64_t first_product = 0x2A94FCBFCCB43499ull;
    const std::uint64_t second_xor = 0x2A94FCBA9E2BA360ull;
    const std::uint64_t second_product = 0x6E789E6A7D485920ull;
    const std::uint64_t result = 0x6E789E6AA1B965F4ull;

    ASSERT_EQ(value ^ (value >> 30), first_xor);
    ASSERT_EQ(first_xor * Random::K_FIRST_MULTIPLIER, first_product);
    ASSERT_EQ(first_product ^ (first_product >> 27), second_xor);
    ASSERT_EQ(second_xor * Random::K_SECOND_MULTIPLIER, second_product);
    ASSERT_EQ(second_product ^ (second_product >> 31), result);

    EXPECT_EQ(Random::mix(value), result);
}

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

TEST(Small_ImagingRandom, UniformStaysInTheHalfOpenUnitInterval)
{
    Random random(K_SEED);

    for (std::int32_t i = 0; i < K_SAMPLES; ++i)
    {
        const float value = random.uniform();

        ASSERT_GE(value, 0.0f) << "draw " << i;
        ASSERT_LT(value, 1.0f) << "draw " << i;
    }
}

TEST(Small_ImagingRandom, UniformHasTheMeanAndVarianceOfAUniformVariable)
{
    Random random(K_SEED);

    double sum = 0.0;
    double sum_of_squares = 0.0;

    for (std::int32_t i = 0; i < K_SAMPLES; ++i)
    {
        const double value = static_cast<double>(random.uniform());
        sum += value;
        sum_of_squares += value * value;
    }

    const double mean = sum / K_SAMPLES;
    const double variance = sum_of_squares / K_SAMPLES - mean * mean;

    // The standard error of the mean over 65536 draws is 0.0011; 0.006 is five of them.
    EXPECT_NEAR(mean, 0.5, 0.006);

    // The standard error of the variance over 65536 draws is 0.00029; 0.0015 is five of them.
    EXPECT_NEAR(variance, 1.0 / 12.0, 0.0015);
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

TEST(Small_ImagingRandom, RangedUniformStaysInItsHalfOpenInterval)
{
    Random random(K_SEED);

    for (std::int32_t i = 0; i < K_SAMPLES; ++i)
    {
        const float value = random.uniform(0.4f, 0.6f);

        ASSERT_GE(value, 0.4f) << "draw " << i;
        ASSERT_LT(value, 0.6f) << "draw " << i;
    }
}

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

TEST(Small_ImagingRandom, IndexOfOneIsAlwaysZeroAndStillDraws)
{
    Random random(K_SEED);
    Random reference(K_SEED);

    for (std::int32_t i = 0; i < 64; ++i)
    {
        ASSERT_EQ(random.index(1), 0u);
        static_cast<void>(reference.next_u64());
    }

    EXPECT_EQ(random, reference);
}

TEST(Small_ImagingRandom, IndexStaysBelowTheCountAndFillsEveryBucketEvenly)
{
    constexpr std::uint32_t K_BUCKETS = 7;
    constexpr std::int32_t K_DRAWS = 70000;

    Random random(K_SEED);
    std::array<std::int32_t, K_BUCKETS> buckets{};

    for (std::int32_t i = 0; i < K_DRAWS; ++i)
    {
        const std::uint32_t value = random.index(K_BUCKETS);

        ASSERT_LT(value, K_BUCKETS) << "draw " << i;
        ++buckets[value];
    }

    // A bucket expects 10000 hits with a standard deviation of 93; 500 is above five of them.
    for (std::size_t bucket = 0; bucket < buckets.size(); ++bucket)
        EXPECT_NEAR(buckets[bucket], K_DRAWS / static_cast<std::int32_t>(K_BUCKETS), 500)
            << "bucket " << bucket;
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

    for (std::int32_t i = 0; i < K_SAMPLES; ++i)
    {
        ASSERT_FALSE(never.bernoulli(0.0f)) << "draw " << i;
        ASSERT_TRUE(always.bernoulli(1.0f)) << "draw " << i;
        static_cast<void>(reference.next_u64());
    }

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

TEST(Small_ImagingRandom, BernoulliSucceedsAtTheRateOfItsProbability)
{
    Random random(K_SEED);
    std::int32_t successes = 0;

    for (std::int32_t i = 0; i < K_SAMPLES; ++i)
        successes += random.bernoulli(0.25f) ? 1 : 0;

    // The standard error of the rate over 65536 draws is 0.0017; 0.01 is above five of them.
    EXPECT_NEAR(static_cast<double>(successes) / K_SAMPLES, 0.25, 0.01);
}

TEST(Small_ImagingRandom, DeriveStartsAtTheMixOfTheStateAndTheKeyedMix)
{
    // mix(0 + K_INCREMENT) is 0xE220A8397B1DCDAF, and 0 ^ that value is the value itself.
    const std::uint64_t keyed = 0xE220A8397B1DCDAFull;
    const std::uint64_t first_xor = 0xE220A83AF39F6D4Aull;
    const std::uint64_t first_product = 0xA0DDC6FB18E12C7Aull;
    const std::uint64_t second_xor = 0xA0DDC6EF0359F319ull;
    const std::uint64_t second_product = 0x482182266F7FD0F3ull;
    const std::uint64_t result = 0x48218226FF3CD4BFull;

    ASSERT_EQ(keyed ^ (keyed >> 30), first_xor);
    ASSERT_EQ(first_xor * Random::K_FIRST_MULTIPLIER, first_product);
    ASSERT_EQ(first_product ^ (first_product >> 27), second_xor);
    ASSERT_EQ(second_xor * Random::K_SECOND_MULTIPLIER, second_product);
    ASSERT_EQ(second_product ^ (second_product >> 31), result);

    EXPECT_EQ(Random(0).derive(0).state(), result);
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

    for (std::int32_t i = 0; i < 16; ++i)
        ASSERT_EQ(copy.next_u64(), random.next_u64()) << "draw " << i;
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
