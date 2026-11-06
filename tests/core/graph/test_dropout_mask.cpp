/**************************************************************************/
/* test_dropout_mask.cpp                                                  */
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
 * @file test_dropout_mask.cpp
 * @brief Tests the draws of the dropout generator against golden values and its own contract.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/dropout_mask.hpp>
#include <support/splitmix64.hpp>

using namespace SushiAI;
using TestSupport::reference_splitmix64;

namespace
{
    /** @brief Holds the site seed every case but the published-sequence one draws from. */
    constexpr std::uint64_t K_SEED = 2026;

    /** @brief Holds the element count of the kept-fraction cases, 2^20. */
    constexpr std::size_t K_DRAWS = std::size_t{1} << 20;

    /** @brief Holds the square root of K_DRAWS, 2^10. */
    constexpr double K_SQRT_DRAWS = 1024.0;

    /**
     * @brief Holds the z-score of the kept-fraction bound: a two-sided false-failure rate of
     * 4.1e-5 for one check of a fair draw.
     */
    constexpr double K_FRACTION_Z = 4.1;

    /**
     * @brief Bounds the absolute difference of the kept fraction from the drop rate. The kept
     * count of K_DRAWS draws has a standard deviation of at most 0.5 / sqrt(K_DRAWS) as a
     * fraction, so the bound is K_FRACTION_Z of them, 0.002002; the seed is fixed, so a run
     * either always passes or always fails.
     */
    constexpr double K_FRACTION_TOLERANCE = K_FRACTION_Z * 0.5 / K_SQRT_DRAWS;

    /** @brief Returns the bits the rule gives element @p element of run @p run at @p seed. */
    std::uint64_t reference_bits(std::uint64_t seed, std::uint64_t run, std::uint64_t element)
    {
        return reference_splitmix64(reference_splitmix64(seed, run), element);
    }

    /**
     * @brief Returns the mask the rule gives: an element is kept when its top 53 bits, as a
     * number in [0, 1), fall below @p keep.
     */
    std::string expected_mask(std::uint64_t seed, std::uint64_t run, std::size_t count,
                              double keep)
    {
        std::string text;
        for (std::size_t element = 0; element < count; ++element)
        {
            const double unit = static_cast<double>(reference_bits(seed, run, element) >> 11)
                              / 9007199254740992.0;
            text.push_back(unit < keep ? '1' : '0');
        }

        return text;
    }

    /** @brief Returns the mask of one run as a string of '1' for kept and '0' for dropped. */
    std::string drawn(std::uint64_t seed, std::uint64_t run, std::size_t count, double keep)
    {
        std::vector<float> mask(count, -1.0f);
        Graph::fill_keep_mask(mask.data(), count, seed, run, keep);

        std::string text;
        for (const float element : mask)
            text.push_back(element == 1.0f ? '1' : '0');

        return text;
    }

    /** @brief Returns the fraction of ones in the first K_DRAWS draws of one run. */
    double kept_fraction(std::uint64_t run, double keep)
    {
        std::vector<float> mask(K_DRAWS);
        Graph::fill_keep_mask(mask.data(), K_DRAWS, K_SEED, run, keep);

        double kept = 0.0;
        for (const float element : mask)
            kept += element;

        return kept / static_cast<double>(K_DRAWS);
    }
}

/** @brief Tests the first four numbers of seed 0 against the published SplitMix64 sequence. */
TEST(Small_DropoutMask, TheSequenceOfSeedZeroIsThePublishedSplitMix64)
{
    EXPECT_EQ(Graph::splitmix64_at(0, 0), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(Graph::splitmix64_at(0, 1), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(Graph::splitmix64_at(0, 2), 0x06C45D188009454Full);
    EXPECT_EQ(Graph::splitmix64_at(0, 3), 0xF88BB8A8724C81ECull);
}

/** @brief Tests that the seed of site s is number s of the trace seed's sequence. */
TEST(Small_DropoutMask, EachSiteOfATraceHasItsOwnSeed)
{
    EXPECT_EQ(Graph::dropout_site_seed(0, 0), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(Graph::dropout_site_seed(0, 1), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(Graph::dropout_site_seed(7, 0), reference_splitmix64(7, 0));
    EXPECT_EQ(Graph::dropout_site_seed(7, 1), reference_splitmix64(7, 1));
}

/** @brief Tests that the bits of an element are the sequence of the run's sequence. */
TEST(Small_DropoutMask, TheBitsOfASeedARunAndAnElementFollowTheRule)
{
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 0, 0), reference_bits(K_SEED, 0, 0));
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 0, 3), reference_bits(K_SEED, 0, 3));
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 1, 0), reference_bits(K_SEED, 1, 0));
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 1, 3), reference_bits(K_SEED, 1, 3));
}

/** @brief Tests that at a keep of one half an element is kept when its bits fall below half. */
TEST(Small_DropoutMask, TheMaskOfARunKeepsTheElementsWhoseBitsFallBelowTheKeep)
{
    EXPECT_EQ(drawn(K_SEED, 0, 32, 0.5), expected_mask(K_SEED, 0, 32, 0.5));
    EXPECT_EQ(drawn(K_SEED, 1, 32, 0.5), expected_mask(K_SEED, 1, 32, 0.5));
}

/** @brief Tests that the fill writes the mask the rule gives for 257 elements at a keep of 0.7. */
TEST(Small_DropoutMask, TheFillWritesTheMaskTheRuleGives)
{
    EXPECT_EQ(drawn(K_SEED, 5, 257, 0.7), expected_mask(K_SEED, 5, 257, 0.7));
}

/** @brief Tests that an element's draw does not depend on how many elements are filled. */
TEST(Small_DropoutMask, AShorterFillIsAPrefixOfALongerOne)
{
    const std::string longer = drawn(K_SEED, 2, 96, 0.9);

    EXPECT_EQ(drawn(K_SEED, 2, 40, 0.9), longer.substr(0, 40));
}

/** @brief Tests that the same seed, run and keep give the same mask on a second call. */
TEST(Small_DropoutMask, TheSameInputsGiveTheSameMask)
{
    EXPECT_EQ(drawn(K_SEED, 9, 4096, 0.8), drawn(K_SEED, 9, 4096, 0.8));
}

/** @brief Tests that two runs of one site draw different masks. */
TEST(Small_DropoutMask, TwoRunsDrawDifferentMasks)
{
    EXPECT_NE(drawn(K_SEED, 0, 4096, 0.5), drawn(K_SEED, 1, 4096, 0.5));
}

/** @brief Tests that two sites of one trace draw different masks in the same run. */
TEST(Small_DropoutMask, TwoSitesDrawDifferentMasks)
{
    EXPECT_NE(drawn(Graph::dropout_site_seed(0, 0), 0, 4096, 0.5),
              drawn(Graph::dropout_site_seed(0, 1), 0, 4096, 0.5));
}

/** @brief Tests the kept fraction of 2^20 draws at three keeps, within K_FRACTION_TOLERANCE. */
TEST(Small_DropoutMask, TheKeptFractionIsTheKeepProbability)
{
    EXPECT_NEAR(kept_fraction(0, 0.9), 0.9, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(kept_fraction(0, 0.5), 0.5, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(kept_fraction(3, 0.9), 0.9, K_FRACTION_TOLERANCE);
}

/** @brief Tests that a keep of 1 keeps every element and a keep of 0 keeps none. */
TEST(Small_DropoutMask, TheTwoEndsOfTheKeepRangeAreExact)
{
    EXPECT_EQ(drawn(K_SEED, 0, 64, 1.0), std::string(64, '1'));
    EXPECT_EQ(drawn(K_SEED, 0, 64, 0.0), std::string(64, '0'));
}
