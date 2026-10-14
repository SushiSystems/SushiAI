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

using namespace SushiAI;

namespace
{
    /** @brief Holds the site seed every case but the published-sequence one draws from. */
    constexpr std::uint64_t K_SEED = 2026;

    /** @brief Holds the element count of the kept-fraction cases, 2^20. */
    constexpr std::size_t K_DRAWS = std::size_t{1} << 20;

    /** @brief Bounds the kept fraction of K_DRAWS draws; a deviation of one part is 0.0003. */
    constexpr double K_FRACTION_TOLERANCE = 0.002;

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
TEST(Unit_DropoutMask, TheSequenceOfSeedZeroIsThePublishedSplitMix64)
{
    EXPECT_EQ(Graph::splitmix64_at(0, 0), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(Graph::splitmix64_at(0, 1), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(Graph::splitmix64_at(0, 2), 0x06C45D188009454Full);
    EXPECT_EQ(Graph::splitmix64_at(0, 3), 0xF88BB8A8724C81ECull);
}

/** @brief Tests that the seed of site s is number s of the trace seed's sequence. */
TEST(Unit_DropoutMask, EachSiteOfATraceHasItsOwnSeed)
{
    EXPECT_EQ(Graph::dropout_site_seed(0, 0), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(Graph::dropout_site_seed(0, 1), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(Graph::dropout_site_seed(7, 0), 0x63CBE1E459320DD7ull);
    EXPECT_EQ(Graph::dropout_site_seed(7, 1), 0x044C3CD7F43C661Cull);
}

/** @brief Tests the bits of the first four elements of runs 0 and 1 against golden values. */
TEST(Unit_DropoutMask, TheBitsOfASeedARunAndAnElementAreGolden)
{
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 0, 0), 0x6E75725323D4929Eull);
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 0, 1), 0x07B389BACFD8F970ull);
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 0, 2), 0x29267FA040AE73FFull);
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 0, 3), 0x43B5CB642EB7CF71ull);
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 1, 0), 0xDFEF6464718A9796ull);
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 1, 1), 0x7F73EFFCD6482B61ull);
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 1, 2), 0x7DCC57420595AF2Full);
    EXPECT_EQ(Graph::dropout_bits(K_SEED, 1, 3), 0x062E26736BA469C5ull);
}

/** @brief Tests the first 32 elements of two runs at a keep of one half against golden masks. */
TEST(Unit_DropoutMask, TheMaskOfARunIsGolden)
{
    EXPECT_EQ(drawn(K_SEED, 0, 32, 0.5), "11110000011001111110011111110000");
    EXPECT_EQ(drawn(K_SEED, 1, 32, 0.5), "01110100100110000111111111000011");
}

/** @brief Tests that the fill writes what dropout_keeps answers, element by element. */
TEST(Unit_DropoutMask, TheFillAgreesWithTheElementwiseQuestion)
{
    constexpr std::size_t K_COUNT = 257;
    constexpr double K_KEEP = 0.7;
    const std::string mask = drawn(K_SEED, 5, K_COUNT, K_KEEP);

    for (std::size_t element = 0; element < K_COUNT; ++element)
    {
        EXPECT_EQ(mask[element] == '1', Graph::dropout_keeps(K_SEED, 5, element, K_KEEP))
            << "at element " << element;
    }
}

/** @brief Tests that an element's draw does not depend on how many elements are filled. */
TEST(Unit_DropoutMask, AShorterFillIsAPrefixOfALongerOne)
{
    const std::string longer = drawn(K_SEED, 2, 96, 0.9);

    EXPECT_EQ(drawn(K_SEED, 2, 40, 0.9), longer.substr(0, 40));
}

/** @brief Tests that the same seed, run and keep give the same mask on a second call. */
TEST(Unit_DropoutMask, TheSameInputsGiveTheSameMask)
{
    EXPECT_EQ(drawn(K_SEED, 9, 4096, 0.8), drawn(K_SEED, 9, 4096, 0.8));
}

/** @brief Tests that two runs differ in 2033 of 4096 elements, and two sites in some. */
TEST(Unit_DropoutMask, TwoRunsAndTwoSitesDrawDifferentMasks)
{
    const std::string first = drawn(K_SEED, 0, 4096, 0.5);
    const std::string second = drawn(K_SEED, 1, 4096, 0.5);

    std::size_t differing = 0;
    for (std::size_t element = 0; element < first.size(); ++element)
        differing += first[element] != second[element] ? 1u : 0u;

    EXPECT_EQ(differing, 2033u);
    EXPECT_NE(drawn(Graph::dropout_site_seed(0, 0), 0, 4096, 0.5),
              drawn(Graph::dropout_site_seed(0, 1), 0, 4096, 0.5));
}

/** @brief Tests the kept fraction of 2^20 draws at three keeps, within K_FRACTION_TOLERANCE. */
TEST(Unit_DropoutMask, TheKeptFractionIsTheKeepProbability)
{
    EXPECT_NEAR(kept_fraction(0, 0.9), 0.9, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(kept_fraction(0, 0.5), 0.5, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(kept_fraction(3, 0.9), 0.9, K_FRACTION_TOLERANCE);
}

/** @brief Tests that a keep of 1 keeps every element and a keep of 0 keeps none. */
TEST(Unit_DropoutMask, TheTwoEndsOfTheKeepRangeAreExact)
{
    EXPECT_EQ(drawn(K_SEED, 0, 64, 1.0), std::string(64, '1'));
    EXPECT_EQ(drawn(K_SEED, 0, 64, 0.0), std::string(64, '0'));
}
