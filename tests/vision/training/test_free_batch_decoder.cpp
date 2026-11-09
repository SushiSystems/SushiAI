/**************************************************************************/
/* test_free_batch_decoder.cpp                                            */
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
 * @file test_free_batch_decoder.cpp
 * @brief Tests the decode of one image over two levels against boxes worked by hand.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <string>
#include <gtest/gtest.h>

#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/training/free_batch_decoder.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Vision::Training::FreeBatchDecoder;
using Vision::Training::FreeLevelLogits;
using Vision::Training::FreeModelFacts;

namespace
{
    /**
     * @brief Holds the absolute tolerance of a decoded value: 2048 times the float unit roundoff,
     *        the smallest power of two not tighter than the earlier 1e-4f.
     */
    constexpr float K_TOLERANCE = static_cast<float>(2048.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds the class count of the fixture. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the anchors of the fixture: two on level 0, one on level 1. */
    constexpr std::size_t K_ANCHORS = 3;

    /** @brief Returns facts of a 1 by 2 level at stride 8 and a 1 by 1 level at stride 16. */
    FreeModelFacts make_facts(std::int64_t reg_max)
    {
        FreeModelFacts facts;
        facts.reg_max = reg_max;
        facts.classes = K_CLASSES;
        facts.levels.resize(2);
        facts.levels[0].height = 1;
        facts.levels[0].width = 2;
        facts.levels[0].stride = 8;
        facts.levels[1].height = 1;
        facts.levels[1].width = 1;
        facts.levels[1].stride = 16;
        return facts;
    }

    /** @brief Holds class logits whose sigmoids are 0.5, 0.75, 0.75, 0.5, 0.25, 0.5. */
    const std::vector<float> K_LEVEL0_SCORES = {0.0f, std::log(3.0f), std::log(3.0f), 0.0f};

    /** @brief Holds the class logits of the single cell of level 1. */
    const std::vector<float> K_LEVEL1_SCORES = {-std::log(3.0f), 0.0f};

    /** @brief Holds the probabilities the class logits above decode to. */
    constexpr std::array<float, K_ANCHORS * 2> K_EXPECTED_SCORES = {0.5f,  0.75f, 0.75f,
                                                                    0.5f,  0.25f, 0.5f};

    /** @brief Decodes the fixture and checks every box and score against @p expected. */
    void expect_decode(std::int64_t reg_max, const std::vector<float>& level0_boxes,
                       const std::vector<float>& level1_boxes,
                       const std::array<float, K_ANCHORS * 4>& expected)
    {
        SA_ASSERT_OK_AND_ASSIGN(FreeBatchDecoder decoder,
                                FreeBatchDecoder::make(make_facts(reg_max), 1));
        ASSERT_EQ(decoder.anchors(), static_cast<std::int64_t>(K_ANCHORS));

        const std::array<FreeLevelLogits, 2> levels = {
            FreeLevelLogits{level0_boxes, K_LEVEL0_SCORES},
            FreeLevelLogits{level1_boxes, K_LEVEL1_SCORES}};
        std::vector<float> boxes(K_ANCHORS * 4);
        std::vector<float> scores(K_ANCHORS * 2);
        SA_EXPECT_OK(decoder.decode(levels, boxes, scores));

        for (std::size_t at = 0; at < expected.size(); ++at)
            EXPECT_NEAR(boxes[at], expected[at], K_TOLERANCE) << "box value " << at;
        for (std::size_t at = 0; at < K_EXPECTED_SCORES.size(); ++at)
            EXPECT_NEAR(scores[at], K_EXPECTED_SCORES[at], K_TOLERANCE) << "score " << at;
    }

    /** @brief Returns the box logits of the single cell of level 1 with a dominant bin 2. */
    std::vector<float> dominant_bin_two_logits()
    {
        std::vector<float> logits(4 * 4, 0.0f);
        for (std::size_t side = 0; side < 4; ++side)
            logits[side * 4 + 2] = 50.0f;

        return logits;
    }

    /** @brief Names a break of the fixture's inputs and what the refusal names. */
    struct NonFiniteInput
    {
        const char* name;                                          /**< Names the row. */
        void (*apply)(std::vector<float>&, std::vector<float>&);  /**< Breaks one input. */
        const char* needle;                                        /**< What the message names. */
    };

    /** @brief Provides the non-finite inputs a decode refuses. */
    class Small_FreeBatchDecoderNonFinite : public ::testing::TestWithParam<NonFiniteInput>
    {

    };

    /** @brief Names facts the decoder cannot be made for and what the refusal names. */
    struct UnmakeableFacts
    {
        const char* name;                          /**< Names the row. */
        void (*apply)(FreeModelFacts&);            /**< Breaks the facts. */
        const char* needle;                        /**< What the message names. */
    };

    /** @brief Provides the facts make refuses. */
    class Small_FreeBatchDecoderFacts : public ::testing::TestWithParam<UnmakeableFacts>
    {

    };
} // namespace

/** @brief Tests that reg_max 1 reads the box channels as distances in grid units. */
TEST(Small_FreeBatchDecoder, ReadsSingleBinLogitsAsDistances)
{
    // Distances are left, top, right, bottom per cell, in grid units.
    const std::vector<float> level0 = {0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f};
    const std::vector<float> level1 = {0.5f, 0.5f, 1.5f, 1.5f};
    const std::array<float, K_ANCHORS * 4> expected = {4.0f,  4.0f,  8.0f,  8.0f,
                                                       12.0f, 4.0f,  16.0f, 0.0f,
                                                       16.0f, 16.0f, 32.0f, 32.0f};
    expect_decode(1, level0, level1, expected);
}

/** @brief Tests that reg_max 4 takes the expectation of each side's bins. */
TEST(Small_FreeBatchDecoder, TakesTheExpectationOfFourBins)
{
    // Uniform bins expect 1.5 on level 0; a dominant bin 2 expects 2 on level 1.
    const std::vector<float> level0(2 * 4 * 4, 0.0f);
    const std::vector<float> level1 = dominant_bin_two_logits();

    const std::array<float, K_ANCHORS * 4> expected = {4.0f,  4.0f, 24.0f, 24.0f,
                                                       12.0f, 4.0f, 24.0f, 24.0f,
                                                       8.0f,  8.0f, 64.0f, 64.0f};
    expect_decode(4, level0, level1, expected);
}

/** @brief Tests that make refuses a batch below 1, naming it. */
TEST(Small_FreeBatchDecoder, MakeRefusesABatchBelowOne)
{
    SA_EXPECT_FAILURE(FreeBatchDecoder::make(make_facts(1), 0), StatusCode::InvalidDescription,
                      "batch 0");
}

/** @brief Tests that a level count other than the facts' is refused. */
TEST(Small_FreeBatchDecoder, RejectsAMissingLevel)
{
    SA_ASSERT_OK_AND_ASSIGN(FreeBatchDecoder decoder, FreeBatchDecoder::make(make_facts(1), 1));
    const std::vector<float> level0(8, 0.0f);
    const std::array<FreeLevelLogits, 1> levels = {FreeLevelLogits{level0, K_LEVEL0_SCORES}};
    std::vector<float> boxes(K_ANCHORS * 4);
    std::vector<float> scores(K_ANCHORS * 2);

    SA_EXPECT_FAILURE(decoder.decode(levels, boxes, scores), StatusCode::InvalidDescription,
                      "got 1 levels");
}

/** @brief Tests that a NaN box output or an infinite score is refused, naming level and index. */
TEST_P(Small_FreeBatchDecoderNonFinite, RefusesNonFiniteBoxOutputsAndScores)
{
    SA_ASSERT_OK_AND_ASSIGN(FreeBatchDecoder decoder, FreeBatchDecoder::make(make_facts(1), 1));
    std::vector<float> level0(8, 0.0f);
    const std::vector<float> level1(4, 0.5f);
    std::vector<float> level1_scores = K_LEVEL1_SCORES;
    GetParam().apply(level0, level1_scores);
    const std::array<FreeLevelLogits, 2> levels = {FreeLevelLogits{level0, K_LEVEL0_SCORES},
                                                   FreeLevelLogits{level1, level1_scores}};
    std::vector<float> boxes(K_ANCHORS * 4);
    std::vector<float> scores(K_ANCHORS * 2);

    SA_EXPECT_FAILURE(decoder.decode(levels, boxes, scores), StatusCode::InvalidDescription,
                      GetParam().needle);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_FreeBatchDecoderNonFinite,
    ::testing::Values(
        NonFiniteInput{"NaNBoxOutput",
                       [](std::vector<float>& level0, std::vector<float>&)
                       { level0[5] = std::numeric_limits<float>::quiet_NaN(); },
                       "level 0 box output 5"},
        NonFiniteInput{"InfiniteScore",
                       [](std::vector<float>&, std::vector<float>& level1_scores)
                       { level1_scores[1] = std::numeric_limits<float>::infinity(); },
                       "level 1 score 1"}),
    [](const ::testing::TestParamInfo<NonFiniteInput>& info) { return std::string(info.param.name); });

/** @brief Tests that facts the decoder cannot be made for are refused, naming the counts. */
TEST_P(Small_FreeBatchDecoderFacts, MakeRefusesFactsItCannotDecode)
{
    FreeModelFacts facts = make_facts(1);
    GetParam().apply(facts);

    SA_EXPECT_FAILURE(FreeBatchDecoder::make(facts, 1), StatusCode::InvalidDescription,
                      GetParam().needle);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_FreeBatchDecoderFacts,
    ::testing::Values(
        UnmakeableFacts{"RegMaxBelowOne", [](FreeModelFacts& facts) { facts.reg_max = 0; },
                        "reg_max 0"},
        UnmakeableFacts{"ClassesBelowOne", [](FreeModelFacts& facts) { facts.classes = 0; },
                        "0 classes"},
        UnmakeableFacts{"NoLevel", [](FreeModelFacts& facts) { facts.levels.clear(); },
                        "0 levels"},
        UnmakeableFacts{"LevelWithoutCells", [](FreeModelFacts& facts)
                        { facts.levels[0].width = 0; },
                        "a feature level needs positive extents"},
        UnmakeableFacts{"LevelWithoutStride", [](FreeModelFacts& facts)
                        { facts.levels[1].stride = 0; },
                        "a positive stride"}),
    [](const ::testing::TestParamInfo<UnmakeableFacts>& info) { return std::string(info.param.name); });
