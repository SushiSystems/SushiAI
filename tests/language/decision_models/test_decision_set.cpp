/**************************************************************************/
/* test_decision_set.cpp                                                  */
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
 * @file test_decision_set.cpp
 * @brief Tests the synthetic decision set: its seed, its mix of types and the rule of each type.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/decision_models/decision_model.hpp>
#include <SushiAI/language/decision_models/decision_set.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::DecisionModels;
using Language::Datasets::DecisionRecord;
using Language::Datasets::DecisionType;

namespace
{
    /** @brief Holds the seed of the training split. */
    constexpr std::uint64_t K_TRAIN_SEED = 11;

    /** @brief Holds the seed of the held-out split. */
    constexpr std::uint64_t K_TEST_SEED = 13;

    /** @brief Returns a decision set of @p records records drawn from @p seed. */
    std::vector<DecisionRecord> draw_set(std::uint64_t seed, std::int64_t records)
    {
        DecisionSetOptions options;
        options.seed = seed;
        options.records = records;
        options.options = static_cast<std::int32_t>(K_TEST_EXTENTS.options);

        return generate_decision_set(options).value();
    }

    /** @brief Returns how many records of @p records have the type @p type. */
    std::int64_t count_type(const std::vector<DecisionRecord>& records, DecisionType type)
    {
        return std::count_if(records.begin(), records.end(), [type](const DecisionRecord& record)
        {
            return record.type == type;
        });
    }

    /** @brief Returns the ids of @p record's options, one id each, sorted. */
    std::vector<Language::Text::TokenID> sorted_option_ids(const DecisionRecord& record)
    {
        std::vector<Language::Text::TokenID> ids;
        for (const std::vector<Language::Text::TokenID>& option : record.options)
            ids.push_back(option.front());

        std::sort(ids.begin(), ids.end());
        return ids;
    }

    /** @brief Expects @p record to be a choice of the key rule: its right option answers its key. */
    void expect_follows_the_key_rule(const DecisionRecord& record)
    {
        ASSERT_EQ(record.text.size(), 2u);
        ASSERT_GE(record.choice, 0);
        ASSERT_LT(static_cast<std::size_t>(record.choice), record.options.size());

        EXPECT_GE(record.text[0], K_CHOICE_KEY_FIRST);
        EXPECT_LT(record.text[0], K_CHOICE_KEY_FIRST + K_CHOICE_KEYS);
        EXPECT_EQ(record.options[static_cast<std::size_t>(record.choice)].front(),
                  right_option_of(record.text[0]));
        EXPECT_GE(record.options.size(), static_cast<std::size_t>(K_MIN_CHOICE_OPTIONS));
        EXPECT_LE(record.options.size(), static_cast<std::size_t>(K_TEST_EXTENTS.options));

        const std::vector<Language::Text::TokenID> ids = sorted_option_ids(record);
        EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
    }

    /** @brief Expects every record of @p records to hold the answer its type's rule gives. */
    void expect_records_follow_their_rules(const std::vector<DecisionRecord>& records)
    {
        for (const DecisionRecord& record : records)
        {
            const SushiRuntime::span<const Language::Text::TokenID> text(record.text.data(),
                                                                         record.text.size());
            if (record.type == DecisionType::Choice)
            {
                expect_follows_the_key_rule(record);
            }
            else if (record.type == DecisionType::YesNo)
            {
                EXPECT_EQ(record.text.size(), static_cast<std::size_t>(K_SCALAR_TEXT_TOKENS));
                EXPECT_EQ(record.yes, yes_no_answer_of(text));
            }
            else
            {
                EXPECT_EQ(record.text.size(), static_cast<std::size_t>(K_SCALAR_TEXT_TOKENS));
                EXPECT_EQ(record.number, number_answer_of(text));
            }
        }
    }

    /** @brief Holds one refusal of the generator: the options it was asked for. */
    struct RefusedSet
    {
        std::string name;          /**< The row name. */
        std::int64_t records;      /**< The count asked for. */
        std::int32_t options;      /**< The most options asked for. */
    };

    /** @brief Names a row by its name field. */
    std::string row_name(const ::testing::TestParamInfo<RefusedSet>& info)
    {
        return info.param.name;
    }

    /** @brief Runs one set the generator refuses. */
    class Small_DecisionSetRefusals : public ::testing::TestWithParam<RefusedSet>
    {

    };

} // namespace

/** @brief Tests that a seed fixes the records of a set, and another seed changes them. */
TEST(Small_DecisionSet, TheSameSeedDrawsTheSameRecords)
{
    const std::vector<DecisionRecord> first = draw_set(K_TRAIN_SEED, 30);
    const std::vector<DecisionRecord> again = draw_set(K_TRAIN_SEED, 30);
    const std::vector<DecisionRecord> other = draw_set(K_TEST_SEED, 30);

    EXPECT_TRUE(first == again);
    EXPECT_FALSE(first == other);
}

/** @brief Tests that the types take turns, so a set of 3 n records holds n of each. */
TEST(Small_DecisionSet, ASetOfThirtyHoldsTenOfEachType)
{
    const std::vector<DecisionRecord> records = draw_set(K_TRAIN_SEED, 30);

    EXPECT_EQ(records.size(), 30u);
    EXPECT_EQ(count_type(records, DecisionType::Choice), 10);
    EXPECT_EQ(count_type(records, DecisionType::YesNo), 10);
    EXPECT_EQ(count_type(records, DecisionType::Number), 10);
}

/** @brief Tests that every drawn record holds the answer the rule of its type gives. */
TEST(Small_DecisionSet, EveryRecordFollowsTheRuleOfItsType)
{
    expect_records_follow_their_rules(draw_set(K_TRAIN_SEED, 300));
}

/** @brief Tests the yes-or-no rule on hand-counted questions: two positives make a yes. */
TEST(Small_DecisionSet, TwoPositiveTokensMakeAYes)
{
    const std::vector<Language::Text::TokenID> two = {K_YES_NO_FIRST, K_YES_NO_FIRST + 1,
                                                      K_YES_NO_FIRST + 4};
    const std::vector<Language::Text::TokenID> one = {K_YES_NO_FIRST, K_YES_NO_FIRST + 4,
                                                      K_YES_NO_FIRST + 5};

    EXPECT_TRUE(yes_no_answer_of(SushiRuntime::span<const Language::Text::TokenID>(two.data(), 3)));
    EXPECT_FALSE(yes_no_answer_of(SushiRuntime::span<const Language::Text::TokenID>(one.data(), 3)));
}

/** @brief Tests the number rule on questions whose mean is 1, 0 and -1 by hand. */
TEST(Small_DecisionSet, TheNumberIsTheMeanOfTheTokenWeights)
{
    const std::vector<Language::Text::TokenID> top(3, K_NUMBER_FIRST + K_NUMBER_TOKENS - 1);
    const std::vector<Language::Text::TokenID> middle(3, K_NUMBER_FIRST + 3);
    const std::vector<Language::Text::TokenID> bottom(3, K_NUMBER_FIRST);

    EXPECT_EQ(number_answer_of(SushiRuntime::span<const Language::Text::TokenID>(top.data(), 3)),
              1.0f);
    EXPECT_EQ(number_answer_of(SushiRuntime::span<const Language::Text::TokenID>(middle.data(), 3)),
              0.0f);
    EXPECT_EQ(number_answer_of(SushiRuntime::span<const Language::Text::TokenID>(bottom.data(), 3)),
              -1.0f);
}

/** @brief Tests that the generator refuses a negative count and options outside its range. */
TEST_P(Small_DecisionSetRefusals, TheGeneratorRefusesTheSet)
{
    DecisionSetOptions options;
    options.records = GetParam().records;
    options.options = GetParam().options;

    SA_EXPECT_FAILURE(generate_decision_set(options), StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(Small_DecisionSetRefusals, Small_DecisionSetRefusals,
                         ::testing::Values(RefusedSet{"NegativeRecords", -1, 4},
                                           RefusedSet{"OneOption", 3, 1},
                                           RefusedSet{"NineOptions", 3, 9}),
                         row_name);
