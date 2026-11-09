/**************************************************************************/
/* test_table_position_feed.cpp                                          */
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
 * @file test_table_position_feed.cpp
 * @brief Tests TablePositionFeed, which feeds a token's row of a position table.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include "position_feed_test_support.hpp"
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/position_feed.hpp>
#include <SushiAI/language/generation/table_position_feed.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;
using Language::Generation::Detail::PositionFeed;
using Language::Generation::Detail::TablePositionFeed;

namespace
{
    /** @brief Names a leaf given to make and what the refusal of it names. */
    struct RefusalCase
    {
        std::string_view leaf;   /**< The leaf given to make. */
        std::string_view first;  /**< A substring the message holds. */
        std::string_view second; /**< Another substring the message holds. */
    };

    /** @brief Returns the positions the place cases write. */
    std::vector<std::int64_t> position_cases()
    {
        return {0, 5};
    }

    /** @brief Returns a leaf the step is not fed and a leaf of another type. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {{Support::K_FEED_POSITION, "'feed_position'", ""},
                {Support::K_FEED_COSINE, "'feed_cosine'", "int32 leaf of [1, 1]"}};
    }

    /** @brief Builds the table feed of a host engine. */
    class Small_GenerationTablePositionFeed : public Support::HostFeedTest
    {
        protected:
            /** @brief Returns the table feed of the position leaf, or null. */
            std::unique_ptr<PositionFeed> make_feed()
            {
                const InMemoryFedLeaves fed = Support::fed_leaves_of(true, false);
                Result<std::unique_ptr<PositionFeed>> made =
                    TablePositionFeed::make(engine(), fed, Support::K_FEED_POSITION);
                EXPECT_TRUE(made.has_value());
                return made.has_value() ? std::move(made).value() : nullptr;
            }
    };

    /** @brief Provides the positions written into the leaf. */
    class Small_GenerationTablePlace : public Small_GenerationTablePositionFeed,
                                       public ::testing::WithParamInterface<std::int64_t>
    {

    };

    /** @brief Provides the leaves that are refused. */
    class Small_GenerationTableRefusal : public Support::HostFeedTest,
                                         public ::testing::WithParamInterface<RefusalCase>
    {

    };
} // namespace

/** @brief Tests that the feed lists one [1, 1] int32 leaf, named by the step. */
TEST_F(Small_GenerationTablePositionFeed, TheFeedListsOneInt32LeafNamedByTheStep)
{
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);

    const SushiRuntime::span<const Train::NamedFeed> fed = feed->feeds();
    ASSERT_EQ(fed.size(), 1u);
    EXPECT_EQ(fed[0].name, Support::K_FEED_POSITION);
    ASSERT_NE(fed[0].tensor, nullptr);
    EXPECT_EQ(fed[0].tensor->dtype, DType::Int32);
    EXPECT_EQ(fed[0].tensor->num_elements, 1);
}

/** @brief Tests that place writes the position into the leaf. */
TEST_P(Small_GenerationTablePlace, PlaceWritesThePositionIntoItsOneLeaf)
{
    const std::int64_t position = GetParam();
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);

    feed->place(position);

    EXPECT_EQ(feed->feeds()[0].tensor->data_as<std::int32_t>()[0], position);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationTablePlace, ::testing::ValuesIn(position_cases()));

/** @brief Tests that a leaf the step is not fed and a leaf of another type are refused. */
TEST_P(Small_GenerationTableRefusal, ALeafThatDoesNotFitIsRefused)
{
    const RefusalCase& row = GetParam();
    const InMemoryFedLeaves fed = Support::fed_leaves_of(false, true);

    SA_EXPECT_FAILURE(TablePositionFeed::make(engine(), fed, row.leaf),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationTableRefusal,
                         ::testing::ValuesIn(refusal_cases()));
