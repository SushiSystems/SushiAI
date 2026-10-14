/**************************************************************************/
/* test_generation_table_position_feed.cpp                                */
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
 * @file test_generation_table_position_feed.cpp
 * @brief Tests TablePositionFeed, which feeds a token's row of a position table.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include "../source/position_feed.hpp"
#include "../source/table_position_feed.hpp"
#include "position_feed_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::Generation;
using Language::Generation::Detail::PositionFeed;
using Language::Generation::Detail::TablePositionFeed;

namespace
{
    /** @brief Builds the steps a table feed is bound to. */
    class Unit_GenerationTablePositionFeed : public Support::FedStepTest
    {

    };
} // namespace

/** @brief Tests that the feed lists one [1, 1] int32 leaf and writes the position into it. */
TEST_F(Unit_GenerationTablePositionFeed, PlaceWritesThePositionIntoItsOneLeaf)
{
    build_step(true, false);
    ASSERT_NE(step, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PositionFeed> feed,
                            TablePositionFeed::make(*engine, *step, Support::K_FEED_POSITION));

    const SushiRuntime::span<const Train::NamedFeed> fed = feed->feeds();
    ASSERT_EQ(fed.size(), 1u);
    EXPECT_EQ(fed[0].name, Support::K_FEED_POSITION);
    ASSERT_NE(fed[0].tensor, nullptr);
    EXPECT_EQ(fed[0].tensor->dtype, DType::Int32);
    EXPECT_EQ(fed[0].tensor->num_elements, 1);

    feed->place(0);
    EXPECT_EQ(fed[0].tensor->data_as<std::int32_t>()[0], 0);
    feed->place(5);
    EXPECT_EQ(fed[0].tensor->data_as<std::int32_t>()[0], 5);
}

/** @brief Tests that a slide leaves no cached row right, so the generator feeds them again. */
TEST_F(Unit_GenerationTablePositionFeed, NoRowStaysRightAfterASlide)
{
    build_step(true, false);
    ASSERT_NE(step, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PositionFeed> feed,
                            TablePositionFeed::make(*engine, *step, Support::K_FEED_POSITION));

    EXPECT_FALSE(feed->keeps_rows_on_slide());
}

/** @brief Tests that a leaf the step is not fed and a leaf of another type are refused. */
TEST_F(Unit_GenerationTablePositionFeed, ALeafThatDoesNotFitIsRefused)
{
    build_step(false, true);
    ASSERT_NE(step, nullptr);

    SA_EXPECT_FAILURE(TablePositionFeed::make(*engine, *step, Support::K_FEED_POSITION),
                      StatusCode::InvalidDescription, "'feed_position'");
    SA_EXPECT_FAILURE(TablePositionFeed::make(*engine, *step, Support::K_FEED_COSINE),
                      StatusCode::InvalidDescription, "CachedGenerator", "'feed_cosine'",
                      "int32 leaf of [1, 1]");
}
