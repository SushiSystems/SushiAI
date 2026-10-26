/**************************************************************************/
/* test_free_branch_weights.cpp                                           */
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
 * @file test_free_branch_weights.cpp
 * @brief Tests the weights that mix the one-to-many and one-to-one detection losses.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/training/free_branch_weights.hpp>

using namespace SushiAI;
using Config::Value;
using Vision::Training::FreeBranchWeights;
using Vision::Training::one_to_many_weight;
using Vision::Training::read_free_branch_weights;

/** @brief Tests that a section without a weight key keeps 0.8 and 0.1. */
TEST(Small_FreeBranchWeights, AnAbsentKeyKeepsItsDefault)
{
    const Value section = Value::object({"type"}, {Value::string("anchor_free")});

    SA_ASSERT_OK_AND_ASSIGN(const FreeBranchWeights weights, read_free_branch_weights(section));

    EXPECT_EQ(weights.start, 0.8f);
    EXPECT_EQ(weights.end, 0.1f);
}

/** @brief Tests that both stated weights are read. */
TEST(Small_FreeBranchWeights, BothStatedWeightsAreRead)
{
    const Value section = Value::object({"one_to_many_weight_start", "one_to_many_weight_end"},
                                        {Value::real(0.5), Value::integer(0)});

    SA_ASSERT_OK_AND_ASSIGN(const FreeBranchWeights weights, read_free_branch_weights(section));

    EXPECT_EQ(weights.start, 0.5f);
    EXPECT_EQ(weights.end, 0.0f);
}

/** @brief Tests that a weight above 1 is refused, naming its key. */
TEST(Small_FreeBranchWeights, AWeightAboveOneIsRefusedNamingItsKey)
{
    const Value section = Value::object({"one_to_many_weight_end"}, {Value::real(1.5)});

    SA_EXPECT_FAILURE(read_free_branch_weights(section), StatusCode::InvalidDescription,
                      "one_to_many_weight_end", "0..1");
}

/** @brief Tests that the weight falls in a straight line to the end and stays there. */
TEST(Small_FreeBranchWeights, TheWeightFallsInAStraightLineAndStaysAtTheEnd)
{
    const FreeBranchWeights weights;

    EXPECT_FLOAT_EQ(one_to_many_weight(weights, 0, 8), 0.8f);
    EXPECT_FLOAT_EQ(one_to_many_weight(weights, 1, 8), 0.7f);
    EXPECT_FLOAT_EQ(one_to_many_weight(weights, 7, 8), 0.1f);
    EXPECT_FLOAT_EQ(one_to_many_weight(weights, 12, 8), 0.1f);
}

/** @brief Tests that a run of one epoch, which has no last epoch to reach, uses the start. */
TEST(Small_FreeBranchWeights, ARunOfOneEpochUsesTheStart)
{
    const FreeBranchWeights weights;

    EXPECT_FLOAT_EQ(one_to_many_weight(weights, 0, 1), 0.8f);
}
