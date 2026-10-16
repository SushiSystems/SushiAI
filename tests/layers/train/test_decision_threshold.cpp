/**************************************************************************/
/* test_decision_threshold.cpp                                            */
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
 * @file test_decision_threshold.cpp
 * @brief Tests the threshold constants and the range validate_threshold accepts.
 * @author Mustafa Garip
 */

#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/decision_threshold.hpp>

using namespace SushiAI;

/** @brief Tests that the key is "threshold" and the default is one half, which is valid. */
TEST(Small_DecisionThreshold, TheKeyIsThresholdAndTheDefaultIsOneHalf)
{
    EXPECT_EQ(Train::K_THRESHOLD_KEY, "threshold");
    EXPECT_EQ(Train::K_DEFAULT_THRESHOLD, 0.5);
    SA_EXPECT_OK(Train::validate_threshold(Train::K_DEFAULT_THRESHOLD));
}

/** @brief Tests that the open interval (0, 1) is accepted right up to its bounds. */
TEST(Small_DecisionThreshold, ProbabilitiesInsideTheOpenIntervalAreAccepted)
{
    for (const double threshold : {1e-9, 0.25, 1.0 - 1e-9})
        SA_EXPECT_OK(Train::validate_threshold(threshold));
}

/** @brief Tests that both bounds, values outside them and a NaN are refused. */
TEST(Small_DecisionThreshold, TheBoundsAndANaNAreRefused)
{
    for (const double threshold : {0.0, 1.0, -0.1, 2.0, std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()})
    {
        SA_EXPECT_FAILURE(Train::validate_threshold(threshold), StatusCode::InvalidDescription,
                          "above 0 and below 1");
    }
}
