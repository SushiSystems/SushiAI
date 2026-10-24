/**************************************************************************/
/* test_exact_sum.cpp                                                     */
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
 * @file test_exact_sum.cpp
 * @brief Tests which types the replicas' results add as exact counts.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiAI/distribution/exact_sum.hpp>

namespace
{
    using SushiBLAS::Core::DataType;
    using SushiAI::Distribution::sums_exactly;
} // namespace

TEST(Small_ExactSum, CountsAddWithWeightOne)
{
    EXPECT_TRUE(sums_exactly(DataType::Int32));
    EXPECT_TRUE(sums_exactly(DataType::Int64));
}

TEST(Small_ExactSum, RegressionUInt32IsACount)
{
    EXPECT_TRUE(sums_exactly(DataType::UInt32));
}

TEST(Small_ExactSum, FloatingAndComplexTypesAreWeighed)
{
    EXPECT_FALSE(sums_exactly(DataType::Half));
    EXPECT_FALSE(sums_exactly(DataType::Float32));
    EXPECT_FALSE(sums_exactly(DataType::Float64));
    EXPECT_FALSE(sums_exactly(DataType::Complex32));
    EXPECT_FALSE(sums_exactly(DataType::Complex64));
}
