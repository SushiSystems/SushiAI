/**************************************************************************/
/* test_dtype.cpp                                                         */
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
 * @file test_dtype.cpp
 * @brief Tests the element-type helpers and K_MAX_RANK of dtype.hpp.
 * @author Mustafa Garip
 */

#include <SushiAI/dtype/dtype.hpp>

#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>

using SushiAI::DType;

/** @brief Tests that every element type has its width in bytes and its diagnostic name. */
TEST(Small_DType, EveryElementTypeHasItsWidthAndItsName)
{
    EXPECT_EQ(SushiAI::dtype_size(DType::Half), 2u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Float32), 4u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Float64), 8u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Complex32), 8u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Complex64), 16u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Int32), 4u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Int64), 8u);
    EXPECT_EQ(SushiAI::dtype_size(DType::UInt32), 4u);

    EXPECT_STREQ(SushiAI::dtype_name(DType::Half), "half");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Float32), "float32");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Float64), "float64");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Complex32), "complex32");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Complex64), "complex64");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Int32), "int32");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Int64), "int64");
    EXPECT_STREQ(SushiAI::dtype_name(DType::UInt32), "uint32");
}

/** @brief Tests that only the three real floating-point types count as real floats. */
TEST(Small_DType, OnlyTheRealFloatingTypesAreRealFloats)
{
    static_assert(SushiAI::is_real_float(DType::Float32), "is_real_float is constexpr");

    EXPECT_TRUE(SushiAI::is_real_float(DType::Half));
    EXPECT_TRUE(SushiAI::is_real_float(DType::Float32));
    EXPECT_TRUE(SushiAI::is_real_float(DType::Float64));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Complex32));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Complex64));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Int32));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Int64));
    EXPECT_FALSE(SushiAI::is_real_float(DType::UInt32));
}

/** @brief Tests that the maximum rank is the one SushiBLAS stores. */
TEST(Small_DType, TheMaximumRankIsTheRankSushiBLASStores)
{
    EXPECT_EQ(SushiAI::K_MAX_RANK,
              static_cast<int>(SushiBLAS::Core::K_MAX_TENSOR_RANK));
}
