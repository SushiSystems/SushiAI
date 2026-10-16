/**************************************************************************/
/* test_common.cpp                                                        */
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
 * @file test_common.cpp
 * @brief Tests the element-type helpers, the Error type and SA_THROW_IF of common.hpp.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <string>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

using SushiAI::DType;

namespace
{
    /** @brief Throws through SA_THROW_IF when @p fail holds, naming @p extent; counts the calls. */
    void throw_when(bool fail, int extent, int& reached)
    {
        SA_THROW_IF(fail, "extent {} is not positive", extent);
        ++reached;
    }
} // namespace

/** @brief Tests that every element type has its width in bytes and its diagnostic name. */
TEST(Small_Common, EveryElementTypeHasItsWidthAndItsName)
{
    EXPECT_EQ(SushiAI::dtype_size(DType::Half), 2u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Float32), 4u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Float64), 8u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Complex32), 8u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Complex64), 16u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Int32), 4u);
    EXPECT_EQ(SushiAI::dtype_size(DType::Int64), 8u);

    EXPECT_STREQ(SushiAI::dtype_name(DType::Half), "half");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Float32), "float32");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Float64), "float64");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Complex32), "complex32");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Complex64), "complex64");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Int32), "int32");
    EXPECT_STREQ(SushiAI::dtype_name(DType::Int64), "int64");
}

/** @brief Tests that only the three real floating-point types count as real floats. */
TEST(Small_Common, OnlyTheRealFloatingTypesAreRealFloats)
{
    static_assert(SushiAI::is_real_float(DType::Float32), "is_real_float is constexpr");

    EXPECT_TRUE(SushiAI::is_real_float(DType::Half));
    EXPECT_TRUE(SushiAI::is_real_float(DType::Float32));
    EXPECT_TRUE(SushiAI::is_real_float(DType::Float64));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Complex32));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Complex64));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Int32));
    EXPECT_FALSE(SushiAI::is_real_float(DType::Int64));
}

/** @brief Tests that the maximum rank is the one SushiBLAS stores. */
TEST(Small_Common, TheMaximumRankIsTheRankSushiBLASStores)
{
    EXPECT_EQ(SushiAI::K_MAX_RANK,
              static_cast<int>(SushiBLAS::Core::K_MAX_TENSOR_RANK));
}

/** @brief Tests that Error is caught as a std::runtime_error and keeps its message. */
TEST(Small_Common, AnErrorIsARuntimeErrorWithItsMessage)
{
    static_assert(std::is_base_of<std::runtime_error, SushiAI::Error>::value,
                  "Error derives from std::runtime_error");

    try
    {
        throw SushiAI::Error("a shape mismatch");
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_EQ(std::string(error.what()), "a shape mismatch");
        return;
    }
    ADD_FAILURE() << "SushiAI::Error was not caught as std::runtime_error";
}

/** @brief Tests that SA_THROW_IF passes on a false condition and throws a formatted Error. */
TEST(Small_Common, ThrowIfThrowsAFormattedErrorOnlyWhenItsConditionHolds)
{
    int reached = 0;
    throw_when(false, 3, reached);
    EXPECT_EQ(reached, 1);

    SA_EXPECT_ERROR(throw_when(true, -2, reached), "extent -2 is not positive");
    EXPECT_EQ(reached, 1);
}
