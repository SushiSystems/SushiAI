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
 * @file test_dtype.cpp
 * @brief Tests the element-type helpers and K_MAX_RANK of dtype.hpp.
 * @author Mustafa Garip
 */

#include <SushiAI/dtype/dtype.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/test_names.hpp>

using SushiAI::DType;

namespace
{
    /** @brief One element type with the width in bytes and the diagnostic name it must have. */
    struct ElementTypeRow
    {
        DType type;
        std::size_t width;
        const char* name;
    };

    /** @brief Lists every element type with its width and its diagnostic name. */
    constexpr std::array<ElementTypeRow, 8> K_ELEMENT_TYPE_ROWS = {{
        {DType::Half, 2U, "half"},
        {DType::Float32, 4U, "float32"},
        {DType::Float64, 8U, "float64"},
        {DType::Complex32, 8U, "complex32"},
        {DType::Complex64, 16U, "complex64"},
        {DType::Int32, 4U, "int32"},
        {DType::Int64, 8U, "int64"},
        {DType::UInt32, 4U, "uint32"},
    }};

    /** @brief Runs one case per element type of K_ELEMENT_TYPE_ROWS. */
    class Small_DTypeRow : public ::testing::TestWithParam<ElementTypeRow>
    {

    };

    /** @brief Names a Small_DTypeRow case after the diagnostic name of its element type. */
    std::string element_type_case_name(const ::testing::TestParamInfo<ElementTypeRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    }
} // namespace

INSTANTIATE_TEST_SUITE_P(, Small_DTypeRow, ::testing::ValuesIn(K_ELEMENT_TYPE_ROWS),
                         element_type_case_name);

/** @brief Tests that an element type has the width in bytes that its row gives. */
TEST_P(Small_DTypeRow, HasItsWidthInBytes)
{
    EXPECT_EQ(SushiAI::dtype_size(GetParam().type), GetParam().width);
}

/** @brief Tests that an element type has the diagnostic name that its row gives. */
TEST_P(Small_DTypeRow, HasItsDiagnosticName)
{
    EXPECT_STREQ(SushiAI::dtype_name(GetParam().type), GetParam().name);
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

/**
 * @brief Tests that a SushiBLAS tensor of rank K_MAX_RANK, every extent filled, reads back as a
 * Shape of that rank with each extent in place.
 */
TEST(Small_DType, TheMaximumRankIsTheRankSushiBLASStores)
{
    SushiBLAS::Tensor tensor;
    tensor.rank = SushiAI::K_MAX_RANK;
    for (int axis = 0; axis < SushiAI::K_MAX_RANK; ++axis)
        tensor.shape[axis] = axis + 2;

    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Shape shape, SushiAI::checked_shape_of(tensor));

    ASSERT_EQ(shape.rank(), SushiAI::K_MAX_RANK);
    for (int axis = 0; axis < SushiAI::K_MAX_RANK; ++axis)
        EXPECT_EQ(shape[axis], axis + 2);
}
