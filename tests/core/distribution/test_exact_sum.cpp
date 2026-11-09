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
 * @file test_exact_sum.cpp
 * @brief Tests which types the replicas' results add as exact counts.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiAI/distribution/exact_sum.hpp>
#include <support/test_names.hpp>

namespace
{
    using SushiBLAS::Core::DataType;
    using SushiAI::Distribution::sums_exactly;

    /** @brief Holds an element type and whether results of it add as exact counts. */
    struct TypeRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the element type. */
        DataType dtype;

        /** @brief Holds whether results of the type add as exact counts. */
        bool exact;
    };

    /** @brief Lists every element type with whether its results add as exact counts. */
    std::vector<TypeRow> type_rows()
    {
        return {
            {"Int32", DataType::Int32, true},
            {"Int64", DataType::Int64, true},
            {"RegressionUInt32IsACount", DataType::UInt32, true},
            {"Half", DataType::Half, false},
            {"Float32", DataType::Float32, false},
            {"Float64", DataType::Float64, false},
            {"Complex32", DataType::Complex32, false},
            {"Complex64", DataType::Complex64, false},
        };
    }

    /** @brief Runs each element type. */
    class Small_ExactSum : public ::testing::TestWithParam<TypeRow>
    {

    };
} // namespace

/** @brief Tests that results add with weight one exactly for the integer types, UInt32 included. */
TEST_P(Small_ExactSum, ATypeAddsAsACountExactlyWhenItIsAnInteger)
{
    EXPECT_EQ(sums_exactly(GetParam().dtype), GetParam().exact);
}

INSTANTIATE_TEST_SUITE_P(Types, Small_ExactSum, ::testing::ValuesIn(type_rows()),
                         [](const ::testing::TestParamInfo<TypeRow>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
