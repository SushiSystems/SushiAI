/**************************************************************************/
/* test_tensor_values.cpp                                                 */
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
 * @file test_tensor_values.cpp
 * @brief Tests the functions that write and read the elements of a tensor from the host.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the tensors are allocated on. */
    class Medium_TensorValues : public SaTest::EngineTest
    {

    };

    /** @brief Returns @p values as the span write_tensor reads. */
    SushiRuntime::span<const double> span_of(const std::vector<double>& values)
    {
        return SushiRuntime::span<const double>(values.data(), values.size());
    }
} // namespace

/** @brief Tests that a float64 tensor reads back exactly what was written. */
TEST_F(Medium_TensorValues, AFloat64TensorRoundTripsExactly)
{
    const std::vector<double> values = {0.1, -2.5, 1e-300, 3.0};
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            allocate_tensor(*engine, Shape{4}, DType::Float64));

    SaTest::write_tensor(tensor, span_of(values));

    EXPECT_EQ(SaTest::read_tensor(tensor), values);
}

/**
 * @brief Tests that rounding to float32 gives the nearest float of each value.
 *
 * The oracle is the nearest float written as a hexadecimal literal: 0.1 lies between
 * 0x1.99999ap-4 and 0x1.999998p-4 and is nearer the first, one third lies between 0x1.555556p-2
 * and 0x1.555554p-2 and is nearer the first, and -2.5 is exact.
 */
TEST(Small_TensorValues, RoundToFloat32GivesTheNearestFloatOfEachValue)
{
    const std::vector<double> values = {0.1, -2.5, 1.0 / 3.0};

    const std::vector<float> rounded = SaTest::round_to_float32(span_of(values));

    EXPECT_EQ(rounded, (std::vector<float>{0x1.99999ap-4f, -2.5f, 0x1.555556p-2f}));
}

/** @brief Tests that a float32 tensor reads back each written value rounded to float. */
TEST_F(Medium_TensorValues, AFloat32TensorHoldsEachValueRoundedToFloat)
{
    const std::vector<double> values = {0.1, -2.5, 1.0 / 3.0};
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            allocate_tensor(*engine, Shape{3}, DType::Float32));

    SaTest::write_tensor(tensor, span_of(values));

    const std::vector<double> expected = {0x1.99999ap-4, -2.5, 0x1.555556p-2};
    EXPECT_EQ(SaTest::read_tensor(tensor), expected);
}

/** @brief Tests that the bytes of a tensor count its elements times the element size. */
TEST_F(Medium_TensorValues, TheBytesOfATensorAreItsElementStorage)
{
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            allocate_tensor(*engine, Shape{2, 3}, DType::Float32));
    SaTest::write_tensor(tensor, span_of(values));

    const std::vector<unsigned char> bytes = SaTest::read_tensor_bytes(tensor);

    const std::vector<float> written = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<unsigned char> expected(written.size() * sizeof(float));
    std::memcpy(expected.data(), written.data(), expected.size());
    EXPECT_EQ(bytes, expected);
}

/** @brief Tests that first_difference finds the first unequal byte, or the shorter length. */
TEST(Small_TensorValues, FirstDifferenceFindsTheFirstUnequalByte)
{
    const std::vector<unsigned char> a = {1, 2, 3, 4};

    EXPECT_EQ(SaTest::first_difference(a, {1, 2, 9, 4}), 2u);
    EXPECT_EQ(SaTest::first_difference(a, a), a.size());
    EXPECT_EQ(SaTest::first_difference(a, {1, 2}), 2u);
}
