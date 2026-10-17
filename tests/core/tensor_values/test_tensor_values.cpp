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
 * @file test_tensor_values.cpp
 * @brief Tests the functions that write and read the elements of a tensor from the host.
 * @author Mustafa Garip
 */

#include <cstddef>
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

/** @brief Tests that a float32 tensor holds each value rounded to float and reads it widened. */
TEST_F(Medium_TensorValues, AFloat32TensorHoldsEachValueRoundedToFloat)
{
    const std::vector<double> values = {0.1, -2.5, 1.0 / 3.0};
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            allocate_tensor(*engine, Shape{3}, DType::Float32));

    SaTest::write_tensor(tensor, span_of(values));
    const std::vector<double> read = SaTest::read_tensor(tensor);

    ASSERT_EQ(read.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        EXPECT_EQ(read[i], static_cast<double>(static_cast<float>(values[i])));
}

/** @brief Tests that the bytes of a tensor count its elements times the element size. */
TEST_F(Medium_TensorValues, TheBytesOfATensorAreItsElementStorage)
{
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            allocate_tensor(*engine, Shape{2, 3}, DType::Float32));
    SaTest::write_tensor(tensor, span_of(values));

    const std::vector<unsigned char> bytes = SaTest::read_tensor_bytes(tensor);

    ASSERT_EQ(bytes.size(), values.size() * sizeof(float));
    const float* const data = tensor.data_as<float>();
    for (std::size_t i = 0; i < bytes.size(); ++i)
        EXPECT_EQ(bytes[i], reinterpret_cast<const unsigned char*>(data)[i]);
}

/** @brief Tests that first_difference finds the first unequal byte, or the shorter length. */
TEST(Small_TensorValues, FirstDifferenceFindsTheFirstUnequalByte)
{
    const std::vector<unsigned char> a = {1, 2, 3, 4};

    EXPECT_EQ(SaTest::first_difference(a, {1, 2, 9, 4}), 2u);
    EXPECT_EQ(SaTest::first_difference(a, a), a.size());
    EXPECT_EQ(SaTest::first_difference(a, {1, 2}), 2u);
}
