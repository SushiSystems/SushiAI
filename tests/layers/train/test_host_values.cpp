/**************************************************************************/
/* test_host_values.cpp                                                   */
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
 * @file test_host_values.cpp
 * @brief Tests the host read of one element of a float32 or float64 tensor.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/host_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Bounds the difference between a stored element and the double read back. */
    constexpr double K_READ_TOLERANCE = 0.0;

    /** @brief Provides the engine the tensors are allocated on. */
    class Medium_HostValues : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a tensor of @p shape and @p dtype that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape, DType dtype,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(*engine, shape, dtype));
                SaTest::write_tensor(tensor, values);
                return tensor;
            }
    };
} // namespace

/** @brief Tests that a float32 tensor is accepted and its element read back as a double. */
TEST_F(Medium_HostValues, AFloat32ElementIsReadAsADouble)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            tensor_of(Shape{3}, DType::Float32, {0.5, -1.25, 3.0}));

    SA_EXPECT_OK(Train::require_host_reals(tensor, 3, "the scores"));
    EXPECT_NEAR(Train::host_real(tensor, 1), -1.25, K_READ_TOLERANCE);
    EXPECT_NEAR(Train::host_real(tensor, 2), 3.0, K_READ_TOLERANCE);
}

/** @brief Tests that a float64 element comes back unrounded, unlike its float32 neighbour. */
TEST_F(Medium_HostValues, AFloat64ElementIsReadUnrounded)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor wide,
                            tensor_of(Shape{2}, DType::Float64, {0.1, 0.2}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor narrow,
                            tensor_of(Shape{2}, DType::Float32, {0.1, 0.2}));

    SA_EXPECT_OK(Train::require_host_reals(wide, 2, "the scores"));
    EXPECT_NEAR(Train::host_real(wide, 0), 0.1, K_READ_TOLERANCE);
    EXPECT_NEAR(Train::host_real(narrow, 0), static_cast<double>(0.1f), K_READ_TOLERANCE);
    EXPECT_NE(Train::host_real(narrow, 0), 0.1);
}
