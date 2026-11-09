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
 * @file test_host_values.cpp
 * @brief Tests the host read of one element of a float32 or float64 tensor.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/host_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the elements of one tensor, the element read and the double it reads as. */
    struct ReadRow
    {
        std::string name;               /**< The row's name. */
        DType dtype;                    /**< The element type of the tensor. */
        std::vector<double> values;     /**< The elements written, as the type holds them. */
        std::size_t index;              /**< The element read. */
        double expected;                /**< The double the read returns. */
    };

    /** @brief Reads one element of a tensor on the host backend for each row. */
    class Small_HostValues : public ::testing::TestWithParam<ReadRow>
    {
        protected:
            /** @brief The host backend the tensor is allocated from. */
            HostExecutionBackend backend;
    };
} // namespace

/**
 * @brief Tests that a tensor of either type is accepted and an element reads back as the
 *        double the type holds. Reading rounds nothing, so the comparison is equality.
 */
TEST_P(Small_HostValues, AnElementIsReadAsTheDoubleItsTypeHolds)
{
    const ReadRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            allocate_tensor(backend.engine(),
                                            Shape{static_cast<std::int64_t>(row.values.size())},
                                            row.dtype));
    SaTest::write_tensor(tensor, row.values);

    SA_EXPECT_OK(Train::require_host_reals(tensor, static_cast<std::int64_t>(row.values.size()),
                                           "the scores"));
    EXPECT_EQ(Train::host_real(tensor, static_cast<std::int64_t>(row.index)), row.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_HostValues,
    ::testing::Values(
        ReadRow{"Float32HoldingExactValues", DType::Float32, {0.5, -1.25, 3.0}, 1, -1.25},
        ReadRow{"Float32RoundsToTheNearestFloat", DType::Float32, {0.1, 0.2}, 0,
                static_cast<double>(0.1f)},
        ReadRow{"Float64KeepsEveryBit", DType::Float64, {0.1, 0.2}, 0, 0.1}),
    [](const ::testing::TestParamInfo<ReadRow>& info) { return info.param.name; });
