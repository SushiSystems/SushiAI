/**************************************************************************/
/* test_host_copy.cpp                                                     */
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
 * @file test_host_copy.cpp
 * @brief Tests the host copies of host_copy.hpp over a tensor only kernels address.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>

using SushiAI::DType;
using SushiAI::Shape;
using SushiAI::StatusCode;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine every copy case allocates on. */
    class Medium_HostCopy : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that elements written from the host read back unchanged. */
TEST_F(Medium_HostCopy, WrittenElementsReadBackUnchanged)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            SushiAI::allocate_tensor(*engine, Shape{4}, DType::Int32));
    const std::vector<std::int32_t> written{3, -1, 7, 42};

    SA_EXPECT_OK(SushiAI::write_elements(tensor, written));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::int32_t> read,
                            SushiAI::read_elements<std::int32_t>(tensor));

    EXPECT_EQ(read, written);
}

/** @brief Tests that a copy into part of a tensor leaves the other elements as they were. */
TEST_F(Medium_HostCopy, ARangeCopyWritesOnlyItsRange)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            SushiAI::allocate_tensor(*engine, Shape{4}, DType::Float32));
    SA_EXPECT_OK(SushiAI::write_elements(tensor, std::vector<float>(4, 0.0f)));
    const std::vector<float> middle{1.5f, 2.5f};

    SA_EXPECT_OK(SushiAI::write_elements(
        tensor, 1, SushiRuntime::span<const float>(middle.data(), middle.size())));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> read,
                            SushiAI::read_elements<float>(tensor));

    EXPECT_EQ(read, (std::vector<float>{0.0f, 1.5f, 2.5f, 0.0f}));
}

/** @brief Tests that the real copies convert both float types through double. */
TEST_F(Medium_HostCopy, RealsRoundTripThroughBothFloatTypes)
{
    const std::vector<double> written{1.0, -2.0, 4.0};
    for (const DType dtype : {DType::Float32, DType::Float64})
    {
        SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                                SushiAI::allocate_tensor(*engine, Shape{3}, dtype));

        SA_EXPECT_OK(SushiAI::write_reals(
            tensor, SushiRuntime::span<const double>(written.data(), written.size())));
        SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> read, SushiAI::read_reals(tensor));

        EXPECT_EQ(read, written) << SushiAI::dtype_name(dtype);
    }
}

/** @brief Tests that a host element of another width than the tensor's is refused. */
TEST_F(Medium_HostCopy, AnotherElementWidthIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            SushiAI::allocate_tensor(*engine, Shape{2}, DType::Float32));
    std::vector<double> destination(2);

    SA_EXPECT_FAILURE(SushiAI::read_elements(tensor, 0,
                                             SushiRuntime::span<double>(destination.data(), 2)),
                      StatusCode::InvalidDescription);
}

/** @brief Tests that a range that runs past the last element is refused. */
TEST_F(Medium_HostCopy, ARangePastTheLastElementIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            SushiAI::allocate_tensor(*engine, Shape{2}, DType::Float32));
    const std::vector<float> source{1.0f, 2.0f};

    SA_EXPECT_FAILURE(SushiAI::write_elements(
                          tensor, 1, SushiRuntime::span<const float>(source.data(), 2)),
                      StatusCode::InvalidDescription);
}
