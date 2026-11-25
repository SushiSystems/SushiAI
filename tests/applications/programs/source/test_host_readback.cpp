/**************************************************************************/
/* test_host_readback.cpp                                                 */
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
 * @file test_host_readback.cpp
 * @brief Tests that a HostReadback refreshed over a device-resident tensor returns its elements.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>

#include "host_readback.hpp"

namespace
{
    using namespace SushiAI;

    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_HostReadback : public Test::EngineTest
    {

    };
} // namespace

/** @brief Tests that a refresh over a tensor only kernels address returns the written values. */
TEST_F(Medium_HostReadback, ARefreshReturnsTheValuesWrittenToADeviceTensor)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            allocate_tensor(*engine, Shape{4}, DType::Float32,
                                            TensorAccess::Kernels));
    const std::vector<float> written{1.5f, -2.0f, 0.0f, 42.25f};
    SA_EXPECT_OK(write_elements(tensor, written));
    Apps::HostReadback readback;
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::span<const float> view, readback.bind(tensor));

    SA_EXPECT_OK(readback.refresh());

    EXPECT_EQ(std::vector<float>(view.begin(), view.end()), written);
}
