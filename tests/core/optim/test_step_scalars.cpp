/**************************************************************************/
/* test_step_scalars.cpp                                                  */
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
 * @file test_step_scalars.cpp
 * @brief Tests that StepScalars holds a zeroed buffer at one address and rewrites it in place.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/step_scalars.hpp>

using namespace SushiAI;
using Optim::StepScalars;

namespace
{
    /** @brief Provides the host engine every buffer is allocated from. */
    class Small_StepScalars : public ::testing::Test
    {
        protected:
            /** @brief Reads the slots of @p scalars into a vector. */
            static std::vector<float> slots_of(const StepScalars& scalars)
            {
                return std::vector<float>(scalars.device_pointer(),
                                          scalars.device_pointer() + scalars.count());
            }

            /** @brief Holds the host adapter the buffers are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Provides the host engine to the case that stops the process. */
    class Medium_StepScalarsFatal : public Small_StepScalars
    {
    };

    /** @brief Holds the slot count of every buffer of this file. */
    constexpr std::size_t K_COUNT = 3;

    /** @brief Views @p values as the span write() takes. */
    SushiRuntime::span<const float> span_of(const std::vector<float>& values)
    {
        return SushiRuntime::span<const float>(values.data(), values.size());
    }
} // namespace

/** @brief Tests that a buffer of no slot is refused. */
TEST_F(Small_StepScalars, AZeroCountIsRefused)
{
    SA_EXPECT_FAILURE(StepScalars::make(backend.engine(), 0), StatusCode::InvalidDescription,
                      "at least one slot");
}

/** @brief Tests that a new buffer holds its count of zeros. */
TEST_F(Small_StepScalars, ANewBufferHoldsZeros)
{
    SA_ASSERT_OK_AND_ASSIGN(const StepScalars scalars,
                            StepScalars::make(backend.engine(), K_COUNT));

    ASSERT_NE(scalars.device_pointer(), nullptr);
    EXPECT_EQ(scalars.count(), K_COUNT);
    EXPECT_EQ(slots_of(scalars), std::vector<float>(K_COUNT, 0.0f));
}

/** @brief Tests that a write replaces every value. */
TEST_F(Small_StepScalars, AWriteReplacesEveryValue)
{
    SA_ASSERT_OK_AND_ASSIGN(StepScalars scalars, StepScalars::make(backend.engine(), K_COUNT));
    const std::vector<float> values{0.5f, -2.0f, 3.25f};

    scalars.write(span_of(values));

    EXPECT_EQ(slots_of(scalars), values);
}

/** @brief Tests that a write leaves the buffer at the address a kernel captured. */
TEST_F(Small_StepScalars, AWriteKeepsTheAddress)
{
    SA_ASSERT_OK_AND_ASSIGN(StepScalars scalars, StepScalars::make(backend.engine(), K_COUNT));
    const float* const before = scalars.device_pointer();

    scalars.write(span_of(std::vector<float>{0.5f, -2.0f, 3.25f}));

    EXPECT_EQ(scalars.device_pointer(), before);
}

/** @brief Tests that a moved buffer keeps the address a kernel captured. */
TEST_F(Small_StepScalars, AMoveKeepsTheAddress)
{
    SA_ASSERT_OK_AND_ASSIGN(StepScalars scalars, StepScalars::make(backend.engine(), K_COUNT));
    const float* const captured = scalars.device_pointer();

    const StepScalars moved(std::move(scalars));

    EXPECT_EQ(moved.device_pointer(), captured);
    EXPECT_EQ(moved.count(), K_COUNT);
}

/** @brief Tests that writing a count of values other than the buffer's stops the process. */
TEST_F(Medium_StepScalarsFatal, AWriteOfAnotherLengthIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(StepScalars scalars, StepScalars::make(backend.engine(), K_COUNT));
    const std::vector<float> short_values{1.0f, 2.0f};

    SA_EXPECT_FATAL(scalars.write(span_of(short_values)), "holds 3 slots", "2 values");
}
