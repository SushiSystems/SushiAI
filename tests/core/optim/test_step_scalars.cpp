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
 * @file test_step_scalars.cpp
 * @brief Tests that StepScalars holds a zeroed buffer at one address and rewrites it in place.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/step_scalars.hpp>

using namespace SushiAI;
using Optim::StepScalars;

// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the Engine every buffer is allocated from. */
    class Medium_StepScalars : public SaTest::EngineTest
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
TEST_F(Medium_StepScalars, AZeroCountIsRefused)
{
    SA_EXPECT_FAILURE(StepScalars::make(*engine, 0), StatusCode::InvalidDescription,
                      "at least one slot");
}

/** @brief Tests that a new buffer holds its count of zeros. */
TEST_F(Medium_StepScalars, ANewBufferHoldsZeros)
{
    SA_ASSERT_OK_AND_ASSIGN(const StepScalars scalars, StepScalars::make(*engine, K_COUNT));

    ASSERT_EQ(scalars.count(), K_COUNT);
    ASSERT_NE(scalars.device_pointer(), nullptr);
    for (std::size_t i = 0; i < K_COUNT; ++i)
        EXPECT_EQ(scalars.device_pointer()[i], 0.0f) << "slot " << i;
}

/** @brief Tests that a write replaces every value and leaves the address where it was. */
TEST_F(Medium_StepScalars, AWriteKeepsTheAddress)
{
    SA_ASSERT_OK_AND_ASSIGN(StepScalars scalars, StepScalars::make(*engine, K_COUNT));
    const float* const before = scalars.device_pointer();
    const std::vector<float> values{0.5f, -2.0f, 3.25f};

    scalars.write(span_of(values));

    EXPECT_EQ(scalars.device_pointer(), before);
    for (std::size_t i = 0; i < K_COUNT; ++i)
        EXPECT_EQ(scalars.device_pointer()[i], values[i]) << "slot " << i;
}

/** @brief Tests that a moved buffer keeps the address a kernel captured, and cannot be copied. */
TEST_F(Medium_StepScalars, AMoveKeepsTheAddress)
{
    EXPECT_FALSE(std::is_copy_constructible<StepScalars>::value);
    EXPECT_FALSE(std::is_copy_assignable<StepScalars>::value);

    SA_ASSERT_OK_AND_ASSIGN(StepScalars scalars, StepScalars::make(*engine, K_COUNT));
    const float* const captured = scalars.device_pointer();

    const StepScalars moved(std::move(scalars));

    EXPECT_EQ(moved.device_pointer(), captured);
    EXPECT_EQ(moved.count(), K_COUNT);
}

/** @brief Tests that writing a count of values other than the buffer's stops the process. */
TEST_F(Medium_StepScalars, AWriteOfAnotherLengthIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(StepScalars scalars, StepScalars::make(*engine, K_COUNT));
    const std::vector<float> short_values{1.0f, 2.0f};

    SA_EXPECT_FATAL(scalars.write(span_of(short_values)), "holds 3 slots", "2 values");
}
