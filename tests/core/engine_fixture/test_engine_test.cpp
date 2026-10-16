/**************************************************************************/
/* test_engine_test.cpp                                                   */
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
 * @file test_engine_test.cpp
 * @brief Tests that EngineTest gives each test an engine that records and flushes.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Runs the cases on the fixture under test. */
    class Medium_EngineTest : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that SetUp leaves the case an engine. */
TEST_F(Medium_EngineTest, SetUpGivesTheCaseAnEngine)
{
    EXPECT_NE(engine, nullptr);
}

/** @brief Tests that flush returns with nothing recorded. */
TEST_F(Medium_EngineTest, FlushReturnsWithNothingRecorded)
{
    flush();

    SUCCEED();
}

/** @brief Tests that the engine allocates on the shared runtime and flushes afterwards. */
TEST_F(Medium_EngineTest, TheEngineAllocatesATensorAndFlushes)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            SushiAI::allocate_tensor(*engine, SushiAI::Shape{2, 3}));
    flush();

    EXPECT_EQ(tensor.num_elements, 6);
    EXPECT_NE(tensor.data(), nullptr);
}
