/**************************************************************************/
/* test_test_graph.cpp                                                    */
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

/**
 * @file test_test_graph.cpp
 * @brief Tests that TestGraph records on a graph of its own and runs it once.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_graph.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Runs the cases on the fixture the graph is built from. */
    class Medium_TestGraph : public SaTest::EngineTest
    {

    };

    /** @brief Holds the value the recorded fill writes. */
    constexpr float K_FILL_VALUE = 7.0f;
} // namespace

/** @brief Tests that run succeeds on a graph with nothing recorded. */
TEST_F(Medium_TestGraph, RunSucceedsWithNothingRecorded)
{
    SaTest::TestGraph recording(*engine);

    SA_EXPECT_OK(recording.run());
}

/** @brief Tests that run executes what was recorded through recorder. */
TEST_F(Medium_TestGraph, RunExecutesWhatTheRecorderRecorded)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            SushiAI::allocate_tensor(*engine, SushiAI::Shape{4}));
    SaTest::TestGraph recording(*engine);

    SushiBLAS::RandomOps random(recording.recorder(), engine->rng());
    random.constant(tensor, K_FILL_VALUE);
    SA_EXPECT_OK(recording.run());
    flush();

    const float* values = tensor.data_as<float>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(values[i], K_FILL_VALUE);
}
