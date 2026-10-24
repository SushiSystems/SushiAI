/**************************************************************************/
/* test_SushiAI.cpp                                                       */
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
 * @file test_SushiAI.cpp
 * @brief Tests that an axpy runs through SushiBLAS when linked through the umbrella header,
 *        and that a distributed build names every distribution header through it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiAI/SushiAI.hpp>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

// SushiAI links and runs one SushiBLAS op end to end; see tests/layers/umbrella/README.md.
TEST(Medium_EngineSmoke, AXPYThroughSushiBLAS)
{
    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime);

    const int64_t n = 4;
    auto x = engine.create_tensor({n});
    auto y = engine.create_tensor({n});

    std::vector<float> ones(static_cast<size_t>(n), 1.0f);
    sycl::queue& q = runtime.advanced().queue(sr::DeviceIndex{0});
    q.memcpy(x.data_as<float>(), ones.data(), ones.size() * sizeof(float)).wait();
    q.memcpy(y.data_as<float>(), ones.data(), ones.size() * sizeof(float)).wait();

    sb::LateCountHold late_counts(runtime);
    sr::Graph graph = runtime.graph();
    sb::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), late_counts,
                              engine.mode());
    sb::BLASOps(recorder).axpy(2.0f, x, y);
    graph.run().value();
    runtime.wait().value();

    std::vector<float> result(static_cast<size_t>(n));
    q.memcpy(result.data(), y.data_as<float>(), result.size() * sizeof(float)).wait();

    for (float v : result)
        EXPECT_FLOAT_EQ(v, 3.0f); // y = 2*x + y = 2*1 + 1
}

// tests/layers/umbrella/CMakeLists.txt defines SA_TEST_DISTRIBUTED_BUILD in a distributed build.
#if defined(SA_TEST_DISTRIBUTED_BUILD)
// Compiles only when the library carries SA_DISTRIBUTED and the umbrella includes the headers.
TEST(Small_UmbrellaDistribution, RegressionNamesEveryDistributionHeader)
{
    EXPECT_TRUE(std::is_class<SushiAI::Distribution::ClusterSection>::value);
    EXPECT_TRUE(std::is_class<SushiAI::Distribution::GradientExchange>::value);
    EXPECT_TRUE(std::is_class<SushiAI::Distribution::ParameterPublish>::value);
    EXPECT_TRUE(std::is_class<SushiAI::Distribution::ReplicaPosition>::value);
    EXPECT_TRUE(std::is_class<SushiAI::Distribution::RowPositions>::value);
    EXPECT_TRUE(SushiAI::Distribution::sums_exactly(SushiBLAS::Core::DataType::UInt32));
}
#endif
