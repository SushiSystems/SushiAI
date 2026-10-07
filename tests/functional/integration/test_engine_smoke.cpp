/**************************************************************************/
/* test_engine_smoke.cpp                                                  */
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
 * @file test_engine_smoke.cpp
 * @brief Tests that an axpy runs through SushiBLAS when linked through SushiAI.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

// SushiAI links and runs one SushiBLAS op end to end; see tests/functional/integration/README.md.
TEST(Integration_EngineSmoke, AxpyThroughSushiBLAS)
{
    sr::Execution::RuntimeContext ctx;
    sb::Engine engine(ctx);

    const int64_t n = 4;
    auto x = engine.create_tensor({n});
    auto y = engine.create_tensor({n});

    std::vector<float> ones(static_cast<size_t>(n), 1.0f);
    auto& q = ctx.get_queue();
    q.memcpy(x.data_as<float>(), ones.data(), ones.size() * sizeof(float)).wait();
    q.memcpy(y.data_as<float>(), ones.data(), ones.size() * sizeof(float)).wait();

    engine.blas().axpy(2.0f, x, y);
    engine.execute();
    ctx.wait_all();

    std::vector<float> result(static_cast<size_t>(n));
    q.memcpy(result.data(), y.data_as<float>(), result.size() * sizeof(float)).wait();

    for (float v : result)
        EXPECT_FLOAT_EQ(v, 3.0f); // y = 2*x + y = 2*1 + 1
}
