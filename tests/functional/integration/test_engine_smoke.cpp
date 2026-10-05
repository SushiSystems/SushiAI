/****************************************************************/
/* test_engine_smoke.cpp                                        */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

// The real P6 "consumer-shaped example" for AI-0: proves the two-sibling D3
// dependency chain (SushiRuntime + SushiBLAS, both linked through SushiAI)
// actually compiles, links, and runs a real op end to end - not a stand-in
// for AI-1's eventual autograd op coverage.
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
