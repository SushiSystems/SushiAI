/****************************************************************/
/* main.cpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The smallest thing that proves the package works: a real op through both
// transitively-resolved siblings, run against an installed SushiAI, from a
// project with no access to this source tree.

#include <cstdio>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

int main()
{
    sr::Execution::RuntimeContext ctx;
    sb::Engine engine(ctx);

    constexpr int64_t N = 64;
    auto x = engine.create_tensor({N});
    auto y = engine.create_tensor({N});

    engine.blas().axpy(2.0f, x, y);
    engine.execute();
    ctx.wait_all();

    std::printf("OK: SushiAI %s ran an op through SushiBLAS %s against an installed package\n",
                SushiAI::version_string(), SUSHIBLAS_VERSION_STRING);
    return 0;
}
