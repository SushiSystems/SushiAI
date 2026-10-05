/****************************************************************/
/* main.cpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file main.cpp
 * @brief Runs one SushiBLAS operation against an installed SushiAI package, out of tree.
 * @author Mustafa Garip
 * @see tests/package/README.md, "main.cpp".
 */

#include <cstdio>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

/** @brief Runs one axpy through the installed package and prints a confirmation. */
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
