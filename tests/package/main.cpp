/**************************************************************************/
/* main.cpp                                                               */
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
 * @file main.cpp
 * @brief Runs one SushiBLAS operation against an installed SushiAI package, out of tree.
 * @author Mustafa Garip
 * @see tests/package/README.md, "main.cpp".
 */

#include <cstdio>
#include <SushiAI/SushiAI.hpp>
#include <SushiBLAS/SushiBLAS.hpp>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

/** @brief Runs one axpy through the installed package and prints a confirmation. */
int main()
{
    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime);

    constexpr int64_t K_N = 64;
    auto x = engine.create_tensor({K_N});
    auto y = engine.create_tensor({K_N});

    sb::BLASOps(engine).axpy(2.0f, x, y);
    engine.execute().value();
    runtime.wait().value();

    std::printf("OK: SushiAI %s ran an op through SushiBLAS %s against an installed package\n",
                SushiAI::version_string(), SUSHIBLAS_VERSION_STRING);
    return 0;
}
