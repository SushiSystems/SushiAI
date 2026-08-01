/**************************************************************************/
/* main.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
