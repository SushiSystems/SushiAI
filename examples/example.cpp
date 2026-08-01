/**************************************************************************/
/* example.cpp                                                           */
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

// AI-0 scope reset: SushiAI has no autograd core yet (AI-1). This example
// only proves the two-sibling D3 dependency chain (SushiRuntime + SushiBLAS)
// actually builds, links, and runs through the SushiAI library target.

#include <iostream>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

int main()
{
    sr::Execution::RuntimeContext ctx;
    sb::Engine engine(ctx);

    auto x = engine.create_tensor({4});
    auto y = engine.create_tensor({4});

    engine.blas().axpy(2.0f, x, y);
    engine.execute();
    ctx.wait_all();

    std::cout << "SushiAI " << SushiAI::version_string()
              << " ran an op through SushiBLAS " << SUSHIBLAS_VERSION_STRING
              << ".\n";

    return 0;
}
