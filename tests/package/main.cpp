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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file main.cpp
 * @brief Runs one SushiBLAS operation against an installed SushiAI package, out of tree.
 * @author Mustafa Garip
 * @see tests/package/README.md, "main.cpp".
 */

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <SushiAI/SushiAI.hpp>
#include <SushiBLAS/SushiBLAS.hpp>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

/** @brief Runs one axpy through the installed package and returns 1 unless y holds 2 * x + y. */
int main()
{
    sr::Runtime runtime = sr::Runtime::create().value();
    sb::Engine engine(runtime);

    constexpr int64_t K_N = 64;
    auto x = engine.create_tensor({K_N});
    auto y = engine.create_tensor({K_N});

    const std::vector<float> ones(static_cast<std::size_t>(K_N), 1.0f);
    sycl::queue& queue = runtime.advanced().queue(sr::DeviceIndex{0});
    queue.memcpy(x.data_as<float>(), ones.data(), ones.size() * sizeof(float)).wait();
    queue.memcpy(y.data_as<float>(), ones.data(), ones.size() * sizeof(float)).wait();

    sb::LateCountHold late_counts(runtime);
    sr::Graph graph = runtime.graph();
    sb::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), late_counts,
                              engine.mode());
    sb::BLASOps(recorder).axpy(2.0f, x, y);
    graph.run().value();
    runtime.wait().value();

    std::vector<float> result(static_cast<std::size_t>(K_N));
    queue.memcpy(result.data(), y.data_as<float>(), result.size() * sizeof(float)).wait();

    // y = 2 * x + y = 2 * 1 + 1, exact in float32.
    const bool right = std::all_of(result.begin(), result.end(),
                                   [](float value) { return value == 3.0f; });
    if (!right)
    {
        std::fprintf(stderr, "FAIL: the axpy of the installed package did not yield 2 * x + y
");
        return 1;
    }

    std::printf("OK: SushiAI %s ran an op through SushiBLAS %s against an installed package
",
                SushiAI::version_string(), SUSHIBLAS_VERSION_STRING);
    return 0;
}
