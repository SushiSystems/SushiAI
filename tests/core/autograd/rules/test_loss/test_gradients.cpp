/**************************************************************************/
/* test_gradients.cpp                                                 */
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
 * @file test_gradients.cpp
 * @brief Tests the backward rule of the fused log-softmax and negative log-likelihood against
 *        central differences in FLOAT64.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include "../rule_gradient_support.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/autograd/README.md.
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::ValueID;
using SaTest::GradientProbe;
using SaTest::expect_gradient_matches;

TEST_F(Medium_Autograd, FusedCrossEntropyAgainstOneHotTargets)
{
    // The fused rule gives (softmax - Y) / N; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape{4, 5}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.log_softmax(x);
    });

    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(201).fill(point, 2.0);

    expect_gradient_matches(probe, point);
}
