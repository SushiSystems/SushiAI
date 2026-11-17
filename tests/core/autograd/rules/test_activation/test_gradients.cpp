/**************************************************************************/
/* test_gradients.cpp                                           */
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
 * @brief Tests the backward rules of relu, leaky relu, gelu, sigmoid and tanh against central
 *        differences in FLOAT64.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
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
using SaTest::GradientProbeOptions;
using SaTest::expect_gradient_matches;

namespace
{
    /** @brief Reports whether the lowering of @p probe chose @p form for any selection. */
    bool spells_form(const GradientProbe& probe, Graph::FusedForm form)
    {
        const SushiRuntime::span<const Graph::FusionSelection> selections = probe.selections();

        return std::any_of(selections.begin(), selections.end(),
                           [form](const Graph::FusionSelection& selection)
                           {
                               return selection.chosen == form;
                           });
    }
} // namespace

TEST_F(Medium_Autograd, ReLUAwayFromItsKink)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.relu(x);
    });

    probe.fill_noise(probe.targets(), 151, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(152).fill(point, 1.0);

    // ReLU is sampled clear of its kink at zero; see tests/core/autograd/README.md.
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, LeakyReLUHasBothOfItsSlopes)
{
    // The sample range puts elements on both slopes; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape{4, 8}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.log_softmax(builder.leaky_relu(x, 0.1));
    });

    probe.fill_one_hot(4, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(171).fill(point, 1.5);

    // LEAKY_RELU is not differentiable at zero, same as RELU.
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, GELU)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.gelu(x);
    });

    probe.fill_noise(probe.targets(), 161, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(162).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AFusedGELUEpilogueHasTheSameDerivativeAsAnUnfusedOne)
{
    // The fused epilogue is checked in FLOAT64; see tests/core/autograd/README.md.
    ValueID weight = Graph::K_NO_VALUE;
    ValueID bias = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 5},
                        [&](const GraphBuilder& builder, ValueID x)
                        {
                            // x is the variable: matmul -> bias_add -> gelu gives [4, 3] logits.
                            weight = builder.input(Shape{3, 5}, "w", DType::Float64);
                            bias = builder.input(Shape{3}, "b", DType::Float64);
                            return builder.gelu(
                                builder.bias_add(builder.matmul(x, weight, false, true), bias));
                        },
                        GradientProbeOptions::fused());

    // The fused arm must really have run; see tests/core/autograd/README.md.
    ASSERT_EQ(probe.fusion().count(Graph::FusedForm::GEMMBiasGELU), 1u)
        << probe.fusion().to_string();

    ASSERT_TRUE(spells_form(probe, Graph::FusedForm::GEMMBiasGELU))
        << "this device declined the fused GELU epilogue, so the gradient below is the "
           "unfused one and this case is not checking what it claims to";

    probe.fill_noise(weight, 171, 0.8);
    probe.fill_noise(bias, 172, 0.5);
    probe.fill_one_hot(4, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(173).fill(point, 1.2);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SigmoidReadsItsForwardOutput)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.sigmoid(x);
    });

    probe.fill_noise(probe.targets(), 171, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(172).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, TanhReadsItsForwardOutput)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.tanh(x);
    });

    probe.fill_noise(probe.targets(), 181, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(182).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}
