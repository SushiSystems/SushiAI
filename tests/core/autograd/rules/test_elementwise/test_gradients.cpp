/**************************************************************************/
/* test_gradients.cpp                                          */
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
 * @brief Tests the backward rules of add, sub, mul, scale, exp, accumulate, sqrt, log and pow
 *        against central differences in FLOAT64.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/test_names.hpp>
#include "../rule_gradient_support.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/autograd/README.md.
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::ValueID;
using SaTest::GradientProbe;
using SaTest::expect_gradient_matches;

namespace
{
    /** @brief Holds one operation of the add-or-sub case: its name, which one, a seed offset. */
    struct BinaryCase
    {
        const char* name;          /**< The case name. */
        bool subtract;             /**< True for sub, false for add. */
        std::uint64_t seed_offset; /**< Added to each seed of the case. */
    };

    /** @brief Runs the add-or-sub case once per operation. */
    class Medium_AutogradBinary : public SaTest::EngineTest,
                                  public ::testing::WithParamInterface<BinaryCase>
    {
    };

    /** @brief Names a power and an exponent. */
    struct PowerCase
    {
        const char* name; /**< The case name. */
        double exponent;  /**< The exponent of the power. */
    };

    /** @brief Runs the power case once per exponent. */
    class Medium_AutogradPow : public SaTest::EngineTest,
                               public ::testing::WithParamInterface<PowerCase>
    {
    };
} // namespace

/** @brief Tests that the variable on the left of add, and of sub, differentiates against central
 *         differences.
 */
TEST_P(Medium_AutogradBinary, ElementwiseAddAndSubtract)
{
    const BinaryCase& row = GetParam();
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 3},
                        [&other, &row](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "b", DType::Float64);
        return row.subtract ? builder.sub(x, other) : builder.add(x, other);
    });

    probe.fill_noise(other, 81 + row.seed_offset, 1.0);
    probe.fill_noise(probe.targets(), 91 + row.seed_offset, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(101 + row.seed_offset).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

INSTANTIATE_TEST_SUITE_P(Operations, Medium_AutogradBinary,
                         ::testing::Values(BinaryCase{"Add", false, 0}, BinaryCase{"Sub", true, 1}),
                         [](const ::testing::TestParamInfo<BinaryCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });

TEST_F(Medium_Autograd, ElementwiseSubtractFromTheRight)
{
    // dB = -dC, the half the left-hand check cannot reach.
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "a", DType::Float64);
        return builder.sub(other, x);
    });

    probe.fill_noise(other, 111, 1.0);
    probe.fill_noise(probe.targets(), 112, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(113).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ElementwiseMultiply)
{
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "b", DType::Float64);
        return builder.mul(x, other);
    });

    probe.fill_noise(other, 121, 1.0);
    probe.fill_noise(probe.targets(), 122, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(123).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ElementwiseMultiplyFromTheRight)
{
    // Covers dB = dC * A, the right-hand half; see tests/core/autograd/README.md.
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "a", DType::Float64);
        return builder.mul(other, x);
    });

    probe.fill_noise(other, 124, 1.0);
    probe.fill_noise(probe.targets(), 125, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(126).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AccumulateInTheForwardGraph)
{
    // Every probe rewrites the whole variable first; see tests/core/autograd/README.md.
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "delta", DType::Float64);
        return builder.accumulate(x, other);
    });

    probe.fill_noise(other, 231, 1.0);
    probe.fill_noise(probe.targets(), 232, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(233).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Scale)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.scale(x, -2.5);
    });

    probe.fill_noise(probe.targets(), 131, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(132).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Exp)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.exp(x);
    });

    probe.fill_noise(probe.targets(), 141, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(142).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, GradientAccumulationAtAFanOut)
{
    // Two branches over one variable make the gradient an ACCUMULATE chain.
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.add(builder.tanh(x), builder.sigmoid(x));
    });

    probe.fill_noise(probe.targets(), 211, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(212).fill(point, 1.2);

    expect_gradient_matches(probe, point);
}

/** @brief Tests that the square root differentiates against central differences above 0.1. */
TEST_F(Medium_Autograd, SquareRootReadsItsForwardResult)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& b, ValueID x)
    {
        return b.sqrt(x);
    });

    probe.fill_noise(probe.targets(), 701, 1.0);

    std::vector<double> point = SaTest::positive_point(probe.size(), 1);

    expect_gradient_matches(probe, point);
}

/** @brief Tests that the logarithm differentiates against central differences above 0.1. */
TEST_F(Medium_Autograd, LogarithmDividesByItsInput)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& b, ValueID x)
    {
        return b.log(x);
    });

    probe.fill_noise(probe.targets(), 702, 1.0);

    std::vector<double> point = SaTest::positive_point(probe.size(), 2);

    expect_gradient_matches(probe, point);
}

/** @brief Tests that the power differentiates at the exponents 2, 0.5 and -0.5. */
TEST_P(Medium_AutogradPow, ThePowerDifferentiatesAtEachExponent)
{
    const PowerCase& row = GetParam();

    GradientProbe probe(backend(), Shape{4, 3}, [&row](const GraphBuilder& b, ValueID x)
    {
        return b.pow(x, row.exponent);
    });

    probe.fill_noise(probe.targets(), 703, 1.0);

    std::vector<double> point = SaTest::positive_point(probe.size(), 3);

    expect_gradient_matches(probe, point);
}

INSTANTIATE_TEST_SUITE_P(Exponents, Medium_AutogradPow,
                         ::testing::Values(PowerCase{"Two", 2.0}, PowerCase{"OneHalf", 0.5},
                                           PowerCase{"MinusOneHalf", -0.5}),
                         [](const ::testing::TestParamInfo<PowerCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
