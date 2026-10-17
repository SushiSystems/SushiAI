/**************************************************************************/
/* test_sgd.cpp                                                           */
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
 * @file test_sgd.cpp
 * @brief Tests what SGD::make() refuses and the update it applies with and without momentum.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using Optim::SGD;
using Optim::SGDOptions;

// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the value every parameter element starts at. */
    constexpr double K_START = 1.0;

    /** @brief Bounds the absolute error of a float32 value of order one after two updates. */
    constexpr double K_UPDATE_TOLERANCE = 1e-6;

    /** @brief Provides an engine and one parameter slot with a gradient the case chooses. */
    class Medium_SGD : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns one slot "weight" of K_START with the gradient @p gradient. */
            std::vector<Optim::ParameterSlot> make_slots(const std::vector<double>& gradient)
            {
                const int64_t elements = static_cast<int64_t>(gradient.size());
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(*engine, Shape{elements}, DType::Float32);
                Result<SushiBLAS::Tensor> slope =
                    allocate_tensor(*engine, Shape{elements}, DType::Float32);
                EXPECT_TRUE(parameter.status().is_ok());
                EXPECT_TRUE(slope.status().is_ok());

                SaTest::write_tensor(parameter.value(),
                                     std::vector<double>(gradient.size(), K_START));
                SaTest::write_tensor(slope.value(), gradient);

                std::vector<Optim::ParameterSlot> slots;
                slots.push_back(Optim::ParameterSlot{"weight", std::move(parameter).value(),
                                                     std::move(slope).value()});
                return slots;
            }

            /** @brief Steps @p optimizer once and returns its one parameter's values. */
            std::vector<double> step_once(SGD& optimizer)
            {
                const SushiBLAS::Tensor parameter = optimizer.parameters()[0].parameter;
                SA_EXPECT_OK(optimizer.step());
                flush();
                return SaTest::read_tensor(parameter);
            }
    };
} // namespace

/** @brief Tests that SGD names itself "sgd" and replays only with the fused kernel. */
TEST_F(Medium_SGD, TheTypeAndReplayFollowTheKernel)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> fused,
                            SGD::make(*engine, make_slots({1.0})));
    SGDOptions separate_options;
    separate_options.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> separate,
                            SGD::make(*engine, make_slots({1.0}), separate_options));

    EXPECT_EQ(fused->type_name(), Optim::K_SGD_TYPE);
    EXPECT_TRUE(fused->is_replay_invariant());
    EXPECT_FALSE(separate->is_replay_invariant());
}

/** @brief Tests that Nesterov is refused without momentum and with dampening. */
TEST_F(Medium_SGD, NesterovNeedsMomentumAndNoDampening)
{
    SGDOptions no_momentum;
    no_momentum.nesterov = true;
    SGDOptions damped;
    damped.nesterov = true;
    damped.momentum = 0.9;
    damped.dampening = 0.1;

    SA_EXPECT_FAILURE(SGD::make(*engine, make_slots({1.0}), no_momentum),
                      StatusCode::InvalidDescription, "positive momentum");
    SA_EXPECT_FAILURE(SGD::make(*engine, make_slots({1.0}), damped),
                      StatusCode::InvalidDescription, "undefined with dampening");
}

/** @brief Tests that a step without momentum subtracts the rate times the gradient. */
TEST_F(Medium_SGD, APlainStepSubtractsTheScaledGradient)
{
    const std::vector<double> gradient{2.0, -1.0, 0.5};
    SGDOptions options;
    options.learning_rate = 0.5;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> optimizer,
                            SGD::make(*engine, make_slots(gradient), options));

    const std::vector<double> values = step_once(*optimizer);

    ASSERT_EQ(values.size(), gradient.size());
    for (std::size_t i = 0; i < gradient.size(); ++i)
        EXPECT_NEAR(values[i], K_START - 0.5 * gradient[i], K_UPDATE_TOLERANCE) << "element " << i;
}

/** @brief Tests that momentum starts from the gradient and then accumulates it. */
TEST_F(Medium_SGD, MomentumAccumulatesFromTheSecondStep)
{
    SGDOptions options;
    options.learning_rate = 0.1;
    options.momentum = 0.9;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> optimizer,
                            SGD::make(*engine, make_slots({1.0}), options));

    // The buffer is the gradient, 1, after the first step and 0.9 * 1 + 1 after the second.
    const std::vector<double> first = step_once(*optimizer);
    const std::vector<double> second = step_once(*optimizer);

    EXPECT_NEAR(first[0], K_START - 0.1, K_UPDATE_TOLERANCE);
    EXPECT_NEAR(second[0], K_START - 0.1 - 0.1 * 1.9, K_UPDATE_TOLERANCE);
}
