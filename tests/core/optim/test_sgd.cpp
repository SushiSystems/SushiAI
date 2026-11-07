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
 * @file test_sgd.cpp
 * @brief Tests what SGD::make() refuses and the update it applies with and without momentum.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Optim::SGD;
using Optim::SGDOptions;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the value every parameter element starts at. */
    constexpr double K_START = 1.0;

    /**
     * @brief Bounds the absolute error of a float32 value of order one after two updates; 32 unit
     * roundoffs of float, the smallest power of two that covers 1e-6.
     */
    constexpr double K_UPDATE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /** @brief Provides the host backend and one parameter slot with a gradient the case chooses. */
    class Small_SGD : public ::testing::Test
    {
        protected:
            /** @brief Returns one slot "weight" of K_START with the gradient @p gradient. */
            std::vector<Optim::ParameterSlot> make_slots(const std::vector<double>& gradient)
            {
                const int64_t elements = static_cast<int64_t>(gradient.size());
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(backend.engine(), Shape{elements}, DType::Float32);
                Result<SushiBLAS::Tensor> slope =
                    allocate_tensor(backend.engine(), Shape{elements}, DType::Float32);
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
                return SaTest::read_tensor(parameter);
            }

            /** @brief Holds the host adapter the optimizer allocates from and runs on. */
            HostExecutionBackend backend;
    };

    /** @brief Names one kernel and whether its update replays. */
    struct KernelRow
    {
        Optim::UpdateKernel kernel; /**< The kernel the options ask for. */
        bool replays;               /**< Whether the update is replay invariant. */
    };

    /** @brief Provides the host backend to each kernel row. */
    class Small_SGDKernel : public Small_SGD, public ::testing::WithParamInterface<KernelRow>
    {
    };
} // namespace

/** @brief Tests that the update replays only with the fused kernel. */
TEST_P(Small_SGDKernel, TheReplayFollowsTheKernel)
{
    SGDOptions options;
    options.kernel = GetParam().kernel;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> optimizer,
                            SGD::make(backend, make_slots({1.0}), options));

    EXPECT_EQ(optimizer->is_replay_invariant(), GetParam().replays);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_SGDKernel,
                         ::testing::Values(KernelRow{Optim::UpdateKernel::Fused, true},
                                           KernelRow{Optim::UpdateKernel::Separate, false}));

/** @brief Tests that Nesterov is refused without momentum. */
TEST_F(Small_SGD, NesterovNeedsMomentum)
{
    SGDOptions options;
    options.nesterov = true;

    SA_EXPECT_FAILURE(SGD::make(backend, make_slots({1.0}), options),
                      StatusCode::InvalidDescription, "positive momentum");
}

/** @brief Tests that Nesterov is refused with dampening. */
TEST_F(Small_SGD, NesterovNeedsNoDampening)
{
    SGDOptions options;
    options.nesterov = true;
    options.momentum = 0.9;
    options.dampening = 0.1;

    SA_EXPECT_FAILURE(SGD::make(backend, make_slots({1.0}), options),
                      StatusCode::InvalidDescription, "undefined with dampening");
}

/** @brief Tests that a step without momentum subtracts the rate times the gradient. */
TEST_F(Small_SGD, APlainStepSubtractsTheScaledGradient)
{
    SGDOptions options;
    options.learning_rate = 0.5;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> optimizer,
                            SGD::make(backend, make_slots({2.0, -1.0, 0.5}), options));

    EXPECT_THAT(step_once(*optimizer),
                ::testing::Pointwise(::testing::DoubleNear(K_UPDATE_TOLERANCE),
                                     std::vector<double>{0.0, 1.5, 0.75}));
}

/** @brief Tests that momentum starts from the gradient: the first step is a plain one. */
TEST_F(Small_SGD, MomentumStartsFromTheGradient)
{
    SGDOptions options;
    options.learning_rate = 0.1;
    options.momentum = 0.9;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> optimizer,
                            SGD::make(backend, make_slots({1.0}), options));

    EXPECT_NEAR(step_once(*optimizer)[0], K_START - 0.1, K_UPDATE_TOLERANCE);
}

/** @brief Tests that momentum accumulates the gradient from the second step. */
TEST_F(Small_SGD, MomentumAccumulatesFromTheSecondStep)
{
    SGDOptions options;
    options.learning_rate = 0.1;
    options.momentum = 0.9;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SGD> optimizer,
                            SGD::make(backend, make_slots({1.0}), options));

    // The buffer is the gradient, 1, after the first step and 0.9 * 1 + 1 after the second.
    static_cast<void>(step_once(*optimizer));
    const std::vector<double> second = step_once(*optimizer);

    EXPECT_NEAR(second[0], K_START - 0.1 - 0.1 * 1.9, K_UPDATE_TOLERANCE);
}
