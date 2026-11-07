/**************************************************************************/
/* test_adamw.cpp                                                         */
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
 * @file test_adamw.cpp
 * @brief Tests what the AdamW options refuse, the state AdamW starts from and its first update.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Optim::AdamW;
using Optim::AdamWOptions;
using Optim::check_adamw_options;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the value every parameter element starts at. */
    constexpr double K_START = 1.0;

    /** @brief Holds the step size of the update case. */
    constexpr double K_RATE = 0.1;

    /**
     * @brief Bounds the absolute error of a float32 value of order one after one update; 32 unit
     * roundoffs of float, the smallest power of two that covers 1e-6.
     */
    constexpr double K_UPDATE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /** @brief Holds the gradient of the one parameter; signs and sizes differ per element. */
    const std::vector<double> K_GRADIENT{2.0, -3.0, 0.5};

    /** @brief Provides the host backend and one parameter slot of K_GRADIENT. */
    class Small_AdamW : public ::testing::Test
    {
        protected:
            /** @brief Returns one slot "weight" of K_START with the gradient K_GRADIENT. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                const int64_t elements = static_cast<int64_t>(K_GRADIENT.size());
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(backend.engine(), Shape{elements}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(backend.engine(), Shape{elements}, DType::Float32);
                EXPECT_TRUE(parameter.status().is_ok());
                EXPECT_TRUE(gradient.status().is_ok());

                SaTest::write_tensor(parameter.value(),
                                     std::vector<double>(K_GRADIENT.size(), K_START));
                SaTest::write_tensor(gradient.value(), K_GRADIENT);

                std::vector<Optim::ParameterSlot> slots;
                slots.push_back(Optim::ParameterSlot{"weight", std::move(parameter).value(),
                                                     std::move(gradient).value()});
                return slots;
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
    class Small_AdamWKernel : public Small_AdamW,
                              public ::testing::WithParamInterface<KernelRow>
    {
    };

    /** @brief Names one option set and what its refusal must say. */
    struct OptionRow
    {
        AdamWOptions options;      /**< The options the check is handed. */
        std::string first_phrase;  /**< The key the refusal names. */
        std::string second_phrase; /**< The rest of the refusal. */
    };

    /** @brief Returns the default options with beta1 set to @p value. */
    AdamWOptions with_beta1(double value)
    {
        AdamWOptions options;
        options.beta1 = value;
        return options;
    }

    /** @brief Returns the default options with beta2 set to @p value. */
    AdamWOptions with_beta2(double value)
    {
        AdamWOptions options;
        options.beta2 = value;
        return options;
    }

    /** @brief Returns the default options with eps set to @p value. */
    AdamWOptions with_eps(double value)
    {
        AdamWOptions options;
        options.eps = value;
        return options;
    }

    /** @brief Provides one row of options the check refuses. */
    class Small_AdamWOptions : public ::testing::TestWithParam<OptionRow>
    {
    };
} // namespace

/** @brief Tests that the update replays only with the fused kernel. */
TEST_P(Small_AdamWKernel, TheReplayFollowsTheKernel)
{
    AdamWOptions options;
    options.kernel = GetParam().kernel;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(backend, make_slots(), options));

    EXPECT_EQ(optimizer->is_replay_invariant(), GetParam().replays);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_AdamWKernel,
                         ::testing::Values(KernelRow{Optim::UpdateKernel::Fused, true},
                                           KernelRow{Optim::UpdateKernel::Separate, false}));

/** @brief Tests that a decay outside [0, 1) and a floor that is not positive are refused. */
TEST_P(Small_AdamWOptions, TheCheckRefusesOptionsOutsideTheirRange)
{
    SA_EXPECT_FAILURE(check_adamw_options(GetParam().options), StatusCode::InvalidDescription,
                      GetParam().first_phrase, GetParam().second_phrase);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_AdamWOptions,
                         ::testing::Values(OptionRow{with_beta1(1.0), "beta1", "[0, 1)"},
                                           OptionRow{with_beta2(-0.1), "beta2", "[0, 1)"},
                                           OptionRow{with_eps(0.0), "eps", "must be positive"}));

/** @brief Tests that make() refuses the options the check refuses, before it allocates. */
TEST_F(Small_AdamW, MakeRunsTheOptionsCheck)
{
    SA_EXPECT_FAILURE(AdamW::make(backend, make_slots(), with_beta1(1.0)),
                      StatusCode::InvalidDescription, "beta1", "[0, 1)");
}

/** @brief Tests that both moments of a new optimizer hold zeros. */
TEST_F(Small_AdamW, TheMomentsStartAtZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(backend, make_slots()));
    const std::vector<double> zeros(K_GRADIENT.size(), 0.0);

    EXPECT_EQ(SaTest::read_tensor(optimizer->first_moment(0)), zeros);
    EXPECT_EQ(SaTest::read_tensor(optimizer->second_moment(0)), zeros);
}

/** @brief Tests that the bias-corrected first step moves each element by the rate. */
TEST_F(Small_AdamW, TheFirstStepMovesEachElementByTheRate)
{
    AdamWOptions options;
    options.learning_rate = K_RATE;
    options.weight_decay = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(backend, make_slots(), options));
    const SushiBLAS::Tensor parameter = optimizer->parameters()[0].parameter;

    SA_EXPECT_OK(optimizer->step());

    EXPECT_THAT(SaTest::read_tensor(parameter),
                ::testing::Pointwise(::testing::DoubleNear(K_UPDATE_TOLERANCE),
                                     std::vector<double>{K_START - K_RATE, K_START + K_RATE,
                                                         K_START - K_RATE}));
}

/** @brief Tests that the first step leaves the first moment at (1 - beta1) times the gradient. */
TEST_F(Small_AdamW, TheFirstStepFillsTheFirstMoment)
{
    AdamWOptions options;
    options.weight_decay = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(backend, make_slots(), options));

    SA_EXPECT_OK(optimizer->step());

    EXPECT_THAT(SaTest::read_tensor(optimizer->first_moment(0)),
                ::testing::Pointwise(::testing::DoubleNear(K_UPDATE_TOLERANCE),
                                     std::vector<double>{0.2, -0.3, 0.05}));
}

/** @brief Tests that the moment of a parameter the optimizer lacks stops the process. */
TEST_F(Small_AdamW, AMomentPastTheLastParameterIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(backend, make_slots()));

    SA_EXPECT_FATAL((void)optimizer->first_moment(1), "updates 1 parameters", "no moment 1");
}
