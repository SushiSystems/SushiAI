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
 * @file test_adamw.cpp
 * @brief Tests what AdamW::make() refuses, the state it starts from and its first update.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using Optim::AdamW;
using Optim::AdamWOptions;

// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the value every parameter element starts at. */
    constexpr double K_START = 1.0;

    /** @brief Holds the step size of the update case. */
    constexpr double K_RATE = 0.1;

    /** @brief Bounds the absolute error of a float32 value of order one after one update. */
    constexpr double K_UPDATE_TOLERANCE = 1e-6;

    /** @brief Holds the gradient of the one parameter; signs and sizes differ per element. */
    const std::vector<double> K_GRADIENT{2.0, -3.0, 0.5};

    /** @brief Provides an engine and one parameter slot of K_GRADIENT. */
    class Medium_AdamW : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns one slot "weight" of K_START with the gradient K_GRADIENT. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                const int64_t elements = static_cast<int64_t>(K_GRADIENT.size());
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(*engine, Shape{elements}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(*engine, Shape{elements}, DType::Float32);
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
    };
} // namespace

/** @brief Tests that AdamW names itself "adamw" and replays only with the fused kernel. */
TEST_F(Medium_AdamW, TheTypeAndReplayFollowTheKernel)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> fused,
                            AdamW::make(*engine, make_slots()));
    AdamWOptions separate_options;
    separate_options.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> separate,
                            AdamW::make(*engine, make_slots(), separate_options));

    EXPECT_EQ(fused->type_name(), Optim::K_ADAMW_TYPE);
    EXPECT_TRUE(fused->is_replay_invariant());
    EXPECT_FALSE(separate->is_replay_invariant());
}

/** @brief Tests that a decay outside [0, 1) and a floor that is not positive are refused. */
TEST_F(Medium_AdamW, MakeRefusesOptionsOutsideTheirRange)
{
    AdamWOptions beta1;
    beta1.beta1 = 1.0;
    AdamWOptions beta2;
    beta2.beta2 = -0.1;
    AdamWOptions eps;
    eps.eps = 0.0;

    SA_EXPECT_FAILURE(AdamW::make(*engine, make_slots(), beta1), StatusCode::InvalidDescription,
                      "beta1", "[0, 1)");
    SA_EXPECT_FAILURE(AdamW::make(*engine, make_slots(), beta2), StatusCode::InvalidDescription,
                      "beta2", "[0, 1)");
    SA_EXPECT_FAILURE(AdamW::make(*engine, make_slots(), eps), StatusCode::InvalidDescription,
                      "eps must be positive");
}

/** @brief Tests that both moments of a new optimizer hold zeros. */
TEST_F(Medium_AdamW, TheMomentsStartAtZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(*engine, make_slots()));

    for (const double value : SaTest::read_tensor(optimizer->first_moment(0)))
        EXPECT_EQ(value, 0.0);
    for (const double value : SaTest::read_tensor(optimizer->second_moment(0)))
        EXPECT_EQ(value, 0.0);
}

/** @brief Tests that the bias-corrected first step moves each element by the rate. */
TEST_F(Medium_AdamW, TheFirstStepMovesEachElementByTheRate)
{
    AdamWOptions options;
    options.learning_rate = K_RATE;
    options.weight_decay = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(*engine, make_slots(), options));
    const SushiBLAS::Tensor parameter = optimizer->parameters()[0].parameter;

    SA_EXPECT_OK(optimizer->step());
    flush();

    const std::vector<double> values = SaTest::read_tensor(parameter);
    const std::vector<double> first = SaTest::read_tensor(optimizer->first_moment(0));
    ASSERT_EQ(values.size(), K_GRADIENT.size());
    for (std::size_t i = 0; i < K_GRADIENT.size(); ++i)
    {
        const double sign = K_GRADIENT[i] > 0.0 ? 1.0 : -1.0;
        EXPECT_NEAR(values[i], K_START - K_RATE * sign, K_UPDATE_TOLERANCE) << "element " << i;
        EXPECT_NEAR(first[i], (1.0 - options.beta1) * K_GRADIENT[i], K_UPDATE_TOLERANCE)
            << "element " << i;
    }
}

/** @brief Tests that the moment of a parameter the optimizer lacks stops the process. */
TEST_F(Medium_AdamW, AMomentPastTheLastParameterIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AdamW> optimizer,
                            AdamW::make(*engine, make_slots()));

    SA_EXPECT_FATAL((void)optimizer->first_moment(1), "updates 1 parameters", "no moment 1");
}
