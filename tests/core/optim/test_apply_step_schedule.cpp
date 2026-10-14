/**************************************************************************/
/* test_apply_step_schedule.cpp                                           */
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
 * @file test_apply_step_schedule.cpp
 * @brief Tests that a schedule's rate reaches an optimizer before each update.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;
using Optim::apply_step_schedule;
using Optim::StepSchedule;

namespace
{
    /** @brief Holds the elements of the one parameter the optimizers update. */
    constexpr std::int64_t K_ELEMENTS = 3;

    /** @brief Holds the gradient of every element. */
    constexpr double K_GRADIENT = 2.0;

    /** @brief Holds the rate the optimizers are made with, which no schedule below gives. */
    constexpr double K_MADE_RATE = 0.75;

    /** @brief Bounds the error of a rate that is one product of two doubles. */
    constexpr double K_RATE_TOLERANCE = 1e-15;

    /** @brief Bounds the absolute error of a float32 parameter of order one after an update. */
    constexpr double K_UPDATE_TOLERANCE = 1e-6;

    /** @brief Returns a rate of 0.5 that halves from update 2 and again from update 3. */
    StepSchedule make_halving_schedule()
    {
        StepSchedule schedule;
        schedule.base_rate = 0.5;
        schedule.steps = {2, 3};
        schedule.scales = {0.5, 0.5};
        return schedule;
    }

    /** @brief Provides an engine and one parameter of zeros with a constant gradient. */
    class Unit_ApplyStepSchedule : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns the one slot: a parameter of zeros and a gradient of K_GRADIENT. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(*engine, Shape{K_ELEMENTS}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(*engine, Shape{K_ELEMENTS}, DType::Float32);
                EXPECT_TRUE(parameter.status().is_ok());
                EXPECT_TRUE(gradient.status().is_ok());

                const std::vector<double> zeros(static_cast<std::size_t>(K_ELEMENTS), 0.0);
                const std::vector<double> slopes(static_cast<std::size_t>(K_ELEMENTS),
                                                 K_GRADIENT);
                SaTest::write_tensor(parameter.value(), zeros);
                SaTest::write_tensor(gradient.value(), slopes);

                std::vector<Optim::ParameterSlot> slots;
                slots.push_back(Optim::ParameterSlot{"weight", std::move(parameter).value(),
                                                     std::move(gradient).value()});
                return slots;
            }
    };
} // namespace

/** @brief Tests that the rate set is the schedule's at the number of the next update. */
TEST_F(Unit_ApplyStepSchedule, TheRateSetIsTheScheduleAtTheNextUpdate)
{
    Optim::SGDOptions options;
    options.learning_rate = K_MADE_RATE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, make_slots(), options));
    const StepSchedule schedule = make_halving_schedule();

    const double expected[] = {0.5, 0.25, 0.125, 0.125};
    for (const double rate : expected)
    {
        EXPECT_NEAR(apply_step_schedule(schedule, *optimizer), rate, K_RATE_TOLERANCE);
        EXPECT_NEAR(optimizer->learning_rate(), rate, K_RATE_TOLERANCE);
        SA_EXPECT_OK(optimizer->step());
    }

    EXPECT_EQ(optimizer->step_count(), 4u);
}

/** @brief Tests that an update moves the parameter by the rate that was set, not the made one. */
TEST_F(Unit_ApplyStepSchedule, AnUpdateUsesTheRateThatWasSet)
{
    Optim::SGDOptions options;
    options.learning_rate = K_MADE_RATE;
    std::vector<Optim::ParameterSlot> slots = make_slots();
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, std::move(slots), options));
    const StepSchedule schedule = make_halving_schedule();

    double expected = 0.0;
    for (int update = 0; update < 3; ++update)
    {
        expected -= apply_step_schedule(schedule, *optimizer) * K_GRADIENT;
        SA_EXPECT_OK(optimizer->step());
        flush();

        for (const double value : SaTest::read_tensor(parameter))
            EXPECT_NEAR(value, expected, K_UPDATE_TOLERANCE);
    }
}

/** @brief Tests that AdamW takes a rate through the same seam as SGD. */
TEST_F(Unit_ApplyStepSchedule, AdamWTakesTheRateThroughTheSameSeam)
{
    Optim::AdamWOptions options;
    options.learning_rate = K_MADE_RATE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> made,
                            Optim::AdamW::make(*engine, make_slots(), options));
    Optim::Optimizer& optimizer = *made;

    EXPECT_NEAR(optimizer.learning_rate(), K_MADE_RATE, K_RATE_TOLERANCE);
    EXPECT_NEAR(apply_step_schedule(make_halving_schedule(), optimizer), 0.5,
                K_RATE_TOLERANCE);
    EXPECT_NEAR(made->options().learning_rate, 0.5, K_RATE_TOLERANCE);
}

/** @brief Tests that a rate no update can use stops the process. */
TEST_F(Unit_ApplyStepSchedule, ANegativeRateIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, make_slots(), Optim::SGDOptions{}));

    SA_EXPECT_FATAL(optimizer->set_learning_rate(-0.1), "learning rate", "negative");
}

/** @brief Tests that a cosine schedule reaches an optimizer by itself and held in a variant. */
TEST_F(Unit_ApplyStepSchedule, ACosineScheduleTakesTheSameSeam)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, make_slots(), Optim::SGDOptions{}));

    Optim::CosineSchedule cosine;
    cosine.base_rate = 0.5;
    cosine.warmup_iterations = 2;
    cosine.total_iterations = 4;

    // Update 1 is half the warm-up; update 2 is the base rate; update 3 is half the decay.
    EXPECT_NEAR(apply_step_schedule(cosine, *optimizer), 0.25, K_RATE_TOLERANCE);
    SA_EXPECT_OK(optimizer->step());

    const Optim::RateSchedule held = cosine;
    EXPECT_NEAR(apply_step_schedule(held, *optimizer), 0.5, K_RATE_TOLERANCE);
    EXPECT_NEAR(optimizer->learning_rate(), 0.5, K_RATE_TOLERANCE);
    SA_EXPECT_OK(optimizer->step());

    const Optim::RateSchedule stepped = make_halving_schedule();
    EXPECT_NEAR(apply_step_schedule(stepped, *optimizer), 0.125, K_RATE_TOLERANCE);
    EXPECT_NEAR(apply_step_schedule(held, *optimizer), 0.25, 1e-12);
}
