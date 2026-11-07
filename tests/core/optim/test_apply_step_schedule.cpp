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
 * @file test_apply_step_schedule.cpp
 * @brief Tests that a schedule's rate reaches an optimizer before each update.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;
using Optim::apply_step_schedule;
using Optim::StepSchedule;

namespace
{
    /**
     * @brief Bounds the absolute error of a double result; 16384 unit roundoffs of double, the
     * smallest power of two that covers 1e-12.
     */
    constexpr double K_DOUBLE_TOLERANCE = 16384.0 * TestSupport::unit_roundoff<double>();

    /** @brief Holds the elements of the one parameter the optimizers update. */
    constexpr std::int64_t K_ELEMENTS = 3;

    /** @brief Holds the gradient of every element. */
    constexpr double K_GRADIENT = 2.0;

    /** @brief Holds the rate the optimizers are made with, which no schedule below gives. */
    constexpr double K_MADE_RATE = 0.75;

    /** @brief Returns a rate of 0.5 that halves from update 2 and again from update 3. */
    StepSchedule make_halving_schedule()
    {
        StepSchedule schedule;
        schedule.base_rate = 0.5;
        schedule.steps = {2, 3};
        schedule.scales = {0.5, 0.5};
        return schedule;
    }

    /** @brief Returns a cosine schedule of rate 0.5 with a warm-up of 2 and a decay to 4. */
    Optim::CosineSchedule make_cosine_schedule()
    {
        Optim::CosineSchedule cosine;
        cosine.base_rate = 0.5;
        cosine.warmup_iterations = 2;
        cosine.total_iterations = 4;
        return cosine;
    }

    /** @brief Holds the rates a run of updates applied and the optimizer then held. */
    struct Applied
    {
        std::vector<double> returned; /**< What apply_step_schedule returned before each update. */
        std::vector<double> held;     /**< The optimizer's learning rate after each apply. */
    };

    /** @brief Applies @p schedule before each of @p updates updates of @p optimizer, and steps. */
    Applied apply_over(const StepSchedule& schedule, Optim::Optimizer& optimizer,
                       std::size_t updates)
    {
        Applied applied;
        for (std::size_t update = 0; update < updates; ++update)
        {
            applied.returned.push_back(apply_step_schedule(schedule, optimizer));
            applied.held.push_back(optimizer.learning_rate());
            SA_EXPECT_OK(optimizer.step());
        }

        return applied;
    }

    /** @brief Steps @p optimizer @p updates times without touching its rate. */
    void step_times(Optim::Optimizer& optimizer, std::size_t updates)
    {
        for (std::size_t update = 0; update < updates; ++update)
            SA_EXPECT_OK(optimizer.step());
    }

    /** @brief Provides a host backend and one parameter of zeros with a constant gradient. */
    class Small_ApplyStepSchedule : public ::testing::Test
    {
        protected:
            /** @brief Returns the one slot: a parameter of zeros and a gradient of K_GRADIENT. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(backend.engine(), Shape{K_ELEMENTS}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(backend.engine(), Shape{K_ELEMENTS}, DType::Float32);
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

            /** @brief Holds the host adapter the optimizers are built on. */
            HostExecutionBackend backend;
    };

    /** @brief Holds the updates to run and the parameter value they leave in every element. */
    struct UpdateRow
    {
        std::size_t updates; /**< How many updates run, each after applying the schedule. */
        double value;        /**< The value of every element: minus the rates times K_GRADIENT. */
    };

    /** @brief Runs the update cases once per row. */
    class Small_ApplyStepScheduleUpdates : public Small_ApplyStepSchedule,
                                           public ::testing::WithParamInterface<UpdateRow>
    {

    };

    /** @brief Runs the learning-rate check once per rate. */
    class Small_LearningRateCheck : public ::testing::TestWithParam<double>
    {

    };
} // namespace

/** @brief Tests that the rate set is the schedule's at the number of the next update. */
TEST_F(Small_ApplyStepSchedule, TheRateSetIsTheScheduleAtTheNextUpdate)
{
    Optim::SGDOptions options;
    options.learning_rate = K_MADE_RATE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), options));

    const Applied applied = apply_over(make_halving_schedule(), *optimizer, 4);

    // Every rate is a power of two, so the products are exact in double.
    const std::vector<double> expected{0.5, 0.25, 0.125, 0.125};
    EXPECT_EQ(applied.returned, expected);
    EXPECT_EQ(applied.held, expected);
    EXPECT_EQ(optimizer->step_count(), 4u);
}

/** @brief Tests that an update moves the parameter by the rate that was set, not the made one. */
TEST_P(Small_ApplyStepScheduleUpdates, AnUpdateUsesTheRateThatWasSet)
{
    Optim::SGDOptions options;
    options.learning_rate = K_MADE_RATE;
    std::vector<Optim::ParameterSlot> slots = make_slots();
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, std::move(slots), options));

    (void)apply_over(make_halving_schedule(), *optimizer, GetParam().updates);

    EXPECT_EQ(SaTest::read_tensor(parameter),
              std::vector<double>(static_cast<std::size_t>(K_ELEMENTS), GetParam().value));
}

// Rates 0.5, 0.25 and 0.125 times the gradient 2 add up to 1, 1.5 and 1.75, all exact in float32.
INSTANTIATE_TEST_SUITE_P(Rows, Small_ApplyStepScheduleUpdates,
                         ::testing::Values(UpdateRow{1, -1.0}, UpdateRow{2, -1.5},
                                           UpdateRow{3, -1.75}));

/** @brief Tests that the rate reaches AdamW through the same seam as SGD. */
TEST_F(Small_ApplyStepSchedule, AdamWTakesTheRateThroughTheSameSeam)
{
    Optim::AdamWOptions options;
    options.learning_rate = K_MADE_RATE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> made,
                            Optim::AdamW::make(backend, make_slots(), options));
    Optim::Optimizer& optimizer = *made;

    EXPECT_EQ(apply_step_schedule(make_halving_schedule(), optimizer), 0.5);
    EXPECT_EQ(made->options().learning_rate, 0.5);
}

/** @brief Tests that a rate no update can use is refused, naming the rate. */
TEST_P(Small_LearningRateCheck, ARateNoUpdateCanUseIsRefused)
{
    SA_EXPECT_FAILURE(Optim::check_learning_rate(GetParam()), StatusCode::InvalidDescription,
                      "learning rate", "negative or not finite");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LearningRateCheck,
                         ::testing::Values(-0.1, -std::numeric_limits<double>::infinity(),
                                           std::numeric_limits<double>::infinity(),
                                           std::numeric_limits<double>::quiet_NaN()));

/** @brief Tests that a rate of zero or more is accepted. */
TEST(Small_LearningRateAccepted, AZeroOrPositiveRateIsAccepted)
{
    SA_EXPECT_OK(Optim::check_learning_rate(0.0));
    SA_EXPECT_OK(Optim::check_learning_rate(0.5));
}

/** @brief Tests that a cosine schedule sets the linear warm-up rate before the first update. */
TEST_F(Small_ApplyStepSchedule, ACosineScheduleSetsTheWarmupRate)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), Optim::SGDOptions{}));

    // Update 1 is half the warm-up of 2 updates.
    EXPECT_EQ(apply_step_schedule(make_cosine_schedule(), *optimizer), 0.25);
}

/** @brief Tests that a cosine schedule held in a variant sets the base rate once warmed up. */
TEST_F(Small_ApplyStepSchedule, AHeldCosineScheduleSetsTheBaseRateAfterTheWarmup)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), Optim::SGDOptions{}));
    step_times(*optimizer, 1);
    const Optim::RateSchedule held = make_cosine_schedule();

    EXPECT_EQ(apply_step_schedule(held, *optimizer), 0.5);
    EXPECT_EQ(optimizer->learning_rate(), 0.5);
}

/** @brief Tests that a step schedule held in a variant sets the rate of its third update. */
TEST_F(Small_ApplyStepSchedule, AHeldStepScheduleSetsTheRateOfTheThirdUpdate)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), Optim::SGDOptions{}));
    step_times(*optimizer, 2);
    const Optim::RateSchedule stepped = make_halving_schedule();

    EXPECT_EQ(apply_step_schedule(stepped, *optimizer), 0.125);
}

/** @brief Tests that a cosine schedule sets half its decay at the midpoint of the decay. */
TEST_F(Small_ApplyStepSchedule, ACosineScheduleSetsHalfTheBaseRateAtTheMidpointOfTheDecay)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), Optim::SGDOptions{}));
    step_times(*optimizer, 2);

    // The weight of the decay at its midpoint is (1 + cos(pi / 2)) / 2, which rounds off 0.5.
    EXPECT_NEAR(apply_step_schedule(make_cosine_schedule(), *optimizer), 0.25, K_DOUBLE_TOLERANCE);
}
