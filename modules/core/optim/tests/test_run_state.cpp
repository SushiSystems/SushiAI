/**************************************************************************/
/* test_run_state.cpp                                                     */
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
 * @file test_run_state.cpp
 * @brief Tests the run state an optimizer lists and restores, and each refusal of a restore.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/loss_scaler.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/run_state.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Optim::RunState;
using Optim::StateScalar;

namespace
{
    /** @brief Holds the elements of each parameter the optimizers update. */
    constexpr std::int64_t K_ELEMENTS = 3;

    /** @brief Bounds the error of a scale, which is a power of two and is kept exactly. */
    constexpr double K_SCALE_TOLERANCE = 0.0;

    /** @brief Views @p scalars as restore_scalars() takes them. */
    SushiRuntime::span<const StateScalar> view(const std::vector<StateScalar>& scalars)
    {
        return SushiRuntime::span<const StateScalar>(scalars.data(), scalars.size());
    }

    /** @brief Returns the one scalar of an optimizer without a loss scaler, at @p steps. */
    std::vector<StateScalar> step_count_of(std::uint64_t steps)
    {
        return {StateScalar{std::string(Optim::K_STEP_COUNT_STATE), steps}};
    }

    /** @brief Provides an engine and two parameters, "weight" and "bias", with gradients. */
    class Unit_OptimizerRunState : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns one slot called @p name: zeros with a gradient of ones. */
            Optim::ParameterSlot make_slot(const char* name)
            {
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(*engine, Shape{K_ELEMENTS}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(*engine, Shape{K_ELEMENTS}, DType::Float32);
                EXPECT_TRUE(parameter.status().is_ok());
                EXPECT_TRUE(gradient.status().is_ok());

                const std::vector<double> zeros(static_cast<std::size_t>(K_ELEMENTS), 0.0);
                const std::vector<double> ones(static_cast<std::size_t>(K_ELEMENTS), 1.0);
                SaTest::write_tensor(parameter.value(), zeros);
                SaTest::write_tensor(gradient.value(), ones);

                return Optim::ParameterSlot{name, std::move(parameter).value(),
                                            std::move(gradient).value()};
            }

            /** @brief Returns the slots "weight" and "bias", in that order. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                std::vector<Optim::ParameterSlot> slots;
                slots.push_back(make_slot("weight"));
                slots.push_back(make_slot("bias"));
                return slots;
            }

            /** @brief Returns an SGD over make_slots() with @p momentum. */
            std::unique_ptr<Optim::SGD> make_sgd(double momentum)
            {
                Optim::SGDOptions options;
                options.momentum = momentum;
                Result<std::unique_ptr<Optim::SGD>> made =
                    Optim::SGD::make(*engine, make_slots(), options);
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }
    };
} // namespace

/** @brief Tests that SGD with momentum lists one buffer per slot and the step count. */
TEST_F(Unit_OptimizerRunState, SGDListsItsMomentumBuffersAndItsStepCount)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);
    SA_EXPECT_OK(optimizer->step());

    SA_ASSERT_OK_AND_ASSIGN(const RunState state, optimizer->run_state());

    EXPECT_EQ(optimizer->type_name(), "sgd");
    ASSERT_EQ(state.tensors.size(), 2u);
    EXPECT_EQ(state.tensors[0].name, "weight.momentum");
    EXPECT_EQ(state.tensors[1].name, "bias.momentum");
    EXPECT_EQ(state.tensors[0].tensor.num_elements, K_ELEMENTS);

    ASSERT_EQ(state.scalars.size(), 1u);
    EXPECT_EQ(state.scalars[0].name, "step_count");
    ASSERT_TRUE(std::holds_alternative<std::uint64_t>(state.scalars[0].value));
    EXPECT_EQ(std::get<std::uint64_t>(state.scalars[0].value), 1u);
}

/** @brief Tests that SGD without momentum lists no tensor. */
TEST_F(Unit_OptimizerRunState, SGDWithoutMomentumListsNoTensor)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.0);

    SA_ASSERT_OK_AND_ASSIGN(const RunState state, optimizer->run_state());

    EXPECT_TRUE(state.tensors.empty());
    ASSERT_EQ(state.scalars.size(), 1u);
    EXPECT_EQ(std::get<std::uint64_t>(state.scalars[0].value), 0u);
}

/** @brief Tests that AdamW lists both moments of each slot, the first moment first. */
TEST_F(Unit_OptimizerRunState, AdamWListsBothMomentsOfEachSlot)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, make_slots()));

    SA_ASSERT_OK_AND_ASSIGN(const RunState state, optimizer->run_state());

    EXPECT_EQ(optimizer->type_name(), "adamw");
    ASSERT_EQ(state.tensors.size(), 4u);
    EXPECT_EQ(state.tensors[0].name, "weight.first_moment");
    EXPECT_EQ(state.tensors[1].name, "weight.second_moment");
    EXPECT_EQ(state.tensors[2].name, "bias.first_moment");
    EXPECT_EQ(state.tensors[3].name, "bias.second_moment");

    // A listed tensor shares the optimizer's storage, so a write through it is a restore.
    EXPECT_EQ(state.tensors[0].tensor.data_as<float>(),
              optimizer->first_moment(0).data_as<float>());
    EXPECT_EQ(state.tensors[3].tensor.data_as<float>(),
              optimizer->second_moment(1).data_as<float>());
}

/** @brief Tests that the step count is restored and the next update counts on from it. */
TEST_F(Unit_OptimizerRunState, TheStepCountIsRestored)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);

    SA_EXPECT_OK(optimizer->restore_scalars(view(step_count_of(41))));
    EXPECT_EQ(optimizer->step_count(), 41u);

    SA_EXPECT_OK(optimizer->step());
    EXPECT_EQ(optimizer->step_count(), 42u);
}

/** @brief Tests that run state of another optimizer type is refused with both names. */
TEST_F(Unit_OptimizerRunState, AnotherOwnerIsRefusedWithBothTypes)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);

    SA_EXPECT_OK(optimizer->check_state_owner("sgd"));
    SA_EXPECT_FAILURE(optimizer->check_state_owner("adamw"), StatusCode::Incompatible,
                      "'adamw'", "'sgd'");
}

/** @brief Tests that a missing, a surplus, a repeated and a mistyped scalar are refused. */
TEST_F(Unit_OptimizerRunState, AScalarThatDoesNotFitIsRefusedByName)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);

    SA_EXPECT_FAILURE(optimizer->restore_scalars(view({})), StatusCode::Incompatible,
                      "does not hold", "'step_count'", "'sgd'");

    std::vector<StateScalar> surplus = step_count_of(3);
    surplus.push_back(StateScalar{std::string(Optim::K_LOSS_SCALE_STATE), 1024.0});
    SA_EXPECT_FAILURE(optimizer->restore_scalars(view(surplus)), StatusCode::Incompatible,
                      "'loss_scale'", "'sgd'", "does not keep");

    std::vector<StateScalar> repeated = step_count_of(3);
    repeated.push_back(repeated.front());
    SA_EXPECT_FAILURE(optimizer->restore_scalars(view(repeated)), StatusCode::Incompatible,
                      "'step_count'", "more than once");

    const std::vector<StateScalar> mistyped = {
        StateScalar{std::string(Optim::K_STEP_COUNT_STATE), 3.0}};
    SA_EXPECT_FAILURE(optimizer->restore_scalars(view(mistyped)), StatusCode::Incompatible,
                      "'step_count'", "a real number where a count is kept");

    EXPECT_EQ(optimizer->step_count(), 0u);
}

/** @brief Tests that a loss scale state takes a saved scale and refuses one it cannot hold. */
TEST(Unit_LossScaleStateRestore, TakesASavedScaleAndRefusesOneOutOfBounds)
{
    SA_ASSERT_OK_AND_ASSIGN(Optim::LossScaleState state, Optim::LossScaleState::make());

    SA_EXPECT_OK(state.restore(4096.0, 1999));
    EXPECT_NEAR(state.scale(), 4096.0, K_SCALE_TOLERANCE);
    EXPECT_EQ(state.consecutive_good_steps(), 1999u);

    // One more clean step completes the growth interval of 2000.
    state.observe(false);
    EXPECT_NEAR(state.scale(), 8192.0, K_SCALE_TOLERANCE);
    EXPECT_EQ(state.consecutive_good_steps(), 0u);

    SA_EXPECT_FAILURE(state.restore(3000.0, 0), StatusCode::Incompatible, "3000",
                      "power of two");
    SA_EXPECT_FAILURE(state.restore(0.5, 0), StatusCode::Incompatible, "0.5", "bounds");
    SA_EXPECT_FAILURE(state.restore(4096.0, 2000), StatusCode::Incompatible, "2000",
                      "growth interval");
    EXPECT_NEAR(state.scale(), 8192.0, K_SCALE_TOLERANCE);
}
