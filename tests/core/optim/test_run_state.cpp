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
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/loss_scaler.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/run_state.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;
using Optim::RunState;
using Optim::StateScalar;

namespace
{
    /** @brief Holds the elements of each parameter the optimizers update. */
    constexpr std::int64_t K_ELEMENTS = 3;

    /** @brief Views @p scalars as the scalar functions take them. */
    SushiRuntime::span<const StateScalar> view(const std::vector<StateScalar>& scalars)
    {
        return SushiRuntime::span<const StateScalar>(scalars.data(), scalars.size());
    }

    /** @brief Returns the one scalar of an optimizer without a loss scaler, at @p steps. */
    std::vector<StateScalar> step_count_of(std::uint64_t steps)
    {
        return {StateScalar{std::string(Optim::K_STEP_COUNT_STATE), steps}};
    }

    /** @brief Provides a host backend and two parameters, "weight" and "bias", with gradients. */
    class Small_OptimizerRunState : public ::testing::Test
    {
        protected:
            /** @brief Returns one slot called @p name: zeros with a gradient of ones. */
            Optim::ParameterSlot make_slot(const char* name)
            {
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(backend.engine(), Shape{K_ELEMENTS}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(backend.engine(), Shape{K_ELEMENTS}, DType::Float32);
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
                    Optim::SGD::make(backend, make_slots(), options);
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }

            /** @brief Holds the host adapter the optimizers are built on. */
            HostExecutionBackend backend;
    };
} // namespace

/** @brief Tests that SGD with momentum lists one buffer per slot, named, of the slot's size. */
TEST_F(Small_OptimizerRunState, SGDListsOneMomentumBufferPerSlot)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);

    SA_ASSERT_OK_AND_ASSIGN(const RunState state, optimizer->run_state());

    EXPECT_EQ(optimizer->type_name(), "sgd");
    ASSERT_EQ(state.tensors.size(), 2u);
    EXPECT_EQ(state.tensors[0].name, "weight.momentum");
    EXPECT_EQ(state.tensors[1].name, "bias.momentum");
    EXPECT_EQ(state.tensors[0].tensor.num_elements, K_ELEMENTS);
}

/** @brief Tests that SGD lists the number of updates it has applied as its step count. */
TEST_F(Small_OptimizerRunState, SGDListsItsStepCount)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);
    SA_EXPECT_OK(optimizer->step());

    SA_ASSERT_OK_AND_ASSIGN(const RunState state, optimizer->run_state());

    ASSERT_EQ(state.scalars.size(), 1u);
    EXPECT_EQ(state.scalars[0].name, "step_count");
    ASSERT_TRUE(std::holds_alternative<std::uint64_t>(state.scalars[0].value));
    EXPECT_EQ(std::get<std::uint64_t>(state.scalars[0].value), 1u);
}

/** @brief Tests that an SGD of no momentum names no state tensor. */
TEST(Small_SGDStateNames, NoMomentumNamesNoTensor)
{
    Optim::SGDOptions options;
    options.momentum = 0.0;

    EXPECT_TRUE(Optim::sgd_state_names({"weight", "bias"}, options).empty());
}

/** @brief Tests that AdamW lists both moments of each slot, the first moment first. */
TEST_F(Small_OptimizerRunState, AdamWListsBothMomentsOfEachSlot)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, make_slots()));

    SA_ASSERT_OK_AND_ASSIGN(const RunState state, optimizer->run_state());

    EXPECT_EQ(optimizer->type_name(), "adamw");
    ASSERT_EQ(state.tensors.size(), 4u);
    EXPECT_EQ(state.tensors[0].name, "weight.first_moment");
    EXPECT_EQ(state.tensors[1].name, "weight.second_moment");
    EXPECT_EQ(state.tensors[2].name, "bias.first_moment");
    EXPECT_EQ(state.tensors[3].name, "bias.second_moment");
}

/** @brief Tests that a listed tensor shares the optimizer's storage, so a write is a restore. */
TEST_F(Small_OptimizerRunState, AListedTensorSharesTheOptimizersStorage)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, make_slots()));

    SA_ASSERT_OK_AND_ASSIGN(const RunState state, optimizer->run_state());

    EXPECT_EQ(state.tensors[0].tensor.data_as<float>(),
              optimizer->first_moment(0).data_as<float>());
    EXPECT_EQ(state.tensors[3].tensor.data_as<float>(),
              optimizer->second_moment(1).data_as<float>());
}

/** @brief Tests that restoring the scalars sets the step count. */
TEST_F(Small_OptimizerRunState, TheStepCountIsRestored)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);

    SA_EXPECT_OK(optimizer->restore_scalars(view(step_count_of(41))));

    EXPECT_EQ(optimizer->step_count(), 41u);
}

/** @brief Tests that the next update counts on from the restored step count. */
TEST_F(Small_OptimizerRunState, TheNextUpdateCountsOnFromTheRestoredCount)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);
    SA_EXPECT_OK(optimizer->restore_scalars(view(step_count_of(41))));

    SA_EXPECT_OK(optimizer->step());

    EXPECT_EQ(optimizer->step_count(), 42u);
}

/** @brief Tests that run state of the optimizer's own type is accepted. */
TEST(Small_StateOwner, TheOwnTypeIsAccepted)
{
    SA_EXPECT_OK(Optim::check_state_owner("sgd", "sgd"));
}

/** @brief Tests that run state of another optimizer type is refused with both names. */
TEST(Small_StateOwner, AnotherTypeIsRefusedWithBothNames)
{
    SA_EXPECT_FAILURE(Optim::check_state_owner("adamw", "sgd"), StatusCode::Incompatible,
                      "'adamw'", "'sgd'");
}

namespace
{
/** @brief Holds a scalar list the check refuses and the words its refusal uses. */
struct ScalarListRow
{
    std::vector<StateScalar> given; /**< The scalars of the saved run state. */
    const char* first;              /**< A fragment of the refusal. */
    const char* second;             /**< Another fragment of the refusal. */
};
} // namespace

namespace
{
/** @brief Runs the scalar-list refusals as rows. */
class Small_ScalarList : public ::testing::TestWithParam<ScalarListRow>
{

};
} // namespace

/** @brief Tests that a missing, a surplus, a repeated and a mistyped scalar are refused. */
TEST_P(Small_ScalarList, AScalarThatDoesNotFitIsRefusedByName)
{
    const std::vector<StateScalar> kept = step_count_of(0);

    SA_EXPECT_FAILURE(Optim::check_scalar_list(view(kept), view(GetParam().given), "sgd"),
                      StatusCode::Incompatible, GetParam().first, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ScalarList,
    ::testing::Values(
        ScalarListRow{{}, "does not hold", "'step_count'"},
        ScalarListRow{{StateScalar{"step_count", std::uint64_t{3}},
                       StateScalar{"loss_scale", 1024.0}},
                      "'loss_scale'", "does not keep"},
        ScalarListRow{{StateScalar{"step_count", std::uint64_t{3}},
                       StateScalar{"step_count", std::uint64_t{3}}},
                      "'step_count'", "more than once"},
        ScalarListRow{{StateScalar{"step_count", 3.0}}, "'step_count'",
                      "a real number where a count is kept"}));

/** @brief Tests that a scalar list that fits is accepted. */
TEST(Small_ScalarListFit, TheListTheOptimizerKeepsIsAccepted)
{
    const std::vector<StateScalar> kept = step_count_of(0);
    const std::vector<StateScalar> given = step_count_of(3);

    SA_EXPECT_OK(Optim::check_scalar_list(view(kept), view(given), "sgd"));
}

/** @brief Tests that a refused scalar list leaves the step count where it was. */
TEST_F(Small_OptimizerRunState, ARefusedScalarListLeavesTheStepCount)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(0.9);
    const std::vector<StateScalar> mistyped = {
        StateScalar{std::string(Optim::K_STEP_COUNT_STATE), 3.0}};

    SA_EXPECT_FAILURE(optimizer->restore_scalars(view(mistyped)), StatusCode::Incompatible,
                      "'step_count'");

    EXPECT_EQ(optimizer->step_count(), 0u);
}

/** @brief Tests that a loss scale state takes a saved scale and its clean-step count. */
TEST(Small_LossScaleStateRestore, TakesASavedScale)
{
    SA_ASSERT_OK_AND_ASSIGN(Optim::LossScaleState state, Optim::LossScaleState::make());

    SA_EXPECT_OK(state.restore(4096.0, 1999));

    EXPECT_EQ(state.scale(), 4096.0);
    EXPECT_EQ(state.consecutive_good_steps(), 1999u);
}

/** @brief Tests that one more clean step completes the growth interval of 2000. */
TEST(Small_LossScaleStateRestore, TheRestoredCountContinuesTheGrowthInterval)
{
    SA_ASSERT_OK_AND_ASSIGN(Optim::LossScaleState state, Optim::LossScaleState::make());
    SA_EXPECT_OK(state.restore(4096.0, 1999));

    state.observe(false);

    EXPECT_EQ(state.scale(), 8192.0);
    EXPECT_EQ(state.consecutive_good_steps(), 0u);
}

namespace
{
/** @brief Holds a saved loss scale the restore refuses and the words its refusal uses. */
struct LossScaleRow
{
    double scale;             /**< The saved scale. */
    std::uint64_t good_steps; /**< The saved clean-step count. */
    const char* first;        /**< A fragment of the refusal. */
    const char* second;       /**< Another fragment of the refusal. */
};
} // namespace

namespace
{
/** @brief Runs the loss scale refusals as rows. */
class Small_LossScaleStateRefusal : public ::testing::TestWithParam<LossScaleRow>
{

};
} // namespace

/** @brief Tests that a restore refuses a scale or a count the state cannot hold, naming it. */
TEST_P(Small_LossScaleStateRefusal, RestoreRefusesWhatTheStateCannotHold)
{
    SA_ASSERT_OK_AND_ASSIGN(Optim::LossScaleState state, Optim::LossScaleState::make());

    SA_EXPECT_FAILURE(state.restore(GetParam().scale, GetParam().good_steps),
                      StatusCode::Incompatible, GetParam().first, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LossScaleStateRefusal,
                         ::testing::Values(LossScaleRow{3000.0, 0, "3000", "power of two"},
                                           LossScaleRow{0.5, 0, "0.5", "bounds"},
                                           LossScaleRow{4096.0, 2000, "2000", "growth interval"}));

/** @brief Tests that a refused restore leaves the scale where it was. */
TEST(Small_LossScaleStateRestore, ARefusedRestoreLeavesTheScale)
{
    SA_ASSERT_OK_AND_ASSIGN(Optim::LossScaleState state, Optim::LossScaleState::make());
    SA_EXPECT_OK(state.restore(8192.0, 0));

    SA_EXPECT_FAILURE(state.restore(3000.0, 0), StatusCode::Incompatible, "3000");

    EXPECT_EQ(state.scale(), 8192.0);
}
