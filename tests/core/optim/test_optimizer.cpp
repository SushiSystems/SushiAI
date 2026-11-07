/**************************************************************************/
/* test_optimizer.cpp                                                     */
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
 * @file test_optimizer.cpp
 * @brief Tests that weight decay reaches the slots that take it and no other.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the elements of each parameter. */
    constexpr std::int64_t K_ELEMENTS = 3;

    /** @brief Holds the value every parameter starts at. */
    constexpr double K_START = 2.0;

    /** @brief Holds the rate of both optimizers. */
    constexpr double K_RATE = 0.5;

    /** @brief Holds the decay of both optimizers. */
    constexpr double K_DECAY = 0.5;

    /**
     * @brief Holds a parameter after one decayed update of a zero gradient, for both rules.
     *
     * Every factor is a power of two, so the product is exact in float32: 2 * (1 - 0.5 * 0.5).
     */
    constexpr double K_DECAYED = K_START * (1.0 - K_RATE * K_DECAY);

    /** @brief Holds the values of the two parameters after one update. */
    struct Updated
    {
        std::vector<double> weight; /**< The slot that decays unless the optimizer says all. */
        std::vector<double> bias;   /**< The slot that never takes the grouped decay. */
    };

    /** @brief Returns @p value once per element of a parameter. */
    std::vector<double> each(double value)
    {
        return std::vector<double>(static_cast<std::size_t>(K_ELEMENTS), value);
    }

    /** @brief Provides a host backend and two slots of zero gradient, the second without decay. */
    class Small_ParameterDecay : public ::testing::Test
    {
        protected:
            /** @brief Returns "weight", which decays, then "bias", which does not. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                std::vector<Optim::ParameterSlot> slots;
                slots.push_back(make_slot("weight", true));
                slots.push_back(make_slot("bias", false));
                return slots;
            }

            /** @brief Steps @p optimizer once and returns the values of its two parameters. */
            Updated updated_by_one_step(Optim::Optimizer& optimizer)
            {
                const SushiBLAS::Tensor first = optimizer.parameters()[0].parameter;
                const SushiBLAS::Tensor second = optimizer.parameters()[1].parameter;

                SA_EXPECT_OK(optimizer.step());

                return Updated{SaTest::read_tensor(first), SaTest::read_tensor(second)};
            }

            /** @brief Returns SGD options of K_RATE and K_DECAY with the kernel @p kernel. */
            static Optim::SGDOptions sgd_options(Optim::UpdateKernel kernel)
            {
                Optim::SGDOptions options;
                options.learning_rate = K_RATE;
                options.weight_decay = K_DECAY;
                options.kernel = kernel;
                return options;
            }

            /** @brief Returns AdamW options of K_RATE and K_DECAY with the kernel @p kernel. */
            static Optim::AdamWOptions adamw_options(Optim::UpdateKernel kernel)
            {
                Optim::AdamWOptions options;
                options.learning_rate = K_RATE;
                options.weight_decay = K_DECAY;
                options.kernel = kernel;
                return options;
            }

            /** @brief Holds the host adapter the optimizers are built on. */
            HostExecutionBackend backend;

        private:
            /** @brief Returns one slot called @p name that starts at K_START with no gradient. */
            Optim::ParameterSlot make_slot(const char* name, bool decays)
            {
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(backend.engine(), Shape{K_ELEMENTS}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(backend.engine(), Shape{K_ELEMENTS}, DType::Float32);
                EXPECT_TRUE(parameter.status().is_ok());
                EXPECT_TRUE(gradient.status().is_ok());

                SaTest::write_tensor(parameter.value(), each(K_START));
                SaTest::write_tensor(gradient.value(), each(0.0));

                Optim::ParameterSlot slot{name, std::move(parameter).value(),
                                          std::move(gradient).value()};
                slot.decays = decays;
                return slot;
            }
    };

    /** @brief Runs the death case on the slots of Small_ParameterDecay; it starts a child. */
    class Medium_SetLearningRate : public Small_ParameterDecay
    {
    };

    /** @brief Runs a case once per update kernel. */
    class Small_ParameterDecayKernel : public Small_ParameterDecay,
                                       public ::testing::WithParamInterface<Optim::UpdateKernel>
    {

    };

    /** @brief Runs a case once per optimizer type of the registry. */
    class Small_ParameterDecayType : public Small_ParameterDecay,
                                     public ::testing::WithParamInterface<std::string_view>
    {
        protected:
            /** @brief Returns the optimizer of the row's type built on @p attributes. */
            std::unique_ptr<Optim::Optimizer> make_from(std::string_view attributes)
            {
                Result<std::unique_ptr<Optim::Optimizer>> made = Optim::make_optimizer(
                    backend, GetParam(), make_slots(),
                    SushiAI::Test::require_value(Config::parse_json(attributes, "<test>")),
                    "optimizer");
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }
    };

    /** @brief Holds the attributes that state a decay and no decay_all. */
    constexpr std::string_view K_GROUPED = R"({"learning_rate": 0.5, "weight_decay": 0.5})";

    /** @brief Holds the attributes that state a decay and decay_all. */
    constexpr std::string_view K_ALL =
        R"({"learning_rate": 0.5, "weight_decay": 0.5, "decay_all": true})";
} // namespace

/** @brief Tests that SGD decays the weight and leaves the bias, whichever kernel runs. */
TEST_P(Small_ParameterDecayKernel, SGDDecaysOnlyTheSlotsThatTakeIt)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), sgd_options(GetParam())));

    const Updated after = updated_by_one_step(*optimizer);

    EXPECT_EQ(after.weight, each(K_DECAYED));
    EXPECT_EQ(after.bias, each(K_START));
}

/** @brief Tests that AdamW decays the weight and leaves the bias, whichever kernel runs. */
TEST_P(Small_ParameterDecayKernel, AdamWDecaysOnlyTheSlotsThatTakeIt)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, make_slots(), adamw_options(GetParam())));

    const Updated after = updated_by_one_step(*optimizer);

    EXPECT_EQ(after.weight, each(K_DECAYED));
    EXPECT_EQ(after.bias, each(K_START));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ParameterDecayKernel,
                         ::testing::Values(Optim::UpdateKernel::Fused,
                                           Optim::UpdateKernel::Separate));

/** @brief Tests that decay_all makes SGD decay the slot that would not take it. */
TEST_F(Small_ParameterDecay, SGDDecayAllDecaysEverySlot)
{
    Optim::SGDOptions options = sgd_options(Optim::UpdateKernel::Fused);
    options.decay_all = true;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), options));

    const Updated after = updated_by_one_step(*optimizer);

    EXPECT_EQ(after.weight, each(K_DECAYED));
    EXPECT_EQ(after.bias, each(K_DECAYED));
}

/** @brief Tests that decay_all makes AdamW decay the slot that would not take it. */
TEST_F(Small_ParameterDecay, AdamWDecayAllDecaysEverySlot)
{
    Optim::AdamWOptions options = adamw_options(Optim::UpdateKernel::Fused);
    options.decay_all = true;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, make_slots(), options));

    const Updated after = updated_by_one_step(*optimizer);

    EXPECT_EQ(after.weight, each(K_DECAYED));
    EXPECT_EQ(after.bias, each(K_DECAYED));
}

/** @brief Tests that an optimizer built without the key decay_all leaves the bias alone. */
TEST_P(Small_ParameterDecayType, WithoutTheKeyDecayAllTheBiasIsLeftAlone)
{
    const std::unique_ptr<Optim::Optimizer> optimizer = make_from(K_GROUPED);

    const Updated after = updated_by_one_step(*optimizer);

    EXPECT_EQ(after.weight, each(K_DECAYED));
    EXPECT_EQ(after.bias, each(K_START));
}

/** @brief Tests that the key decay_all reaches the optimizer the registry builds. */
TEST_P(Small_ParameterDecayType, TheKeyDecayAllReachesTheOptimizer)
{
    const std::unique_ptr<Optim::Optimizer> optimizer = make_from(K_ALL);

    const Updated after = updated_by_one_step(*optimizer);

    EXPECT_EQ(after.weight, each(K_DECAYED));
    EXPECT_EQ(after.bias, each(K_DECAYED));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ParameterDecayType,
                         ::testing::Values(std::string_view("sgd"), std::string_view("adamw")));

/** @brief Tests that set_learning_rate stops the process on the refusal of check_learning_rate. */
TEST_F(Medium_SetLearningRate, ANegativeRateIsFatal)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, make_slots(), Optim::SGDOptions{}));

    EXPECT_DEATH(optimizer->set_learning_rate(-0.5),
                 SaTest::death_pattern({"-0.5", "negative or not finite"}));
}
