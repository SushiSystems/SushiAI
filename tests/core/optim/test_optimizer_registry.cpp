/**************************************************************************/
/* test_optimizer_registry.cpp                                            */
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
 * @file test_optimizer_registry.cpp
 * @brief Tests the rows of the optimizer registry and the optimizer each row builds.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/sgd.hpp>

using namespace SushiAI;
using Optim::K_OPTIMIZER_REGISTRY;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Builds every optimizer of the registry on the host backend, one row per type. */
    class Small_OptimizerRegistry : public ::testing::TestWithParam<Optim::OptimizerEntry>
    {
        protected:
            /** @brief Returns one slot "weight" of four elements. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(backend.engine(), Shape{4}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(backend.engine(), Shape{4}, DType::Float32);
                EXPECT_TRUE(parameter.status().is_ok());
                EXPECT_TRUE(gradient.status().is_ok());

                std::vector<Optim::ParameterSlot> slots;
                slots.push_back(Optim::ParameterSlot{"weight", std::move(parameter).value(),
                                                     std::move(gradient).value()});
                return slots;
            }

            /** @brief Holds the host adapter every optimizer is built on. */
            HostExecutionBackend backend;
    };

    /** @brief Parses @p text as the attributes of an optimizer. */
    Config::Value attributes_from(std::string_view text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that a row builds an optimizer that names itself by the row's type. */
TEST_P(Small_OptimizerRegistry, ARowBuildsAnOptimizerOfItsType)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::Optimizer> optimizer,
        Optim::make_optimizer(backend, GetParam().type, make_slots(), attributes_from("{}"),
                              "optimizer"));

    EXPECT_EQ(optimizer->type_name(), GetParam().type);
    EXPECT_EQ(optimizer->parameter_count(), 1u);
}

/** @brief Tests that the learning rate the attributes state reaches the optimizer built. */
TEST_P(Small_OptimizerRegistry, TheStatedRateReachesTheOptimizer)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::Optimizer> optimizer,
        GetParam().make(backend, make_slots(), attributes_from(R"({"learning_rate": 0.25})"),
                        "optimizer"));

    EXPECT_EQ(optimizer->learning_rate(), 0.25);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_OptimizerRegistry, ::testing::ValuesIn(K_OPTIMIZER_REGISTRY));
