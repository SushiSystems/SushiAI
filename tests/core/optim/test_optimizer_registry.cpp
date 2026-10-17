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
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/sgd.hpp>

using namespace SushiAI;
using Optim::K_OPTIMIZER_REGISTRY;

// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Bounds the error of a rate a configuration states and a double holds as read. */
    constexpr double K_RATE_TOLERANCE = 0.0;

    /** @brief Provides an engine and one parameter slot for every optimizer built. */
    class Medium_OptimizerRegistry : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns one slot "weight" of four elements. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                Result<SushiBLAS::Tensor> parameter =
                    allocate_tensor(*engine, Shape{4}, DType::Float32);
                Result<SushiBLAS::Tensor> gradient =
                    allocate_tensor(*engine, Shape{4}, DType::Float32);
                EXPECT_TRUE(parameter.status().is_ok());
                EXPECT_TRUE(gradient.status().is_ok());

                std::vector<Optim::ParameterSlot> slots;
                slots.push_back(Optim::ParameterSlot{"weight", std::move(parameter).value(),
                                                     std::move(gradient).value()});
                return slots;
            }
    };

    /** @brief Parses @p text as the attributes of an optimizer. */
    Config::Value attributes_from(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }
} // namespace

/** @brief Tests that the registry holds one row per optimizer type, each with its factory. */
TEST(Small_OptimizerRegistry, EachRowNamesItsTypeAndFactory)
{
    ASSERT_EQ(K_OPTIMIZER_REGISTRY.size(), 2u);
    EXPECT_EQ(K_OPTIMIZER_REGISTRY[0].type, Optim::K_ADAMW_TYPE);
    EXPECT_EQ(K_OPTIMIZER_REGISTRY[0].make, &Optim::make_adamw);
    EXPECT_EQ(K_OPTIMIZER_REGISTRY[1].type, Optim::K_SGD_TYPE);
    EXPECT_EQ(K_OPTIMIZER_REGISTRY[1].make, &Optim::make_sgd);
}

/** @brief Tests that every row builds an optimizer that names itself by the row's type. */
TEST_F(Medium_OptimizerRegistry, EveryTypeBuildsAnOptimizerOfThatType)
{
    for (const Optim::OptimizerEntry& entry : K_OPTIMIZER_REGISTRY)
    {
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Optim::Optimizer> optimizer,
            Optim::make_optimizer(*engine, entry.type, make_slots(), attributes_from("{}"),
                                  "optimizer"));

        EXPECT_EQ(optimizer->type_name(), entry.type);
        EXPECT_EQ(optimizer->parameter_count(), 1u);
    }
}

/** @brief Tests that the learning rate the attributes state reaches the optimizer built. */
TEST_F(Medium_OptimizerRegistry, TheStatedRateReachesTheOptimizer)
{
    for (const Optim::OptimizerEntry& entry : K_OPTIMIZER_REGISTRY)
    {
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Optim::Optimizer> optimizer,
            entry.make(*engine, make_slots(), attributes_from(R"({"learning_rate": 0.25})"),
                       "optimizer"));

        EXPECT_NEAR(optimizer->learning_rate(), 0.25, K_RATE_TOLERANCE) << entry.type;
    }
}
