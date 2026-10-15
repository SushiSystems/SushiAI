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
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
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

    /** @brief Holds a parameter after one decayed update of a zero gradient, for both rules. */
    constexpr double K_DECAYED = K_START * (1.0 - K_RATE * K_DECAY);

    /** @brief Bounds the absolute error of a float32 value of order one after one update. */
    constexpr double K_UPDATE_TOLERANCE = 1e-6;

    /** @brief Provides an engine and two slots of zero gradient, the second without decay. */
    class Unit_ParameterDecay : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns "weight", which decays, then "bias", which does not. */
            std::vector<Optim::ParameterSlot> make_slots()
            {
                std::vector<Optim::ParameterSlot> slots;
                for (const std::string_view name : {std::string_view("weight"),
                                                    std::string_view("bias")})
                {
                    Result<SushiBLAS::Tensor> parameter =
                        allocate_tensor(*engine, Shape{K_ELEMENTS}, DType::Float32);
                    Result<SushiBLAS::Tensor> gradient =
                        allocate_tensor(*engine, Shape{K_ELEMENTS}, DType::Float32);
                    EXPECT_TRUE(parameter.status().is_ok());
                    EXPECT_TRUE(gradient.status().is_ok());

                    const std::size_t count = static_cast<std::size_t>(K_ELEMENTS);
                    SaTest::write_tensor(parameter.value(), std::vector<double>(count, K_START));
                    SaTest::write_tensor(gradient.value(), std::vector<double>(count, 0.0));

                    Optim::ParameterSlot slot{std::string(name), std::move(parameter).value(),
                                              std::move(gradient).value()};
                    slot.decays = name == "weight";
                    slots.push_back(std::move(slot));
                }
                return slots;
            }

            /** @brief Steps @p optimizer once and expects the two parameters' values. */
            void expect_after_one_step(Optim::Optimizer& optimizer, double weight, double bias)
            {
                const SushiBLAS::Tensor first = optimizer.parameters()[0].parameter;
                const SushiBLAS::Tensor second = optimizer.parameters()[1].parameter;

                SA_EXPECT_OK(optimizer.step());
                flush();

                for (const double value : SaTest::read_tensor(first))
                    EXPECT_NEAR(value, weight, K_UPDATE_TOLERANCE);
                for (const double value : SaTest::read_tensor(second))
                    EXPECT_NEAR(value, bias, K_UPDATE_TOLERANCE);
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
    };
} // namespace

/** @brief Tests that a slot made by hand decays, as every slot did before the flag. */
TEST_F(Unit_ParameterDecay, ASlotDecaysUnlessItSaysOtherwise)
{
    const Optim::ParameterSlot slot{};
    EXPECT_TRUE(slot.decays);
}

/** @brief Tests that both kernels of SGD decay the weight and leave the bias. */
TEST_F(Unit_ParameterDecay, SGDDecaysOnlyTheSlotsThatTakeIt)
{
    for (const Optim::UpdateKernel kernel : {Optim::UpdateKernel::Fused,
                                             Optim::UpdateKernel::Separate})
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                                Optim::SGD::make(*engine, make_slots(), sgd_options(kernel)));
        expect_after_one_step(*optimizer, K_DECAYED, K_START);
    }
}

/** @brief Tests that both kernels of AdamW decay the weight and leave the bias. */
TEST_F(Unit_ParameterDecay, AdamWDecaysOnlyTheSlotsThatTakeIt)
{
    for (const Optim::UpdateKernel kernel : {Optim::UpdateKernel::Fused,
                                             Optim::UpdateKernel::Separate})
    {
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Optim::AdamW> optimizer,
            Optim::AdamW::make(*engine, make_slots(), adamw_options(kernel)));
        expect_after_one_step(*optimizer, K_DECAYED, K_START);
    }
}

/** @brief Tests that decay_all decays the slot that would not take it, in both optimizers. */
TEST_F(Unit_ParameterDecay, DecayAllDecaysEverySlot)
{
    Optim::SGDOptions sgd = sgd_options(Optim::UpdateKernel::Fused);
    sgd.decay_all = true;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> first,
                            Optim::SGD::make(*engine, make_slots(), sgd));
    expect_after_one_step(*first, K_DECAYED, K_DECAYED);

    Optim::AdamWOptions adamw = adamw_options(Optim::UpdateKernel::Fused);
    adamw.decay_all = true;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> second,
                            Optim::AdamW::make(*engine, make_slots(), adamw));
    expect_after_one_step(*second, K_DECAYED, K_DECAYED);
}

/** @brief Tests that the key decay_all reaches both optimizers and is false when absent. */
TEST_F(Unit_ParameterDecay, TheKeyDecayAllReachesTheOptimizer)
{
    const auto attributes = [](std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    };
    constexpr std::string_view K_GROUPED =
        R"({"learning_rate": 0.5, "weight_decay": 0.5})";
    constexpr std::string_view K_ALL =
        R"({"learning_rate": 0.5, "weight_decay": 0.5, "decay_all": true})";

    for (const std::string_view type : {std::string_view("sgd"), std::string_view("adamw")})
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::Optimizer> grouped,
                                Optim::make_optimizer(*engine, type, make_slots(),
                                                      attributes(K_GROUPED), "optimizer"));
        expect_after_one_step(*grouped, K_DECAYED, K_START);

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::Optimizer> all,
                                Optim::make_optimizer(*engine, type, make_slots(),
                                                      attributes(K_ALL), "optimizer"));
        expect_after_one_step(*all, K_DECAYED, K_DECAYED);
    }
}
