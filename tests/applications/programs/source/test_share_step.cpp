/**************************************************************************/
/* test_share_step.cpp                                                    */
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
 * @file test_share_step.cpp
 * @brief Tests one share of a split step on a CPU runtime of its own.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include "registries.hpp"
#include "share_step.hpp"
#include "tiny_run_files.hpp"

using namespace SushiAI;
namespace TinyRun = SushiAI::Apps::TinyRunTest;

namespace
{
    /** @brief The share the cases build: the last two rows of each batch, on the CPU. */
    constexpr Config::DeviceShare K_CPU_SHARE{Config::DeviceClass::CPU, 2};

    /** @brief The parameters of the tiny model: a weight and a bias for each linear layer. */
    constexpr std::size_t K_PARAMETERS = 4;

    /** @brief Gives each case a CPU share of the tiny run, or skips without a CPU device. */
    class Medium_ShareStep : public ::testing::Test
    {
        protected:
            /** @brief Builds the share; a machine without a CPU device skips the case. */
            void SetUp() override
            {
                spec_ = TinyRun::tiny_spec("");
                const Config::ModelSpec model =
                    SushiAI::Test::require_value(Config::load_model(spec_.model_path));
                Result<std::unique_ptr<Apps::ShareStep>> made = Apps::ShareStep::make(
                    Apps::compose_registries().value(), spec_, model, K_CPU_SHARE,
                    TinyRun::K_BATCH_ROWS - K_CPU_SHARE.rows, 1, SushiRuntime::RuntimeConfig{});
                if (!made.has_value() && made.status().code() == StatusCode::Unsupported)
                    GTEST_SKIP() << made.status().message();
                share_ = SushiAI::Test::require_value(std::move(made));
                share_->enter_epoch(0);
            }

            /** @brief The training file the share was built from; outlives the share. */
            Config::TrainSpec spec_;

            /** @brief The share under test. */
            std::unique_ptr<Apps::ShareStep> share_;
    };
} // namespace

/** @brief Tests that a share replays its step once for each batch it runs. */
TEST_F(Medium_ShareStep, AShareReplaysItsStepOncePerBatch)
{
    SA_EXPECT_OK(share_->run(0));
    SA_EXPECT_OK(share_->run(1));

    const Apps::ShareRun ran = share_->report();
    EXPECT_EQ(ran.device, Config::DeviceClass::CPU);
    EXPECT_EQ(ran.rows, K_CPU_SHARE.rows);
    EXPECT_EQ(ran.step_runs, 2u);
    EXPECT_FALSE(ran.device_name.empty());
}

/** @brief Tests that a share holds a gradient for every parameter of the model. */
TEST_F(Medium_ShareStep, AShareHoldsEveryParameterOfTheModel)
{
    EXPECT_EQ(share_->parameters().size(), K_PARAMETERS);
}

/** @brief Tests that a share runs no batch the data does not have. */
TEST_F(Medium_ShareStep, AShareRefusesABatchPastTheEpoch)
{
    SA_EXPECT_FAILURE(share_->run(2), StatusCode::InvalidDescription, "Batch 2 is out of range");
}
