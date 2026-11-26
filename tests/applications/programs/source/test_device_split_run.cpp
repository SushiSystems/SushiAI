/**************************************************************************/
/* test_device_split_run.cpp                                              */
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
 * @file test_device_split_run.cpp
 * @brief Tests a training step split across a GPU and a CPU, and the run without a split.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/execution/runtime_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include "share_runtime.hpp"
#include "tiny_run_files.hpp"
#include "training_program.hpp"

using namespace SushiAI;
namespace TinyRun = SushiAI::Apps::TinyRunTest;

namespace
{
    /** @brief The updates of every run: two epochs of two batches. */
    constexpr std::size_t K_STEPS = 4;

    /** @brief The widest gap allowed between the split loss and the GPU-only loss. */
    constexpr double K_LOSS_TOLERANCE = 1e-3;

    /** @brief The rows of each batch of eight the GPU share takes; the CPU takes the rest. */
    constexpr int K_GPU_ROWS = 6;

    /** @brief The rows of each batch the CPU share takes. */
    constexpr int K_CPU_ROWS = TinyRun::K_BATCH_ROWS - K_GPU_ROWS;

    /** @brief Returns the split of each batch into the GPU's rows and @p cpu_rows on the CPU. */
    std::string gpu_cpu_split(int cpu_rows)
    {
        return R"(, "device_split": [{"device": "gpu", "rows": )" + std::to_string(K_GPU_ROWS)
               + R"(}, {"device": "cpu", "rows": )" + std::to_string(cpu_rows) + "}]";
    }

    /** @brief Returns the loss on the last epoch line @p printed holds. */
    double last_printed_loss(const std::string& printed)
    {
        const std::size_t at = printed.rfind(" loss ");
        return at == std::string::npos ? -1.0 : std::strtod(printed.c_str() + at + 6, nullptr);
    }

    /** @brief Holds the outcome of a training run and what it printed. */
    struct TrainedRun
    {
        /** @brief What run_training_file returned. */
        Result<Apps::TrainingOutcome> outcome;

        /** @brief What the run wrote to standard output. */
        std::string printed;
    };

    /**
     * @class Medium_DeviceSplitRun
     * @brief Gives each case a backend over a runtime that drives the GPU class alone, and
     *        skips on a machine without a GPU or without a CPU device.
     */
    class Medium_DeviceSplitRun : public ::testing::Test
    {
        protected:
            /** @brief Opens the GPU runtime once and checks that a CPU device exists. */
            static void SetUpTestSuite()
            {
                Result<SushiRuntime::Runtime> gpu = SushiAI::Detail::create_runtime(
                    Apps::narrow_to_class(SushiRuntime::RuntimeConfig{}, Config::DeviceClass::GPU));
                Result<SushiRuntime::Runtime> cpu = SushiAI::Detail::create_runtime(
                    Apps::narrow_to_class(SushiRuntime::RuntimeConfig{}, Config::DeviceClass::CPU));
                if (!gpu.has_value() || !cpu.has_value()
                    || !Apps::check_share_device(gpu.value(), Config::DeviceClass::GPU).is_ok()
                    || !Apps::check_share_device(cpu.value(), Config::DeviceClass::CPU).is_ok())
                    return;
                runtime_ = std::make_unique<SushiRuntime::Runtime>(std::move(gpu.value()));
                engine_ = std::make_unique<SushiBLAS::Engine>(*runtime_);
                backend_ = std::make_unique<RuntimeExecutionBackend>(*runtime_, *engine_);
            }

            /** @brief Closes the backend, the engine and the runtime, in that order. */
            static void TearDownTestSuite()
            {
                backend_.reset();
                engine_.reset();
                runtime_.reset();
            }

            /** @brief Skips a case on a machine that lacks a GPU or a CPU device. */
            void SetUp() override
            {
                if (backend_ == nullptr)
                    GTEST_SKIP() << "this machine has no GPU device or no CPU device";
            }

            /** @brief Trains @p spec on the GPU backend and captures its standard output. */
            static TrainedRun train(const Config::TrainSpec& spec)
            {
                testing::internal::CaptureStdout();
                Result<Apps::TrainingOutcome> outcome =
                    Apps::run_training_file("<test>", spec, *backend_);
                return TrainedRun{std::move(outcome), testing::internal::GetCapturedStdout()};
            }

        private:
            /** @brief The runtime of the GPU class; opened once for the suite. */
            static std::unique_ptr<SushiRuntime::Runtime> runtime_;

            /** @brief The engine on the GPU runtime. */
            static std::unique_ptr<SushiBLAS::Engine> engine_;

            /** @brief The backend every case trains on. */
            static std::unique_ptr<RuntimeExecutionBackend> backend_;
    };

    std::unique_ptr<SushiRuntime::Runtime> Medium_DeviceSplitRun::runtime_;
    std::unique_ptr<SushiBLAS::Engine> Medium_DeviceSplitRun::engine_;
    std::unique_ptr<RuntimeExecutionBackend> Medium_DeviceSplitRun::backend_;
} // namespace

/** @brief Tests that a split run steps on a GPU device and on a CPU device, every batch. */
TEST_F(Medium_DeviceSplitRun, TheSplitStepRunsOnTheGpuAndOnTheCpu)
{
    const TrainedRun run = train(TinyRun::tiny_spec(gpu_cpu_split(K_CPU_ROWS)));

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    const std::vector<Apps::ShareRun>& shares = run.outcome.value().summary.shares;
    ASSERT_EQ(shares.size(), 2u) << run.printed;
    EXPECT_EQ(shares[0].device, Config::DeviceClass::GPU);
    EXPECT_EQ(shares[0].rows, K_GPU_ROWS);
    EXPECT_EQ(shares[0].step_runs, K_STEPS);
    EXPECT_EQ(shares[1].device, Config::DeviceClass::CPU);
    EXPECT_EQ(shares[1].rows, K_CPU_ROWS);
    EXPECT_EQ(shares[1].step_runs, K_STEPS);
    EXPECT_EQ(run.outcome.value().optimizer_steps, K_STEPS);
}

/** @brief Tests that a split run reaches the loss of the GPU-only run within the tolerance. */
TEST_F(Medium_DeviceSplitRun, TheSplitStepReachesTheGpuOnlyLoss)
{
    const TrainedRun alone = train(TinyRun::tiny_spec(""));
    const TrainedRun split = train(TinyRun::tiny_spec(gpu_cpu_split(K_CPU_ROWS)));

    ASSERT_TRUE(alone.outcome.has_value()) << alone.outcome.status().message() << alone.printed;
    ASSERT_TRUE(split.outcome.has_value()) << split.outcome.status().message() << split.printed;
    EXPECT_NEAR(last_printed_loss(split.printed), last_printed_loss(alone.printed),
                K_LOSS_TOLERANCE)
        << alone.printed << split.printed;
}

/** @brief Tests that a run without a split reports no share and prints no share line. */
TEST_F(Medium_DeviceSplitRun, WithoutASplitTheRunReportsNoShare)
{
    const TrainedRun run = train(TinyRun::tiny_spec(""));

    ASSERT_TRUE(run.outcome.has_value()) << run.outcome.status().message() << run.printed;
    EXPECT_TRUE(run.outcome.value().summary.shares.empty());
    EXPECT_EQ(run.printed.find("Devices"), std::string::npos) << run.printed;
    EXPECT_EQ(run.printed.find("share "), std::string::npos) << run.printed;
}

/** @brief Tests that shares whose rows miss the data's batch size are refused. */
TEST_F(Medium_DeviceSplitRun, SharesThatMissTheBatchSizeAreRefused)
{
    const TrainedRun run = train(TinyRun::tiny_spec(gpu_cpu_split(1)));

    SA_EXPECT_FAILURE(run.outcome, StatusCode::InvalidDescription, "7 rows", "batches of 8");
}
