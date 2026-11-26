/**************************************************************************/
/* test_share_runtime.cpp                                                 */
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
 * @file test_share_runtime.cpp
 * @brief Tests how a training file's device split narrows the runtime the run opens.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/config/train_config.hpp>
#include "share_runtime.hpp"

using namespace SushiAI;

namespace
{
    /** @brief Returns a training file's spec whose split names @p first, then the other class. */
    Config::TrainSpec split_spec(Config::DeviceClass first)
    {
        const Config::DeviceClass second =
            first == Config::DeviceClass::GPU ? Config::DeviceClass::CPU : Config::DeviceClass::GPU;
        Config::TrainSpec spec;
        spec.device_split = {Config::DeviceShare{first, 6}, Config::DeviceShare{second, 2}};
        return spec;
    }

    /** @brief Returns a runtime configuration that differs from the default in every field read. */
    SushiRuntime::RuntimeConfig stated_config()
    {
        SushiRuntime::RuntimeConfig config;
        config.worker_count = 3;
        config.devices.include_accelerator = false;
        return config;
    }
} // namespace

/** @brief Tests that a file without a split leaves the runtime configuration as it was. */
TEST(Small_ShareRuntime, WithoutASplitTheRuntimeConfigIsUnchanged)
{
    const SushiRuntime::RuntimeConfig narrowed =
        Apps::narrow_to_split(stated_config(), Config::TrainSpec{});

    EXPECT_TRUE(narrowed.devices.include_cpu);
    EXPECT_TRUE(narrowed.devices.include_gpu);
    EXPECT_FALSE(narrowed.devices.include_accelerator);
    EXPECT_EQ(narrowed.worker_count, 3u);
}

/** @brief Tests that a split whose first share is the GPU narrows the runtime to GPUs. */
TEST(Small_ShareRuntime, AGpuFirstShareNarrowsTheRuntimeToGpus)
{
    const SushiRuntime::RuntimeConfig narrowed =
        Apps::narrow_to_split(stated_config(), split_spec(Config::DeviceClass::GPU));

    EXPECT_TRUE(narrowed.devices.include_gpu);
    EXPECT_FALSE(narrowed.devices.include_cpu);
    EXPECT_FALSE(narrowed.devices.include_accelerator);
    EXPECT_EQ(narrowed.worker_count, 3u);
}

/** @brief Tests that a split whose first share is the CPU narrows the runtime to CPUs. */
TEST(Small_ShareRuntime, ACpuFirstShareNarrowsTheRuntimeToCpus)
{
    const SushiRuntime::RuntimeConfig narrowed =
        Apps::narrow_to_split(stated_config(), split_spec(Config::DeviceClass::CPU));

    EXPECT_TRUE(narrowed.devices.include_cpu);
    EXPECT_FALSE(narrowed.devices.include_gpu);
    EXPECT_FALSE(narrowed.devices.include_accelerator);
}
