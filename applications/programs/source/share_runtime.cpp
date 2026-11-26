/**************************************************************************/
/* share_runtime.cpp                                                      */
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
 * @file share_runtime.cpp
 * @brief Implements the narrowing of a runtime to one device class and its checks.
 * @author Mustafa Garip
 */

#include "share_runtime.hpp"

#include <utility>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Returns the kind SushiRuntime reports for a device of @p device's class. */
            [[nodiscard]] SushiRuntime::DeviceKind kind_of(Config::DeviceClass device) noexcept
            {
                switch (device)
                {
                    case Config::DeviceClass::GPU: return SushiRuntime::DeviceKind::GPU;
                    case Config::DeviceClass::CPU: return SushiRuntime::DeviceKind::CPU;
                }
                return SushiRuntime::DeviceKind::Other;
            }
        } // namespace

        SushiRuntime::RuntimeConfig narrow_to_class(SushiRuntime::RuntimeConfig config,
                                                    Config::DeviceClass device) noexcept
        {
            config.devices.include_gpu = device == Config::DeviceClass::GPU;
            config.devices.include_cpu = device == Config::DeviceClass::CPU;
            config.devices.include_accelerator = false;
            return config;
        }

        SushiRuntime::RuntimeConfig narrow_to_split(SushiRuntime::RuntimeConfig config,
                                                    const Config::TrainSpec& spec) noexcept
        {
            if (spec.device_split.empty())
                return config;
            return narrow_to_class(std::move(config), spec.device_split.front().device);
        }

        Status check_share_device(const SushiRuntime::Runtime& runtime,
                                  Config::DeviceClass device)
        {
            const SushiRuntime::NodeFacts& facts = runtime.topology();
            SA_FAIL_IF(facts.devices.empty() || facts.devices.front().kind != kind_of(device),
                       StatusCode::Unsupported,
                       "the share of class '{}' found no device of that class first in its "
                       "runtime, so this machine cannot run that share",
                       Config::device_class_name(device));
            return Status();
        }

        std::string first_device_name(const SushiRuntime::Runtime& runtime)
        {
            const SushiRuntime::NodeFacts& facts = runtime.topology();
            return facts.devices.empty() ? std::string() : facts.devices.front().name;
        }
    } // namespace Apps
} // namespace SushiAI
