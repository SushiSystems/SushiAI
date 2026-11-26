/**************************************************************************/
/* share_runtime.hpp                                                      */
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
 * @file share_runtime.hpp
 * @brief Declares the runtime configuration and the device checks of a share of a split step.
 * @author Mustafa Garip
 */

#pragma once

#include <string>

#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Returns @p config with its device filter narrowed to the class of @p device.
         * @return The same configuration, admitting no device of another class.
         */
        [[nodiscard]] SushiRuntime::RuntimeConfig narrow_to_class(
            SushiRuntime::RuntimeConfig config, Config::DeviceClass device) noexcept;

        /**
         * @brief Returns the runtime configuration the first share of @p spec's split runs on.
         * @return @p config narrowed to the class of the first share, or @p config unchanged for
         *         a file without "device_split".
         */
        [[nodiscard]] SushiRuntime::RuntimeConfig narrow_to_split(
            SushiRuntime::RuntimeConfig config, const Config::TrainSpec& spec) noexcept;

        /**
         * @brief Checks that the first device of @p runtime, the one tensors are placed on, is of
         *        the class of @p device.
         * @return Unsupported, naming the class, for a runtime with no device or a first device
         *         of another class.
         */
        [[nodiscard]] Status check_share_device(const SushiRuntime::Runtime& runtime,
                                                Config::DeviceClass device);

        /** @brief Returns the name the first device of @p runtime reports, or "" for none. */
        [[nodiscard]] std::string first_device_name(const SushiRuntime::Runtime& runtime);
    } // namespace Apps
} // namespace SushiAI
