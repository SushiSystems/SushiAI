/**************************************************************************/
/* storage_device.hpp                                                     */
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
 * @file storage_device.hpp
 * @brief Declares the device check that places a distribution pack beside its tensors.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Returns the device a pack beside @p tensor is allocated on, its storage's device.
         * @param what Names the tensor's role in a refusal, with @p index.
         * @return The device, or InvalidDescription for a tensor with no storage or a
         *         runtime-placed storage, which names no device.
         */
        [[nodiscard]] inline Result<std::size_t> require_pack_device(
            const SushiBLAS::Tensor& tensor, std::string_view what, std::size_t index)
        {
            SA_FAIL_IF(!tensor.storage, StatusCode::InvalidDescription, "{} {} has no storage",
                       what, index);
            const std::optional<std::size_t> device = tensor.storage->device_index();
            SA_FAIL_IF(!device, StatusCode::InvalidDescription,
                       "{} {} is placed by the runtime, so it names no device for its pack",
                       what, index);
            return *device;
        }
    } // namespace Distribution
} // namespace SushiAI
