/**************************************************************************/
/* host_readback.cpp                                                      */
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
 * @file host_readback.cpp
 * @brief Implements HostReadback.
 * @author Mustafa Garip
 */

#include "host_readback.hpp"

#include <cstddef>
#include <utility>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Result<SushiRuntime::span<const float>> HostReadback::bind(
            const SushiBLAS::Tensor& tensor) noexcept
        {
            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<float> values(static_cast<std::size_t>(tensor.num_elements));
                const SushiRuntime::span<const float> view(values.data(), values.size());
                entries_.push_back(Entry{&tensor, std::move(values)});
                return view;
            });
        }

        Status HostReadback::refresh() const noexcept
        {
            for (const Entry& entry : entries_)
            {
                SA_TRY(read_elements(*entry.tensor, 0,
                                     SushiRuntime::span<float>(entry.values.data(),
                                                               entry.values.size())));
            }

            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
