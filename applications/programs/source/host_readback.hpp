/**************************************************************************/
/* host_readback.hpp                                                      */
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
 * @file host_readback.hpp
 * @brief Declares HostReadback, the host copies of the outputs a report decodes.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds a host copy of each float32 output it binds; refresh() copies them. */
        class HostReadback
        {
            public:
                /**
                 * @brief Binds @p tensor, which outlives this object.
                 * @return A view of its host copy, stable for this object's lifetime, or the
                 *         failure of the allocation.
                 */
                [[nodiscard]] Result<SushiRuntime::span<const float>> bind(
                    const SushiBLAS::Tensor& tensor) noexcept;

                /**
                 * @brief Copies every bound tensor into its host copy.
                 * @return Success, or the failure of the first copy that fails.
                 */
                [[nodiscard]] Status refresh() const noexcept;

            private:
                /** @brief Pairs a bound tensor with its host copy. */
                struct Entry
                {
                    const SushiBLAS::Tensor* tensor; /**< The output; not owned. */
                    mutable std::vector<float> values; /**< Its host copy. */
                };

                /// @brief Holds the bound outputs in bind order.
                std::vector<Entry> entries_;
        };

    } // namespace Apps
} // namespace SushiAI
