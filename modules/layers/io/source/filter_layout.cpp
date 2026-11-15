/**************************************************************************/
/* filter_layout.cpp                                                      */
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
 * @file filter_layout.cpp
 * @brief Defines the moves of convolution weights to the kernel-last layouts.
 * @author Mustafa Garip
 */

#include <SushiAI/io/filter_layout.hpp>

#include <cstddef>
#include <cstdint>

namespace SushiAI
{
    namespace IO
    {
        void copy_kernel_last(const float* source, const FilterExtents& extents,
                              float* destination) noexcept
        {
            const auto outputs = static_cast<std::size_t>(extents.outputs);
            const auto inputs = static_cast<std::size_t>(extents.inputs);
            const auto taps = static_cast<std::size_t>(extents.height * extents.width);
            for (std::size_t o = 0; o < outputs; ++o)
            {
                for (std::size_t i = 0; i < inputs; ++i)
                {
                    for (std::size_t t = 0; t < taps; ++t)
                        destination[(o * taps + t) * inputs + i] = source[(o * inputs + i) * taps + t];
                }
            }
        }

        void copy_depthwise_kernel_last(const float* source, const FilterExtents& extents,
                                        float* destination) noexcept
        {
            const auto channels = static_cast<std::size_t>(extents.outputs);
            const auto taps = static_cast<std::size_t>(extents.height * extents.width);
            for (std::size_t c = 0; c < channels; ++c)
            {
                for (std::size_t t = 0; t < taps; ++t)
                    destination[t * channels + c] = source[c * taps + t];
            }
        }
    } // namespace IO
} // namespace SushiAI
