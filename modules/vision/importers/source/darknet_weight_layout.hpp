/**************************************************************************/
/* darknet_weight_layout.hpp                                              */
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
 * @file darknet_weight_layout.hpp
 * @brief Defines the move of Darknet convolution weights to the kernel-last layout.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <SushiAI/io/filter_layout.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            /**
             * @brief Copies convolution weights into a buffer, moved to [out, kh, kw, in].
             * @param source The weights as the file orders them, [out, in, kh, kw].
             * @param shape The layer whose filters, channels and kernel side give the layout.
             * @param destination The first of as many floats as @p source holds.
             */
            inline void copy_kernel_last(const std::vector<float>& source,
                                         const DarknetConvolutionShape& shape,
                                         float* destination) noexcept
            {
                const IO::FilterExtents extents{shape.filters, shape.channels, shape.size,
                                                shape.size};
                IO::copy_kernel_last(source.data(), extents, destination);
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
