/**************************************************************************/
/* filter_layout.hpp                                                      */
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
 * @file filter_layout.hpp
 * @brief Declares the moves of convolution weights from the channel-first layout to SushiAI's.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <cstdint>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Holds the extents of a convolution weight stored [out, in, kh, kw]. */
        struct FilterExtents
        {
            std::int64_t outputs = 0; /**< The filters. */
            std::int64_t inputs = 0;  /**< The channels each filter reads. */
            std::int64_t height = 0;  /**< The kernel's rows. */
            std::int64_t width = 0;   /**< The kernel's columns. */
        };

        /**
         * @brief Copies a weight stored [out, in, kh, kw] into @p destination as [out, kh, kw, in].
         * @pre @p source and @p destination each hold the product of @p extents and do not overlap.
         */
        void copy_kernel_last(const float* source, const FilterExtents& extents,
                              float* destination) noexcept;

        /**
         * @brief Copies a depthwise weight stored [C, 1, kh, kw] into @p destination as
         *        [kh, kw, C].
         * @param extents Its outputs are the channels C; its inputs must be 1.
         * @pre @p source and @p destination each hold C kh kw floats and do not overlap.
         */
        void copy_depthwise_kernel_last(const float* source, const FilterExtents& extents,
                                        float* destination) noexcept;
    } // namespace IO
} // namespace SushiAI
