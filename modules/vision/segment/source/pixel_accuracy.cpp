/**************************************************************************/
/* pixel_accuracy.cpp                                                     */
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
 * @file pixel_accuracy.cpp
 * @brief Implements pixel accuracy.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/segment/pixel_accuracy.hpp>

#include <cstdint>

namespace SushiAI::Vision::Segment
{
    double pixel_accuracy(const ConfusionMatrix& matrix) noexcept
    {
        const std::int64_t counted = matrix.pixel_count();
        if (counted == 0)
            return 0.0;

        std::int64_t right = 0;
        for (std::int32_t class_index = 0; class_index < matrix.class_count(); ++class_index)
            right += matrix.count(class_index, class_index);

        return static_cast<double>(right) / static_cast<double>(counted);
    }

} // namespace SushiAI::Vision::Segment
