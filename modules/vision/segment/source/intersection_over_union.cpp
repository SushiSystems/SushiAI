/**************************************************************************/
/* intersection_over_union.cpp                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file intersection_over_union.cpp
 * @brief Implements intersection over union per class and its mean.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/segment/intersection_over_union.hpp>

namespace SushiAI::Vision::Segment
{
    bool class_occurs(const ConfusionMatrix& matrix, std::int32_t class_index) noexcept
    {
        return matrix.truth_pixels(class_index) > 0;
    }

    double class_iou(const ConfusionMatrix& matrix, std::int32_t class_index) noexcept
    {
        const std::int64_t both = matrix.count(class_index, class_index);
        const std::int64_t either = matrix.truth_pixels(class_index)
                                  + matrix.predicted_pixels(class_index) - both;
        if (either == 0)
            return 0.0;

        return static_cast<double>(both) / static_cast<double>(either);
    }

    double mean_iou(const ConfusionMatrix& matrix) noexcept
    {
        double total = 0.0;
        std::int32_t occurring = 0;
        for (std::int32_t class_index = 0; class_index < matrix.class_count(); ++class_index)
        {
            if (!class_occurs(matrix, class_index))
                continue;

            total += class_iou(matrix, class_index);
            ++occurring;
        }

        if (occurring == 0)
            return 0.0;

        return total / static_cast<double>(occurring);
    }

} // namespace SushiAI::Vision::Segment
