/**************************************************************************/
/* argmax_classes.hpp                                                     */
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
 * @file argmax_classes.hpp
 * @brief Defines the pick of the highest-scoring class of each pixel.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiRuntime/core/span.hpp>

/** @brief Holds the mask geometry and the metrics of semantic segmentation. */
namespace SushiAI::Vision::Segment
{
    /**
     * @brief Writes, for each pixel, the class whose score is the largest.
     *
     * Equal scores give the lowest class.
     * @tparam Real float or double.
     * @param scores @p classes scores per pixel, pixel after pixel.
     * @pre @p classes is at least 1 and scores.size() equals picked.size() * classes.
     */
    template<typename Real>
    void argmax_classes(SushiRuntime::span<const Real> scores, std::int32_t classes,
                        SushiRuntime::span<std::int32_t> picked) noexcept
    {
        const std::size_t width = static_cast<std::size_t>(classes);
        for (std::size_t pixel = 0; pixel < picked.size(); ++pixel)
        {
            const Real* const row = scores.data() + pixel * width;
            std::size_t best = 0;
            for (std::size_t class_index = 1; class_index < width; ++class_index)
            {
                if (row[class_index] > row[best])
                    best = class_index;
            }

            picked[pixel] = static_cast<std::int32_t>(best);
        }
    }

} // namespace SushiAI::Vision::Segment
