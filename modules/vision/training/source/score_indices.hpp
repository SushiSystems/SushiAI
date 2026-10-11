/**************************************************************************/
/* score_indices.hpp                                                      */
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
 * @file score_indices.hpp
 * @brief Defines the index of each pixel's own score in [R, C] scores read as R * C rows.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiAI::Vision::Training::Detail
{
    /**
     * @brief Writes, for each of @p pixels pixels r, r * classes + label[r] into @p out.
     *
     * An ignored pixel gets r * classes, the first score of its own row, so its index is valid.
     * @pre Every label is in [0, classes) or is @p ignore_index, and pixels * classes fits INT32.
     */
    inline void write_score_indices(const std::int32_t* label, std::int64_t pixels,
                                    std::int64_t classes, std::int32_t ignore_index,
                                    std::int32_t* out) noexcept
    {
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
        {
            const std::int64_t column = label[pixel] != ignore_index ? label[pixel] : 0;
            out[pixel] = static_cast<std::int32_t>(pixel * classes + column);
        }
    }

} // namespace SushiAI::Vision::Training::Detail
