/**************************************************************************/
/* pixel_extents.hpp                                                      */
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
 * @file pixel_extents.hpp
 * @brief Defines the extents of the logits a per-pixel objective reads.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the extents of NHWC [N, H, W, C] logits: one score per pixel and class. */
    struct PixelExtents
    {
        std::int64_t batch = 0;   /**< N, the samples of a batch. */
        std::int64_t height = 0;  /**< H, the rows of a mask. */
        std::int64_t width = 0;   /**< W, the columns of a mask. */
        std::int64_t classes = 0; /**< C, the scores of a pixel. */

        /**
         * @brief Returns N * H * W, the pixels of a batch.
         * @pre The three extents are positive and their product fits in int64.
         */
        [[nodiscard]] constexpr std::int64_t pixels() const noexcept
        {
            return batch * height * width;
        }
    };

} // namespace SushiAI::Vision::Training
