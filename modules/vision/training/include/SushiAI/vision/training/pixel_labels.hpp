/**************************************************************************/
/* pixel_labels.hpp                                                       */
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
 * @file pixel_labels.hpp
 * @brief Declares the check of the mask role a per-pixel objective feeds its leaves from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Checks that @p mask holds one class or the ignore index for each of the pixels.
     * @param mask INT32 class indices of any shape, readable from the host.
     * @param pixels The element count the step was compiled for, N * H * W.
     * @param classes C; a label is in [0, C) or equals @p ignore_index.
     * @return Success, or Incompatible for another element type or count, or for a label
     *         outside those bounds, naming its pixel.
     */
    [[nodiscard]] Status validate_pixel_labels(const SushiBLAS::Tensor& mask,
                                               std::int64_t pixels, std::int64_t classes,
                                               std::int32_t ignore_index) noexcept;

} // namespace SushiAI::Vision::Training
