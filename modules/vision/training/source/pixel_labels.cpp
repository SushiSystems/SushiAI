/**************************************************************************/
/* pixel_labels.cpp                                                       */
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
 * @file pixel_labels.cpp
 * @brief Implements the check of the mask role a per-pixel objective feeds its leaves from.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/pixel_labels.hpp>

#include <SushiAI/dtype/dtype.hpp>

namespace SushiAI::Vision::Training
{
    Status validate_pixel_labels(const SushiBLAS::Tensor& mask, std::int64_t pixels,
                                 std::int64_t classes, std::int32_t ignore_index)
    {
        SA_FAIL_IF(mask.dtype != DType::Int32, StatusCode::Incompatible,
                   "The batch's mask is {}; a class index per pixel is read as {}",
                   dtype_name(mask.dtype), dtype_name(DType::Int32));
        SA_FAIL_IF(mask.num_elements != pixels, StatusCode::Incompatible,
                   "The batch's mask holds {} pixels but the compiled step expects {}",
                   mask.num_elements, pixels);

        const std::int32_t* const label = mask.data_as<std::int32_t>();
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
        {
            SA_FAIL_IF(label[pixel] != ignore_index
                       && (label[pixel] < 0 || label[pixel] >= classes),
                       StatusCode::Incompatible,
                       "The mask holds {} at pixel {}; a class is below {} and {} names none",
                       label[pixel], pixel, classes, ignore_index);
        }

        return Status();
    }

} // namespace SushiAI::Vision::Training
