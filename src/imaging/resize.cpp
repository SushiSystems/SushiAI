/**************************************************************************/
/* resize.cpp                                                             */
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
 * @file resize.cpp
 * @brief Implements the bilinear resize of an 8-bit image and the training-load size rule.
 * @author Mustafa Garip
 */

#include <SushiAI/imaging/resize.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <SushiAI/core/common.hpp>

#include "bilinear.hpp"

namespace SushiAI::Imaging
{
    ImageSize fit_long_side(ImageSize original, std::int32_t image_size) noexcept
    {
        const std::int32_t long_side = std::max(original.height, original.width);
        const double ratio = static_cast<double>(image_size) / static_cast<double>(long_side);

        if (ratio == 1.0)
            return original;

        const double height = std::ceil(static_cast<double>(original.height) * ratio);
        const double width = std::ceil(static_cast<double>(original.width) * ratio);
        const double limit = static_cast<double>(image_size);

        return ImageSize{static_cast<std::int32_t>(std::min(height, limit)),
                         static_cast<std::int32_t>(std::min(width, limit))};
    }

    void resize_bilinear(ImageView source, ImageSize size, Image& destination)
    {
        check_source(source, destination, "resize_bilinear");
        SA_THROW_IF(!is_valid_size(size),
                    "resize_bilinear: a destination size of {}x{} is outside [1, {}]", size.height,
                    size.width, K_MAX_IMAGE_SIDE);

        destination.reset(size, source.channels());
        Detail::resize_rows(source, size, destination.data(), destination.row_bytes());
    }

    void resize_long_side(ImageView source, std::int32_t image_size, Image& destination)
    {
        SA_THROW_IF(!is_valid_side(image_size),
                    "resize_long_side: an image size of {} is outside [1, {}]", image_size,
                    K_MAX_IMAGE_SIDE);
        check_source(source, destination, "resize_long_side");

        const ImageSize fitted = fit_long_side(source.size(), image_size);
        destination.reset(fitted, source.channels());

        if (fitted == source.size())
        {
            std::memcpy(destination.data(), source.data(), source.byte_count());
            return;
        }

        Detail::resize_rows(source, fitted, destination.data(), destination.row_bytes());
    }
} // namespace SushiAI::Imaging
