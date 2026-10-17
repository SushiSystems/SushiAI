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
 * @brief Implements the resizes of an 8-bit image and the training-load size rule.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/resize.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <SushiAI/failure/error.hpp>

#include "bilinear.hpp"

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /**
         * @brief Returns the source index the centre of destination index @p at falls in.
         * @pre @p source and @p destination are positive and @p at is in [0, destination).
         */
        [[nodiscard]] constexpr std::int32_t nearest_source_index(
            std::int32_t at, std::int32_t source, std::int32_t destination) noexcept
        {
            const std::int64_t index = (2 * static_cast<std::int64_t>(at) + 1) * source /
                                       (2 * static_cast<std::int64_t>(destination));

            return static_cast<std::int32_t>(std::min<std::int64_t>(index, source - 1));
        }
    } // namespace

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

    void resize_nearest(ImageView source, ImageSize size, Image& destination)
    {
        check_source(source, destination, "resize_nearest");

        // reset() throws on a size outside the bounds of an image.
        destination.reset(size, source.channels());

        const std::size_t channels = static_cast<std::size_t>(source.channels());
        std::vector<std::size_t> offsets(static_cast<std::size_t>(size.width));
        for (std::int32_t x = 0; x < size.width; ++x)
        {
            offsets[static_cast<std::size_t>(x)] =
                static_cast<std::size_t>(nearest_source_index(x, source.width(), size.width)) *
                channels;
        }

        for (std::int32_t y = 0; y < size.height; ++y)
        {
            const std::uint8_t* const in =
                source.row(nearest_source_index(y, source.height(), size.height));
            std::uint8_t* out = destination.row(y);
            for (const std::size_t offset : offsets)
            {
                std::memcpy(out, in + offset, channels);
                out += channels;
            }
        }
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
} // namespace SushiAI::Vision::Imaging
