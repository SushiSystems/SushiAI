/**************************************************************************/
/* image.cpp                                                              */
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
 * @file image.cpp
 * @brief Implements the owning 8-bit image, the overlap predicate and the source check.
 * @author Mustafa Garip
 */

#include <SushiAI/imaging/image.hpp>

#include <algorithm>
#include <functional>

#include <SushiAI/core/common.hpp>

namespace SushiAI::Imaging
{
    namespace
    {
        /**
         * @brief Returns the byte count of an image after checking its extents.
         *
         * Throws Error when the size fails is_valid_size or the channel count fails
         * is_valid_channel_count.
         * @param operation Names the calling function at the start of the message.
         */
        [[nodiscard]] std::size_t checked_byte_count(ImageSize size, std::int32_t channels,
                                                     std::string_view operation)
        {
            SA_THROW_IF(!is_valid_size(size), "{}: a size of {}x{} is outside [1, {}]", operation,
                        size.height, size.width, K_MAX_IMAGE_SIDE);
            SA_THROW_IF(!is_valid_channel_count(channels),
                        "{}: a channel count of {} is outside [1, {}]", operation, channels,
                        K_MAX_CHANNELS);

            return ImageView(nullptr, size, channels).byte_count();
        }
    } // namespace

    Image::Image(ImageSize size, std::int32_t channels)
        : pixels_(checked_byte_count(size, channels, "Image"), std::uint8_t{0}), size_(size),
          channels_(channels)
    {

    }

    void Image::reset(ImageSize size, std::int32_t channels)
    {
        const std::size_t needed = checked_byte_count(size, channels, "Image::reset");

        // pixels_ never shrinks, so data() stays put while the allocation is large enough.
        if (pixels_.size() < needed)
            pixels_.resize(needed);

        size_ = size;
        channels_ = channels;
    }

    void Image::fill(std::uint8_t value) noexcept
    {
        std::fill_n(pixels_.data(), byte_count(), value);
    }

    bool overlaps(ImageView view, const Image& image) noexcept
    {
        const bool is_readable = view.data() != nullptr && is_valid_size(view.size()) &&
                                 is_valid_channel_count(view.channels());

        if (!is_readable || image.allocated_bytes() == 0)
            return false;

        const std::less<const std::uint8_t*> before;
        const std::uint8_t* view_end = view.data() + view.byte_count();
        const std::uint8_t* image_end = image.data() + image.allocated_bytes();

        return before(view.data(), image_end) && before(image.data(), view_end);
    }

    void check_source(ImageView source, const Image& destination, std::string_view operation)
    {
        SA_THROW_IF(source.data() == nullptr, "{}: the source image has no pixels", operation);
        SA_THROW_IF(!is_valid_size(source.size()),
                    "{}: a source size of {}x{} is outside [1, {}]", operation, source.height(),
                    source.width(), K_MAX_IMAGE_SIDE);
        SA_THROW_IF(!is_valid_channel_count(source.channels()),
                    "{}: a source channel count of {} is outside [1, {}]", operation,
                    source.channels(), K_MAX_CHANNELS);
        SA_THROW_IF(overlaps(source, destination),
                    "{}: the destination holds the source's pixels", operation);
    }
} // namespace SushiAI::Imaging
