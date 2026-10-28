/**************************************************************************/
/* stored_pixels.hpp                                                      */
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
 * @file stored_pixels.hpp
 * @brief Declares the place of one sample's raw pixels inside a file, and their reader.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the order of the bytes of one stored image. */
    enum class PlaneLayout : std::uint8_t
    {
        Interleaved, /**< Row-major with channels innermost, the layout of Imaging::Image. */
        Planar,      /**< One whole row-major plane per channel, channel 0 first. */
    };

    /** @brief Locates the uncompressed 8-bit pixels of one sample inside a file. */
    struct StoredPixels
    {
        /** @brief Holds the offset of the first pixel byte from the start of the file. */
        std::uint64_t offset = 0;

        /** @brief Holds the height and width, each in [1, Imaging::K_MAX_IMAGE_SIDE]. */
        Imaging::ImageSize size{};

        /** @brief Counts the channels, in [1, Imaging::K_MAX_CHANNELS]. */
        std::int32_t channels = 0;

        /** @brief Names the order the bytes are stored in. */
        PlaneLayout layout = PlaneLayout::Interleaved;

        /** @brief Returns the byte count of the stored image. */
        [[nodiscard]] constexpr std::size_t byte_count() const noexcept
        {
            return static_cast<std::size_t>(size.height) * static_cast<std::size_t>(size.width) *
                   static_cast<std::size_t>(channels);
        }
    };

    /**
     * @brief Copies one stored image out of a stream into an interleaved image.
     * @param in A seekable binary stream over the file the pixels are in.
     * @param destination Reset to the stored size and channel count.
     * @return Success; InvalidDescription for a size or a channel count outside the bounds of
     *         Imaging::Image; InvalidFile when the stream ends early; OutOfMemory.
     */
    [[nodiscard]] Status read_stored_pixels(std::istream& in, const StoredPixels& pixels,
                                            Imaging::Image& destination) noexcept;

    /**
     * @brief Opens a file and copies one stored image out of it into an interleaved image.
     * @return What the stream form returns, and InvalidFile when the file cannot be opened.
     *         Every message opens with the path.
     */
    [[nodiscard]] Status read_stored_pixels(const std::filesystem::path& file,
                                            const StoredPixels& pixels,
                                            Imaging::Image& destination) noexcept;
} // namespace SushiAI::Vision::Datasets
