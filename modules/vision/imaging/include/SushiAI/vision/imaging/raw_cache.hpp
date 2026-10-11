/**************************************************************************/
/* raw_cache.hpp                                                          */
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
 * @file raw_cache.hpp
 * @brief Declares the raw image file a decode tool writes once, with its reader and writer.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string_view>

#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds the first four bytes of a raw image file, "SAIM", as a little-endian word. */
    inline constexpr std::uint32_t K_RAW_IMAGE_MAGIC = 0x4D494153u;

    /** @brief Holds the format version this build writes, and the newest it reads. */
    inline constexpr std::uint32_t K_RAW_IMAGE_VERSION = 1u;

    /** @brief Holds the byte count of the header that precedes the pixels. */
    inline constexpr std::size_t K_RAW_IMAGE_HEADER_BYTES = 24;

    /** @brief Holds the file name extension of a raw image file. */
    inline constexpr std::string_view K_RAW_IMAGE_EXTENSION = ".saim";

    /** @brief Names the order of the channels stored in a raw image file. */
    enum class ChannelOrder : std::uint8_t
    {
        Gray = 0, /**< One channel of luminance; expand_gray makes it RGB for detection. */
        RGB = 1,  /**< Three channels, red first; the order of the module. */
        BGR = 2,  /**< Three channels, blue first. */
    };

    /** @brief Holds the validated fields of a raw image header. */
    struct RawImageHeader
    {
        /** @brief Holds the height and width, each in [1, K_MAX_IMAGE_SIDE]. */
        ImageSize size{};

        /** @brief Counts the channels: 1 for Gray, 3 for RGB and BGR. */
        std::int32_t channels = 0;

        /** @brief Names the order of the stored channels. */
        ChannelOrder order = ChannelOrder::RGB;

        /** @brief Returns the byte count of the pixels that follow the header. */
        [[nodiscard]] constexpr std::size_t pixel_bytes() const noexcept
        {
            return static_cast<std::size_t>(size.height) * static_cast<std::size_t>(size.width) *
                   static_cast<std::size_t>(channels);
        }
    };

    /**
     * @brief Writes the header and the pixels of one image, field by field, little-endian.
     *
     * Throws Error when the view is empty, when its channel count does not fit the order,
     * when a side exceeds K_MAX_IMAGE_SIDE or when the stream fails.
     * @param order The order the view's channels are in; the bytes are written as they are.
     */
    void write_raw_image(std::ostream& out, ImageView image, ChannelOrder order);

    /**
     * @brief Reads and validates a header, allocating nothing.
     *
     * Throws Error naming the cause on a short read, a wrong magic, a version other than
     * the ones this build reads (the message names both versions), a side outside
     * [1, K_MAX_IMAGE_SIDE], an unknown order, or a channel count that does not fit it.
     */
    [[nodiscard]] RawImageHeader read_raw_image_header(std::istream& in);

    /**
     * @brief Reads the pixels that follow a header into an RGB or single-channel image.
     *
     * A BGR file has channels 0 and 2 swapped as it is read. Throws Error when the stream
     * ends before header.pixel_bytes() bytes.
     * @param header The value read_raw_image_header returned for this stream.
     * @param destination Reset to the header's size and channels.
     */
    void read_raw_image_pixels(std::istream& in, const RawImageHeader& header,
                               Image& destination);

    /**
     * @brief Writes one image to a file, replacing the file when it exists.
     *
     * Throws Error on what write_raw_image rejects and when the file cannot be opened.
     */
    void write_raw_image_file(const std::filesystem::path& path, ImageView image,
                              ChannelOrder order);

    /**
     * @brief Reads one image file into an RGB or single-channel image.
     *
     * Throws Error when the file cannot be opened, on what read_raw_image_header rejects,
     * and when the file size is not K_RAW_IMAGE_HEADER_BYTES + pixel_bytes(). The size is
     * checked before destination is reset.
     */
    void read_raw_image_file(const std::filesystem::path& path, Image& destination);
} // namespace SushiAI::Vision::Imaging
