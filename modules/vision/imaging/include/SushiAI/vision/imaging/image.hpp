/**************************************************************************/
/* image.hpp                                                              */
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
 * @file image.hpp
 * @brief Declares the owning 8-bit image and its read-only view.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds the largest channel count an image may have. */
    inline constexpr std::int32_t K_MAX_CHANNELS = 4;

    /** @brief Holds the channel count of an RGB image, the form every detection sample has. */
    inline constexpr std::int32_t K_RGB_CHANNELS = 3;

    /** @brief Holds the largest height or width an image may have, in pixels. */
    inline constexpr std::int32_t K_MAX_IMAGE_SIDE = 16384;

    /** @brief Reports whether a height or a width is in [1, K_MAX_IMAGE_SIDE]. */
    [[nodiscard]] constexpr bool is_valid_side(std::int64_t side) noexcept
    {
        return side >= 1 && side <= K_MAX_IMAGE_SIDE;
    }

    /** @brief Reports whether a channel count is in [1, K_MAX_CHANNELS]. */
    [[nodiscard]] constexpr bool is_valid_channel_count(std::int64_t channels) noexcept
    {
        return channels >= 1 && channels <= K_MAX_CHANNELS;
    }

    /** @brief Holds the height and width of an image in pixels, height first. */
    struct ImageSize
    {
        /** @brief Counts the rows. */
        std::int32_t height = 0;

        /** @brief Counts the columns. */
        std::int32_t width = 0;

        /** @brief Reports whether two sizes have the same height and the same width. */
        [[nodiscard]] friend constexpr bool operator==(ImageSize a, ImageSize b) noexcept
        {
            return a.height == b.height && a.width == b.width;
        }

        /** @brief Reports whether two sizes differ in height or in width. */
        [[nodiscard]] friend constexpr bool operator!=(ImageSize a, ImageSize b) noexcept
        {
            return !(a == b);
        }
    };

    /** @brief Reports whether both sides of a size are in [1, K_MAX_IMAGE_SIDE]. */
    [[nodiscard]] constexpr bool is_valid_size(ImageSize size) noexcept
    {
        return is_valid_side(size.height) && is_valid_side(size.width);
    }

    /**
     * @brief Refers to 8-bit pixels it does not own, without the right to change them.
     *
     * The layout is row-major with channels innermost (HWC): the byte of row y, column x and
     * channel c is at offset (y * width + x) * channels + c. A three-channel image is RGB,
     * red at channel 0.
     */
    class ImageView final
    {
        public:
            /** @brief Constructs a view of no pixels. */
            constexpr ImageView() noexcept = default;

            /**
             * @brief Constructs a view over pixels another object owns.
             * @param data size.height * size.width * channels readable bytes that outlive the view.
             * @param size Both sides in [1, K_MAX_IMAGE_SIDE].
             * @param channels In [1, K_MAX_CHANNELS].
             */
            constexpr ImageView(const std::uint8_t* data, ImageSize size,
                                std::int32_t channels) noexcept
                : data_(data), size_(size), channels_(channels)
            {

            }

            /** @brief Returns the first byte of the first row, or null for an empty view. */
            [[nodiscard]] constexpr const std::uint8_t* data() const noexcept { return data_; }

            /** @brief Returns the height and width in pixels. */
            [[nodiscard]] constexpr ImageSize size() const noexcept { return size_; }

            /** @brief Returns the row count. */
            [[nodiscard]] constexpr std::int32_t height() const noexcept { return size_.height; }

            /** @brief Returns the column count. */
            [[nodiscard]] constexpr std::int32_t width() const noexcept { return size_.width; }

            /** @brief Returns the channel count. */
            [[nodiscard]] constexpr std::int32_t channels() const noexcept { return channels_; }

            /** @brief Returns the byte count of one row, width times channels. */
            [[nodiscard]] constexpr std::size_t row_bytes() const noexcept
            {
                return static_cast<std::size_t>(size_.width) * static_cast<std::size_t>(channels_);
            }

            /** @brief Returns the byte count of the whole image. */
            [[nodiscard]] constexpr std::size_t byte_count() const noexcept
            {
                return static_cast<std::size_t>(size_.height) * row_bytes();
            }

            /**
             * @brief Returns the first byte of one row.
             * @param y Row index in [0, height()); not checked.
             */
            [[nodiscard]] constexpr const std::uint8_t* row(std::int32_t y) const noexcept
            {
                return data_ + static_cast<std::size_t>(y) * row_bytes();
            }

        private:
            /// @brief Points at the first byte, or is null for an empty view.
            const std::uint8_t* data_ = nullptr;

            /// @brief Holds the height and width.
            ImageSize size_{};

            /// @brief Counts the channels.
            std::int32_t channels_ = 0;
    };

    /**
     * @brief Owns the pixels of one 8-bit image in the layout ImageView describes.
     *
     * A three-channel image is RGB everywhere in the module. The module never converts to
     * float: the caller that assembles a batch divides by 255 and stacks the images into the
     * model's NHWC tensor.
     */
    class Image final
    {
        public:
            /** @brief Constructs an image of no pixels. */
            Image() noexcept = default;

            /**
             * @brief Constructs an image with every byte zero.
             * @pre @p size passes is_valid_size and @p channels passes is_valid_channel_count; a
             *      size read from a file or a caller is tested with them first.
             */
            Image(ImageSize size, std::int32_t channels);

            /**
             * @brief Gives the image a new size and channel count, keeping its allocation when
             *        the allocation is large enough.
             *
             * The pixel values afterwards are unspecified, and every view made before the
             * call is invalid.
             * @pre The bounds of the constructor hold.
             */
            void reset(ImageSize size, std::int32_t channels);

            /** @brief Sets every byte of the image to one value. */
            void fill(std::uint8_t value) noexcept;

            /** @brief Returns the first byte of the first row; unspecified for an empty image. */
            [[nodiscard]] std::uint8_t* data() noexcept { return pixels_.data(); }

            /** @brief Returns the first byte of the first row; unspecified for an empty image. */
            [[nodiscard]] const std::uint8_t* data() const noexcept { return pixels_.data(); }

            /** @brief Returns the byte count of the allocation, at least byte_count(). */
            [[nodiscard]] std::size_t allocated_bytes() const noexcept { return pixels_.size(); }

            /** @brief Returns the height and width in pixels. */
            [[nodiscard]] ImageSize size() const noexcept { return size_; }

            /** @brief Returns the row count. */
            [[nodiscard]] std::int32_t height() const noexcept { return size_.height; }

            /** @brief Returns the column count. */
            [[nodiscard]] std::int32_t width() const noexcept { return size_.width; }

            /** @brief Returns the channel count. */
            [[nodiscard]] std::int32_t channels() const noexcept { return channels_; }

            /** @brief Returns the byte count of one row, width times channels. */
            [[nodiscard]] std::size_t row_bytes() const noexcept { return view().row_bytes(); }

            /** @brief Returns the byte count of the whole image. */
            [[nodiscard]] std::size_t byte_count() const noexcept { return view().byte_count(); }

            /**
             * @brief Returns the first byte of one row.
             * @param y Row index in [0, height()); not checked.
             */
            [[nodiscard]] std::uint8_t* row(std::int32_t y) noexcept
            {
                return pixels_.data() + static_cast<std::size_t>(y) * row_bytes();
            }

            /**
             * @brief Returns the first byte of one row.
             * @param y Row index in [0, height()); not checked.
             */
            [[nodiscard]] const std::uint8_t* row(std::int32_t y) const noexcept
            {
                return view().row(y);
            }

            /** @brief Returns a view of the pixels, valid until the next reset(). */
            [[nodiscard]] ImageView view() const noexcept
            {
                return ImageView(pixels_.data(), size_, channels_);
            }

        private:
            /// @brief Holds at least height * width * channels bytes; the image is the prefix.
            std::vector<std::uint8_t> pixels_;

            /// @brief Holds the height and width.
            ImageSize size_{};

            /// @brief Counts the channels.
            std::int32_t channels_ = 0;
    };

    /**
     * @brief Reports whether a view reads any byte of the allocation an image owns.
     *
     * The whole allocation counts, not only the bytes of the current size, because a reset
     * may write any of them. A view with no pixels or with extents outside the bounds of
     * Image overlaps nothing, and so does an image that has allocated nothing.
     */
    [[nodiscard]] bool overlaps(ImageView view, const Image& image) noexcept;

    /**
     * @brief Stops the process unless a view can be read while a destination image is written.
     *
     * In this order: the view has pixels, its size passes is_valid_size, its channel count
     * passes is_valid_channel_count, and overlaps(source, destination) is false. This is
     * the one aliasing rule of every function that takes a source and a destination.
     * @param operation Names the calling function at the start of the message.
     */
    void check_source(ImageView source, const Image& destination, std::string_view operation);
} // namespace SushiAI::Vision::Imaging
