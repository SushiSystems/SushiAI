/**************************************************************************/
/* png_decoder.cpp                                                        */
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
 * @file png_decoder.cpp
 * @brief Implements the PNG decoder over the simplified read interface of libpng.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/codecs/png_decoder.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <png.h>

#include <SushiRuntime/core/span.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>

namespace SushiAI::Vision::Codecs
{
    namespace
    {
        /** @brief Holds the bytes of the largest colour map libpng writes: 256 RGBA entries. */
        constexpr std::size_t K_COLORMAP_BYTES = 256 * 4;

        /** @brief Owns a libpng image control and frees what the library hangs on it. */
        class ImageControl final
        {
            public:
                /** @brief Constructs a zeroed control of the version this build compiles with. */
                ImageControl() noexcept
                {
                    control_.version = PNG_IMAGE_VERSION;
                }

                /** @brief Frees the library's state, which a finished read has already freed. */
                ~ImageControl()
                {
                    png_image_free(&control_);
                }

                /** @brief Forbids a copy, which would free the library's state twice. */
                ImageControl(const ImageControl&) = delete;

                /** @brief Forbids copy assignment. */
                ImageControl& operator=(const ImageControl&) = delete;

                /** @brief Forbids a move, which would leave two owners of one state. */
                ImageControl(ImageControl&&) = delete;

                /** @brief Forbids move assignment. */
                ImageControl& operator=(ImageControl&&) = delete;

                /** @brief Returns the control the library reads and writes. */
                [[nodiscard]] png_image& get() noexcept { return control_; }

            private:
                /// @brief Holds the library's control structure.
                png_image control_{};
        };

        /** @brief Holds the format libpng is asked for and the image it becomes. */
        struct ReadPlan
        {
            /** @brief Holds the PNG_FORMAT_* value given to the library. */
            png_uint_32 format;

            /** @brief Counts the bytes per pixel the library writes. */
            std::int32_t read_channels;

            /** @brief Counts the channels of the decoded image. */
            std::int32_t channels;
        };

        /**
         * @brief Chooses the library format for a file's format and the caller's options.
         *
         * A file with alpha is read with its alpha channel, which the decoder then drops;
         * a format without one would make the library composite the colours on a background.
         * @param stored The format png_image_begin_read_from_memory reported.
         */
        [[nodiscard]] ReadPlan plan_read(png_uint_32 stored, const DecodeOptions& options) noexcept
        {
            const bool is_palette = (stored & PNG_FORMAT_FLAG_COLORMAP) != 0;
            if (is_palette && options.palette == PaletteRead::Indices)
                return ReadPlan{PNG_FORMAT_RGBA_COLORMAP, Imaging::K_GRAY_CHANNELS,
                                Imaging::K_GRAY_CHANNELS};

            const bool is_gray = (stored & PNG_FORMAT_FLAG_COLOR) == 0;
            const bool wants_gray = options.layout == ChannelLayout::Gray ||
                                    (options.layout == ChannelLayout::Stored && is_gray);
            const png_uint_32 base = wants_gray ? static_cast<png_uint_32>(PNG_FORMAT_GRAY)
                                                : static_cast<png_uint_32>(PNG_FORMAT_RGB);
            const std::int32_t channels =
                wants_gray ? Imaging::K_GRAY_CHANNELS : Imaging::K_RGB_CHANNELS;

            if ((stored & PNG_FORMAT_FLAG_ALPHA) != 0)
                return ReadPlan{base | PNG_FORMAT_FLAG_ALPHA, channels + 1, channels};

            return ReadPlan{base, channels, channels};
        }

        /** @brief Returns InvalidFile carrying the message the library left in a control. */
        [[nodiscard]] Status failure_of(const png_image& control) noexcept
        {
            return Status::from_message(StatusCode::InvalidFile, [&]()
            {
                return SushiLog::format_text("decode_png: {}",
                                               static_cast<const char*>(control.message));
            });
        }

        /**
         * @brief Copies pixels into an image, leaving out the last channel of each.
         * @param source destination.height() * destination.width() pixels of
         *               destination.channels() + 1 bytes each.
         */
        void drop_last_channel(const std::uint8_t* source, Imaging::Image& destination) noexcept
        {
            const std::size_t kept = static_cast<std::size_t>(destination.channels());
            const std::size_t pixels = destination.byte_count() / kept;
            std::uint8_t* target = destination.data();

            for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            {
                for (std::size_t channel = 0; channel < kept; ++channel)
                    target[channel] = source[channel];

                target += kept;
                source += kept + 1;
            }
        }

        /** @brief Decodes a PNG file; may throw what an allocation or Imaging::Image throws. */
        [[nodiscard]] Result<Imaging::Image> decode_checked(
            SushiRuntime::span<const std::uint8_t> bytes, const DecodeOptions& options)
        {
            ImageControl owner;
            png_image& control = owner.get();
            if (png_image_begin_read_from_memory(&control, bytes.data(), bytes.size()) == 0)
                return failure_of(control);

            SA_FAIL_IF(!Imaging::is_valid_side(control.height) ||
                           !Imaging::is_valid_side(control.width),
                       StatusCode::Unsupported,
                       "decode_png: the image is {}x{}; a side must be in [1, {}]", control.height,
                       control.width, Imaging::K_MAX_IMAGE_SIDE);

            const ReadPlan plan = plan_read(control.format, options);
            control.format = plan.format;
            control.flags |= PNG_IMAGE_FLAG_16BIT_sRGB;

            const Imaging::ImageSize size{static_cast<std::int32_t>(control.height),
                                          static_cast<std::int32_t>(control.width)};
            Imaging::Image image(size, plan.channels);
            const std::size_t read_bytes = static_cast<std::size_t>(size.height) *
                                           static_cast<std::size_t>(size.width) *
                                           static_cast<std::size_t>(plan.read_channels);
            SA_FAIL_IF(PNG_IMAGE_SIZE(control) != read_bytes, StatusCode::Unsupported,
                       "decode_png: libpng writes {} bytes where {} were planned",
                       PNG_IMAGE_SIZE(control), read_bytes);

            std::array<png_byte, K_COLORMAP_BYTES> colormap{};
            if (plan.read_channels == plan.channels)
            {
                if (png_image_finish_read(&control, nullptr, image.data(), 0, colormap.data()) == 0)
                    return failure_of(control);

                return Result<Imaging::Image>(std::move(image));
            }

            std::vector<std::uint8_t> with_alpha(read_bytes);
            if (png_image_finish_read(&control, nullptr, with_alpha.data(), 0,
                                      colormap.data()) == 0)
                return failure_of(control);

            drop_last_channel(with_alpha.data(), image);
            return Result<Imaging::Image>(std::move(image));
        }
    } // namespace

    Result<Imaging::Image> decode_png(SushiRuntime::span<const std::uint8_t> bytes,
                                      const DecodeOptions& options)
    {
        try
        {
            return decode_checked(bytes, options);
        }
        catch (const std::bad_alloc&)
        {
            return Status::from_message(StatusCode::OutOfMemory, []()
            {
                return std::string("decode_png: the image's pixels could not be allocated");
            });
        }
        catch (const std::exception& error)
        {
            return Status::from_message(StatusCode::InvalidFile, [&]()
            {
                return SushiLog::format_text("decode_png: {}", error.what());
            });
        }
    }
} // namespace SushiAI::Vision::Codecs
