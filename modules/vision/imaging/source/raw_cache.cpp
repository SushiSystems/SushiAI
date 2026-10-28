/**************************************************************************/
/* raw_cache.cpp                                                          */
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
 * @file raw_cache.cpp
 * @brief Implements the reader and the writer of the raw image file.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/raw_cache.hpp>

#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <ostream>
#include <system_error>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the byte count of one header field. */
        constexpr std::size_t K_FIELD_BYTES = 4;

        /** @brief Holds the channel count of a Gray file. */
        constexpr std::uint32_t K_GRAY_FILE_CHANNELS = 1u;

        /** @brief Holds the channel count of an RGB or a BGR file. */
        constexpr std::uint32_t K_COLOR_FILE_CHANNELS = 3u;

        /** @brief Holds the largest value a channel_order field may have. */
        constexpr std::uint32_t K_LAST_CHANNEL_ORDER =
            static_cast<std::uint32_t>(ChannelOrder::BGR);

        /** @brief Writes a value as four little-endian bytes at the given address. */
        void store_u32(char* bytes, std::uint32_t value) noexcept
        {
            for (std::size_t i = 0; i < K_FIELD_BYTES; ++i)
                bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
        }

        /** @brief Returns the value of four little-endian bytes at the given address. */
        [[nodiscard]] std::uint32_t load_u32(const char* bytes) noexcept
        {
            std::uint32_t value = 0;
            for (std::size_t i = 0; i < K_FIELD_BYTES; ++i)
            {
                const std::uint32_t byte = static_cast<unsigned char>(bytes[i]);
                value |= byte << (8 * i);
            }

            return value;
        }

        /**
         * @brief Checks that four header fields describe an image this format can hold.
         * The fields are compared as the unsigned words of the file, so no arithmetic runs on
         * a value before its range is known.
         * @param operation Names the calling function at the start of the message.
         * @param code The code of a refusal: InvalidFile for fields read from a file,
         *             InvalidDescription for the fields of an image about to be written.
         */
        Status check_layout(std::uint32_t height, std::uint32_t width, std::uint32_t channels,
                            std::uint32_t order, std::string_view operation, StatusCode code)
        {
            SA_FAIL_IF(!is_valid_side(height), code, "{}: the height {} is outside [1, {}]",
                       operation, height, K_MAX_IMAGE_SIDE);
            SA_FAIL_IF(!is_valid_side(width), code, "{}: the width {} is outside [1, {}]",
                       operation, width, K_MAX_IMAGE_SIDE);
            SA_FAIL_IF(order > K_LAST_CHANNEL_ORDER, code, "{}: the channel order {} is unknown",
                       operation, order);

            const bool is_gray = order == static_cast<std::uint32_t>(ChannelOrder::Gray);
            const std::uint32_t expected = is_gray ? K_GRAY_FILE_CHANNELS : K_COLOR_FILE_CHANNELS;

            SA_FAIL_IF(channels != expected, code,
                       "{}: the channel count {} does not fit channel order {}, which has {}",
                       operation, channels, order, expected);
            return Status();
        }

        /** @brief Checks that a header's fields describe an image this format can hold. */
        Status check_header(const RawImageHeader& header, std::string_view operation,
                            StatusCode code)
        {
            return check_layout(static_cast<std::uint32_t>(header.size.height),
                                static_cast<std::uint32_t>(header.size.width),
                                static_cast<std::uint32_t>(header.channels),
                                static_cast<std::uint32_t>(header.order), operation, code);
        }

        /**
         * @brief Checks that a view has pixels this format can hold in the given order.
         * @return InvalidDescription for a size, a channel count or an order the format cannot
         *         hold.
         * @pre The view has pixels.
         */
        Status check_image(ImageView image, ChannelOrder order, std::string_view operation)
        {
            SA_FATAL_IF(image.data() == nullptr, "{}: the image to write has no pixels",
                        operation);

            return check_header(RawImageHeader{image.size(), image.channels(), order}, operation,
                                StatusCode::InvalidDescription);
        }

        /** @brief Swaps channels 0 and 2 of every pixel of a three-channel image. */
        void swap_red_and_blue(Image& image) noexcept
        {
            std::uint8_t* const pixels = image.data();
            const std::size_t byte_count = image.byte_count();

            for (std::size_t i = 0; i < byte_count; i += K_COLOR_FILE_CHANNELS)
                std::swap(pixels[i], pixels[i + 2]);
        }
    } // namespace

    Status write_raw_image(std::ostream& out, ImageView image, ChannelOrder order) noexcept
    {
        SA_TRY(check_image(image, order, "write_raw_image"));

        char header[K_RAW_IMAGE_HEADER_BYTES];
        store_u32(header + 0 * K_FIELD_BYTES, K_RAW_IMAGE_MAGIC);
        store_u32(header + 1 * K_FIELD_BYTES, K_RAW_IMAGE_VERSION);
        store_u32(header + 2 * K_FIELD_BYTES, static_cast<std::uint32_t>(image.height()));
        store_u32(header + 3 * K_FIELD_BYTES, static_cast<std::uint32_t>(image.width()));
        store_u32(header + 4 * K_FIELD_BYTES, static_cast<std::uint32_t>(image.channels()));
        store_u32(header + 5 * K_FIELD_BYTES, static_cast<std::uint32_t>(order));

        out.write(header, static_cast<std::streamsize>(K_RAW_IMAGE_HEADER_BYTES));
        out.write(reinterpret_cast<const char*>(image.data()),
                  static_cast<std::streamsize>(image.byte_count()));

        SA_FAIL_IF(!out, StatusCode::InvalidFile,
                   "write_raw_image: the stream failed while {} bytes were written",
                   K_RAW_IMAGE_HEADER_BYTES + image.byte_count());
        return Status();
    }

    Result<RawImageHeader> read_raw_image_header(std::istream& in) noexcept
    {
        char bytes[K_RAW_IMAGE_HEADER_BYTES];
        in.read(bytes, static_cast<std::streamsize>(K_RAW_IMAGE_HEADER_BYTES));

        SA_FAIL_IF(in.gcount() != static_cast<std::streamsize>(K_RAW_IMAGE_HEADER_BYTES),
                   StatusCode::InvalidFile,
                   "read_raw_image_header: the header ended after {} of {} bytes", in.gcount(),
                   K_RAW_IMAGE_HEADER_BYTES);

        const std::uint32_t magic = load_u32(bytes + 0 * K_FIELD_BYTES);
        const std::uint32_t version = load_u32(bytes + 1 * K_FIELD_BYTES);
        const std::uint32_t height = load_u32(bytes + 2 * K_FIELD_BYTES);
        const std::uint32_t width = load_u32(bytes + 3 * K_FIELD_BYTES);
        const std::uint32_t channels = load_u32(bytes + 4 * K_FIELD_BYTES);
        const std::uint32_t order = load_u32(bytes + 5 * K_FIELD_BYTES);

        SA_FAIL_IF(magic != K_RAW_IMAGE_MAGIC, StatusCode::InvalidFile,
                   "read_raw_image_header: the magic word is {}, not {}; this is not a raw image "
                   "file",
                   magic, K_RAW_IMAGE_MAGIC);
        SA_FAIL_IF(version != K_RAW_IMAGE_VERSION, StatusCode::InvalidFile,
                   "read_raw_image_header: the file has format version {}; this build reads "
                   "version {}",
                   version, K_RAW_IMAGE_VERSION);
        SA_TRY(check_layout(height, width, channels, order, "read_raw_image_header",
                            StatusCode::InvalidFile));

        RawImageHeader header;
        header.size.height = static_cast<std::int32_t>(height);
        header.size.width = static_cast<std::int32_t>(width);
        header.channels = static_cast<std::int32_t>(channels);
        header.order = static_cast<ChannelOrder>(order);
        return Result<RawImageHeader>(header);
    }

    Status read_raw_image_pixels(std::istream& in, const RawImageHeader& header,
                                 Image& destination) noexcept
    {
        SA_TRY(check_header(header, "read_raw_image_pixels", StatusCode::InvalidFile));
        destination.reset(header.size, header.channels);

        const std::streamsize byte_count = static_cast<std::streamsize>(header.pixel_bytes());
        in.read(reinterpret_cast<char*>(destination.data()), byte_count);

        SA_FAIL_IF(in.gcount() != byte_count, StatusCode::InvalidFile,
                   "read_raw_image_pixels: the pixels ended after {} of {} bytes", in.gcount(),
                   byte_count);

        if (header.order == ChannelOrder::BGR)
            swap_red_and_blue(destination);

        return Status();
    }

    Status write_raw_image_file(const std::filesystem::path& path, ImageView image,
                                ChannelOrder order) noexcept
    {
        // The image is checked before the open, so a rejected image truncates no file.
        SA_TRY(check_image(image, order, "write_raw_image_file"));

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        SA_FAIL_IF(!out.is_open(), StatusCode::InvalidFile,
                   "write_raw_image_file: cannot open '{}' for writing", path.u8string());

        SA_TRY(write_raw_image(out, image, order));
        out.flush();
        SA_FAIL_IF(!out, StatusCode::InvalidFile, "write_raw_image_file: writing '{}' failed",
                   path.u8string());
        return Status();
    }

    Status read_raw_image_file(const std::filesystem::path& path, Image& destination) noexcept
    {
        std::ifstream in(path, std::ios::binary);
        SA_FAIL_IF(!in.is_open(), StatusCode::FileNotFound,
                   "read_raw_image_file: cannot open '{}' for reading", path.u8string());

        std::error_code status;
        const std::uintmax_t file_bytes = std::filesystem::file_size(path, status);
        SA_FAIL_IF(static_cast<bool>(status), StatusCode::InvalidFile,
                   "read_raw_image_file: cannot read the size of '{}': {}", path.u8string(),
                   status.message());

        SA_TRY_ASSIGN(const RawImageHeader header, read_raw_image_header(in));
        const std::uintmax_t expected_bytes = K_RAW_IMAGE_HEADER_BYTES + header.pixel_bytes();

        SA_FAIL_IF(file_bytes != expected_bytes, StatusCode::InvalidFile,
                   "read_raw_image_file: '{}' holds {} bytes; its header declares {}",
                   path.u8string(), file_bytes, expected_bytes);

        return read_raw_image_pixels(in, header, destination);
    }
} // namespace SushiAI::Vision::Imaging
