/**************************************************************************/
/* decode_image.cpp                                                       */
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
 * @file decode_image.cpp
 * @brief Implements the dispatch from an image file's leading bytes to its decoder.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/codecs/decode_image.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <istream>
#include <limits>
#include <new>
#include <streambuf>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/codecs/image_format.hpp>
#include <SushiAI/vision/codecs/jpeg_decoder.hpp>
#include <SushiAI/vision/codecs/png_decoder.hpp>

namespace SushiAI::Vision::Codecs
{
    namespace
    {
        /** @brief Names the signature every decoder of the module has. */
        using DecodeFunction = Result<Imaging::Image> (*)(SushiRuntime::span<const std::uint8_t>,
                                                          const DecodeOptions&);

        /** @brief Holds a format and the function that decodes it. */
        struct Decoder
        {
            /** @brief Names the format detect_image_format reports. */
            ImageFormat format;

            /** @brief Points at the decoder of that format. */
            DecodeFunction decode;
        };

        /** @brief Serves bytes another object owns to a stream, copying none of them. */
        class ByteStreamBuffer final : public std::streambuf
        {
            public:
                /** @brief Constructs a read-only buffer over bytes that must outlive it. */
                explicit ByteStreamBuffer(SushiRuntime::span<const std::uint8_t> bytes)
                {
                    // The get area is only read: std::streambuf has no const form of setg.
                    char* first = const_cast<char*>(reinterpret_cast<const char*>(bytes.data()));
                    setg(first, first, first + bytes.size());
                }
        };

        /** @brief Decodes a raw image file; may throw what the reader of vision/imaging throws. */
        [[nodiscard]] Result<Imaging::Image> decode_raw_checked(
            SushiRuntime::span<const std::uint8_t> bytes, const DecodeOptions& options)
        {
            ByteStreamBuffer buffer(bytes);
            std::istream in(&buffer);
            const Imaging::RawImageHeader header = Imaging::read_raw_image_header(in);

            const std::size_t declared = Imaging::K_RAW_IMAGE_HEADER_BYTES + header.pixel_bytes();
            SA_FAIL_IF(bytes.size() != declared, StatusCode::InvalidFile,
                       "decode_image: the raw image file holds {} bytes; its header declares {}",
                       bytes.size(), declared);

            const bool is_gray = header.channels == Imaging::K_GRAY_CHANNELS;
            SA_FAIL_IF(options.layout == ChannelLayout::Gray && !is_gray, StatusCode::Unsupported,
                       "decode_image: a raw image file of {} channels does not convert to gray",
                       header.channels);

            Imaging::Image stored;
            Imaging::read_raw_image_pixels(in, header, stored);
            if (options.layout != ChannelLayout::Color || !is_gray)
                return Result<Imaging::Image>(std::move(stored));

            Imaging::Image expanded;
            Imaging::expand_gray(stored.view(), expanded);
            return Result<Imaging::Image>(std::move(expanded));
        }

        /** @brief Decodes a raw image file through the reader of vision/imaging. */
        [[nodiscard]] Result<Imaging::Image> decode_raw(
            SushiRuntime::span<const std::uint8_t> bytes, const DecodeOptions& options)
        {
            try
            {
                return decode_raw_checked(bytes, options);
            }
            catch (const std::bad_alloc&)
            {
                return Status::from_message(StatusCode::OutOfMemory, []()
                {
                    return std::string("decode_image: the image's pixels could not be allocated");
                });
            }
            catch (const std::exception& error)
            {
                return Status::from_message(StatusCode::InvalidFile, [&]()
                {
                    return SushiLog::format_text("decode_image: {}", error.what());
                });
            }
        }

        /** @brief Holds one decoder per format; a new format adds a row. */
        constexpr std::array<Decoder, 3> K_DECODERS = {{
            {ImageFormat::JPEG, decode_jpeg},
            {ImageFormat::PNG, decode_png},
            {ImageFormat::Raw, decode_raw},
        }};

        /** @brief Returns the text of a path with forward slashes, for a message. */
        [[nodiscard]] std::string display_of(const std::filesystem::path& path)
        {
            return path.generic_string();
        }

        /**
         * @brief Reads every byte of a file.
         * @return The bytes; FileNotFound when the path names nothing; InvalidFile when it
         *         names something other than a regular file or the read stops early.
         */
        [[nodiscard]] Result<std::vector<std::uint8_t>> read_file(const std::filesystem::path& path)
        {
            std::error_code error;
            const std::filesystem::file_status status = std::filesystem::status(path, error);
            SA_FAIL_IF(!std::filesystem::exists(status), StatusCode::FileNotFound,
                       "'{}': load_image: no such file", display_of(path));
            SA_FAIL_IF(!std::filesystem::is_regular_file(status), StatusCode::InvalidFile,
                       "'{}': load_image: the path does not name a regular file",
                       display_of(path));

            const std::uintmax_t file_bytes = std::filesystem::file_size(path, error);
            SA_FAIL_IF(error, StatusCode::InvalidFile,
                       "'{}': load_image: the file's size cannot be read: {}", display_of(path),
                       error.message());
            SA_FAIL_IF(file_bytes > static_cast<std::uintmax_t>(
                                        std::numeric_limits<std::streamsize>::max()),
                       StatusCode::Unsupported,
                       "'{}': load_image: {} bytes are more than one read holds",
                       display_of(path), file_bytes);

            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                       "'{}': load_image: the file cannot be opened for reading",
                       display_of(path));

            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_bytes));
            in.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
            SA_FAIL_IF(in.gcount() != static_cast<std::streamsize>(bytes.size()),
                       StatusCode::InvalidFile,
                       "'{}': load_image: the read ended after {} of {} bytes", display_of(path),
                       in.gcount(), bytes.size());

            return Result<std::vector<std::uint8_t>>(std::move(bytes));
        }

        /** @brief Reads and decodes a file; may throw what an allocation or a stream throws. */
        [[nodiscard]] Result<Imaging::Image> load_checked(const std::filesystem::path& path,
                                                          const DecodeOptions& options)
        {
            SA_TRY_ASSIGN(const std::vector<std::uint8_t> bytes, read_file(path));

            Result<Imaging::Image> decoded = decode_image(bytes, options);
            if (decoded.has_value())
                return decoded;

            return SA_ANNOTATE(decoded.status(), "'{}'", display_of(path));
        }
    } // namespace

    Result<Imaging::Image> decode_image(SushiRuntime::span<const std::uint8_t> bytes,
                                        const DecodeOptions& options)
    {
        const ImageFormat format = detect_image_format(bytes);
        for (const Decoder& decoder : K_DECODERS)
        {
            if (decoder.format == format)
                return decoder.decode(bytes, options);
        }

        return Status::from_message(StatusCode::Unsupported, [&]()
        {
            return SushiLog::format_text(
                "decode_image: the first of {} bytes open no image format this build decodes",
                bytes.size());
        });
    }

    Result<Imaging::Image> load_image(const std::filesystem::path& path,
                                      const DecodeOptions& options)
    {
        try
        {
            return load_checked(path, options);
        }
        catch (const std::bad_alloc&)
        {
            return Status::from_message(StatusCode::OutOfMemory, [&]()
            {
                return SushiLog::format_text(
                    "'{}': load_image: the file's bytes could not be allocated", display_of(path));
            });
        }
        catch (const std::exception& error)
        {
            return Status::from_message(StatusCode::InvalidFile, [&]()
            {
                return SushiLog::format_text("'{}': load_image: {}", display_of(path),
                                               error.what());
            });
        }
    }
} // namespace SushiAI::Vision::Codecs
