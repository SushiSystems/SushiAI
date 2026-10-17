/**************************************************************************/
/* stored_pixels.cpp                                                      */
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
 * @file stored_pixels.cpp
 * @brief Implements the reader of one sample's raw pixels inside a file.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/stored_pixels.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <istream>
#include <limits>
#include <string_view>


#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_READER = "read_stored_pixels";

        /** @brief Holds the byte count read at once from a plane. */
        constexpr std::size_t K_PLANE_CHUNK_BYTES = 4096;

        /** @brief Reads an exact number of bytes, or fails with the count that arrived. */
        [[nodiscard]] Status read_bytes(std::istream& in, std::uint8_t* out, std::size_t count)
        {
            in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(count));
            SA_FAIL_IF(in.gcount() != static_cast<std::streamsize>(count), StatusCode::InvalidFile,
                       "{}: the stream ended after {} of {} bytes", K_READER, in.gcount(), count);
            return Status();
        }

        /** @brief Reads one plane per channel and writes each into its channel of an image. */
        [[nodiscard]] Status read_planes(std::istream& in, Imaging::Image& destination)
        {
            const std::size_t channels = static_cast<std::size_t>(destination.channels());
            const std::size_t plane_bytes = destination.byte_count() / channels;
            std::array<std::uint8_t, K_PLANE_CHUNK_BYTES> chunk;

            for (std::size_t channel = 0; channel < channels; ++channel)
            {
                for (std::size_t done = 0; done < plane_bytes;)
                {
                    const std::size_t count = std::min(chunk.size(), plane_bytes - done);
                    SA_TRY(read_bytes(in, chunk.data(), count));

                    std::uint8_t* out = destination.data() + done * channels + channel;
                    for (std::size_t i = 0; i < count; ++i)
                        out[i * channels] = chunk[i];

                    done += count;
                }
            }

            return Status();
        }

        /** @brief Copies one stored image; may throw what an allocation throws. */
        [[nodiscard]] Status read_checked(std::istream& in, const StoredPixels& pixels,
                                          Imaging::Image& destination)
        {
            SA_FAIL_IF(!Imaging::is_valid_size(pixels.size) ||
                           !Imaging::is_valid_channel_count(pixels.channels),
                       StatusCode::InvalidDescription,
                       "{}: {} by {} pixels of {} channels are outside the bounds of an image",
                       K_READER, pixels.size.height, pixels.size.width, pixels.channels);
            SA_FAIL_IF(pixels.offset >
                           static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),
                       StatusCode::InvalidDescription,
                       "{}: the offset {} is beyond what a stream can seek to", K_READER,
                       pixels.offset);

            destination.reset(pixels.size, pixels.channels);

            in.clear();
            in.seekg(static_cast<std::streamoff>(pixels.offset), std::ios::beg);
            SA_FAIL_IF(!in, StatusCode::InvalidFile, "{}: the stream cannot seek to byte {}",
                       K_READER, pixels.offset);

            if (pixels.layout == PlaneLayout::Planar)
                return read_planes(in, destination);

            return read_bytes(in, destination.data(), destination.byte_count());
        }
    } // namespace

    Status read_stored_pixels(std::istream& in, const StoredPixels& pixels,
                              Imaging::Image& destination)
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(in, pixels, destination);
        });
    }

    Status read_stored_pixels(const std::filesystem::path& file, const StoredPixels& pixels,
                              Imaging::Image& destination)
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            std::ifstream in(file, std::ios::binary);
            SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for reading",
                       Detail::display_of(file), K_READER);

            return SA_ANNOTATE(read_checked(in, pixels, destination), "'{}'",
                               Detail::display_of(file));
        }, file);
    }
} // namespace SushiAI::Vision::Datasets
