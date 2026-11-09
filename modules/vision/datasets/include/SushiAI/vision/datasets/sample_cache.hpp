/**************************************************************************/
/* sample_cache.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file sample_cache.hpp
 * @brief Declares the raw cache of a sample list: its file names, its writer and its use.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what the caller chooses about a raw cache. */
    struct SampleCacheOptions
    {
        /** @brief Holds the directory the raw image files are written to and read from. */
        std::filesystem::path directory;

        /** @brief Holds the longest side a cached image may have; 0 keeps every stored size. */
        std::int32_t longest_side = 0;

        /** @brief Selects the channels an image is decoded to before it is cached. */
        Codecs::ChannelLayout layout = Codecs::ChannelLayout::Color;
    };

    /** @brief Holds what one pass of write_sample_cache did. */
    struct SampleCacheReport
    {
        std::size_t written = 0;         /**< Counts the images decoded and written. */
        std::size_t reused = 0;          /**< Counts the images whose cache file existed. */
        std::uint64_t bytes_written = 0; /**< Counts the bytes of the files written. */
    };

    /**
     * @brief Returns the cache file of one image file; README.md, section sample_cache.hpp,
     *        states the name.
     * @param files Serves the byte count and modification time of @p source; null is the
     *        file system. Must outlive the call.
     * @return The path; FileNotFound or InvalidFile when @p source cannot be examined.
     */
    [[nodiscard]] Result<std::filesystem::path> sample_cache_path(
        const std::filesystem::path& source, const SampleCacheOptions& options,
        const ByteSource* files = nullptr) noexcept;

    /**
     * @brief Decodes every image file of a list that has no cache file, and writes one.
     *        Stored pixels and masks are left alone; files are written to the file system.
     * @param files Serves the images and the cache files that exist; null is the file system.
     * @return The counts; InvalidDescription for a negative longest side, or a longest side
     *         over a list with masks; the failure of a decode or of a write.
     */
    [[nodiscard]] Result<SampleCacheReport> write_sample_cache(
        const SampleList& list, const SampleCacheOptions& options,
        const ByteSource* files = nullptr) noexcept;

    /**
     * @brief Returns the list with every image file replaced by its cache file.
     * @param files Serves the images' stamps and tells whether a cache file exists; null is
     *        the file system. Must outlive the call.
     * @return The list; FileNotFound, naming the image, when its cache file is absent, as
     *         it is after the image changed.
     */
    [[nodiscard]] Result<SampleList> use_sample_cache(SampleList list,
                                                      const SampleCacheOptions& options,
                                                      const ByteSource* files = nullptr);
} // namespace SushiAI::Vision::Datasets
