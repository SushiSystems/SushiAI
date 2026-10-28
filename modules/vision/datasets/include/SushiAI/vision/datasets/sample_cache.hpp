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
     * @brief Returns the cache file of one image file.
     *
     * The name holds a hash of the source path, its byte count, its modification time and
     * the options, so a source that changed has another cache file. README.md, section
     * sample_cache.hpp, states the mapping.
     * @return The path; FileNotFound or InvalidFile when @p source cannot be examined.
     */
    [[nodiscard]] Result<std::filesystem::path> sample_cache_path(
        const std::filesystem::path& source, const SampleCacheOptions& options) noexcept;

    /**
     * @brief Decodes every image file of a list that has no cache file, and writes one.
     *
     * A file a sample reads stored pixels from, and a mask, are left alone.
     * @return The counts; InvalidDescription for a negative longest side, or a longest side
     *         over a list with masks; the failure of a decode or of a write.
     */
    [[nodiscard]] Result<SampleCacheReport> write_sample_cache(
        const SampleList& list, const SampleCacheOptions& options) noexcept;

    /**
     * @brief Returns the list with every image file replaced by its cache file.
     * @return The list; FileNotFound, naming the image, when its cache file is absent, as
     *         it is after the image changed.
     */
    [[nodiscard]] Result<SampleList> use_sample_cache(SampleList list,
                                                      const SampleCacheOptions& options);
} // namespace SushiAI::Vision::Datasets
