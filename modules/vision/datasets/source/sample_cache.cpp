/**************************************************************************/
/* sample_cache.cpp                                                       */
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
 * @file sample_cache.cpp
 * @brief Implements the raw cache of a sample list: its file names, its writer and its use.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/sample_cache.hpp>

#include <algorithm>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <SushiLog/text/text_format.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/hash/fnv1a.hpp>
#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_CACHE = "sample cache";

        /** @brief Holds what a cache file is called while it is being written. */
        constexpr std::string_view K_PARTIAL_EXTENSION = ".partial";

        /** @brief Holds the digits of a hash in a file name. */
        constexpr std::string_view K_HEX_DIGITS = "0123456789abcdef";

        /** @brief Returns the 64-bit FNV-1a hash of a text as sixteen hexadecimal digits. */
        [[nodiscard]] std::string hash_of(std::string_view text)
        {
            std::uint64_t hash = Hash::fnv1a_text(Hash::K_FNV1A_OFFSET_BASIS, text);

            std::string digits(16, '0');
            for (std::size_t i = digits.size(); i-- > 0; hash >>= 4)
                digits[i] = K_HEX_DIGITS[hash & 0xF];

            return digits;
        }

        /**
         * @brief Tells, per file of a list, whether it is an image the cache holds.
         * @return True for a file a sample names as its image and no sample reads stored
         *         pixels from; InvalidDescription for a sample that names no file of the list.
         */
        [[nodiscard]] Result<std::vector<bool>> find_cached_files(const SampleList& list)
        {
            std::vector<bool> is_image(list.files.size(), false);
            std::vector<bool> is_stored(list.files.size(), false);

            for (std::size_t sample = 0; sample < list.samples.size(); ++sample)
            {
                const Sample& record = list.samples[sample];
                SA_FAIL_IF(record.file >= list.files.size(), StatusCode::InvalidDescription,
                           "{}: sample {} names file {} of a list of {} files", K_CACHE, sample,
                           record.file, list.files.size());

                if (record.stored.has_value())
                    is_stored[record.file] = true;
                else
                    is_image[record.file] = true;
            }

            for (std::size_t file = 0; file < is_image.size(); ++file)
                is_image[file] = is_image[file] && !is_stored[file];

            return Result<std::vector<bool>>(std::move(is_image));
        }

        /** @brief Reports whether a sample of a list has a mask. */
        [[nodiscard]] bool has_masks(const SampleList& list) noexcept
        {
            return std::any_of(list.samples.begin(), list.samples.end(),
                               [](const Sample& sample) { return sample.mask.has_value(); });
        }

        /** @brief Writes one image beside its cache file, then gives it the file's name. */
        [[nodiscard]] Status write_entry(const std::filesystem::path& target,
                                         Imaging::ImageView image)
        {
            SA_FAIL_IF(image.channels() != Imaging::K_GRAY_CHANNELS &&
                           image.channels() != Imaging::K_RGB_CHANNELS,
                       StatusCode::Unsupported,
                       "'{}': {}: an image of {} channels has no raw image file",
                       Detail::display_of(target), K_CACHE, image.channels());

            std::filesystem::path partial = target;
            partial += std::string(K_PARTIAL_EXTENSION);

            // A file under the final name is whole: the rename is the last step.
            SA_TRY(Imaging::write_raw_image_file(partial, image,
                                                 image.channels() == Imaging::K_GRAY_CHANNELS
                                                     ? Imaging::ChannelOrder::Gray
                                                     : Imaging::ChannelOrder::RGB));
            std::filesystem::rename(partial, target);
            return Status();
        }

        /** @brief Writes the missing cache files; may throw what a path or a stream throws. */
        [[nodiscard]] Result<SampleCacheReport> write_checked(const SampleList& list,
                                                              const SampleCacheOptions& options)
        {
            SA_TRY_ASSIGN(const std::vector<bool> is_cached, find_cached_files(list));
            SA_FAIL_IF(options.longest_side > 0 && has_masks(list),
                       StatusCode::InvalidDescription,
                       "{}: a longest side of {} would leave the masks of the list at another "
                       "size than their images; cache a list with masks at its stored size",
                       K_CACHE, options.longest_side);

            std::filesystem::create_directories(options.directory);

            Codecs::DecodeOptions decoding;
            decoding.layout = options.layout;

            SampleCacheReport report;
            Imaging::Image resized;

            for (std::size_t file = 0; file < list.files.size(); ++file)
            {
                if (!is_cached[file])
                    continue;

                SA_TRY_ASSIGN(const std::filesystem::path target,
                              sample_cache_path(list.files[file], options));
                if (std::filesystem::exists(target))
                {
                    ++report.reused;
                    continue;
                }

                SA_TRY_ASSIGN(const Imaging::Image decoded,
                              Codecs::load_image(list.files[file], decoding));

                Imaging::ImageView stored = decoded.view();
                if (options.longest_side > 0 &&
                    std::max(decoded.height(), decoded.width()) > options.longest_side)
                {
                    SA_TRY(Imaging::resize_bilinear(
                        decoded.view(),
                        Imaging::fit_long_side(decoded.size(), options.longest_side), resized));
                    stored = resized.view();
                }

                SA_TRY(write_entry(target, stored));
                ++report.written;
                report.bytes_written += Imaging::K_RAW_IMAGE_HEADER_BYTES + stored.byte_count();
            }

            return report;
        }
    } // namespace

    Result<std::filesystem::path> sample_cache_path(const std::filesystem::path& source,
                                                    const SampleCacheOptions& options)
    {
        return Detail::at_boundary(K_CACHE, [&]() -> Result<std::filesystem::path>
        {
            SA_TRY_ASSIGN(const std::uint64_t bytes, Detail::size_of_file(source, K_CACHE));

            std::error_code error;
            const std::filesystem::file_time_type modified =
                std::filesystem::last_write_time(source, error);
            SA_FAIL_IF(static_cast<bool>(error), StatusCode::InvalidFile,
                       "'{}': {}: the modification time cannot be read: {}",
                       Detail::display_of(source), K_CACHE, error.message());

            const std::string key = SushiLog::format_text(
                "{}|{}|{}|{}|{}", source.generic_u8string(), bytes,
                static_cast<std::int64_t>(modified.time_since_epoch().count()),
                options.longest_side, static_cast<std::int32_t>(options.layout));

            std::filesystem::path name = source.stem();
            name += "-" + hash_of(key);
            name += std::string(Imaging::K_RAW_IMAGE_EXTENSION);
            return options.directory / name;
        }, source);
    }

    Result<SampleCacheReport> write_sample_cache(const SampleList& list,
                                                 const SampleCacheOptions& options)
    {
        SA_FAIL_IF(options.directory.empty(), StatusCode::InvalidDescription,
                   "{}: the cache has no directory", K_CACHE);
        SA_FAIL_IF(options.longest_side < 0 || options.longest_side > Imaging::K_MAX_IMAGE_SIDE,
                   StatusCode::InvalidDescription,
                   "{}: a longest side of {} is outside [0, {}]; 0 keeps the stored size",
                   K_CACHE, options.longest_side, Imaging::K_MAX_IMAGE_SIDE);

        return Detail::at_boundary(K_CACHE, [&]() -> Result<SampleCacheReport>
        {
            return write_checked(list, options);
        }, options.directory);
    }

    Result<SampleList> use_sample_cache(SampleList list, const SampleCacheOptions& options)
    {
        return Detail::at_boundary(K_CACHE, [&]() -> Result<SampleList>
        {
            SA_TRY_ASSIGN(const std::vector<bool> is_cached, find_cached_files(list));

            for (std::size_t file = 0; file < list.files.size(); ++file)
            {
                if (!is_cached[file])
                    continue;

                SA_TRY_ASSIGN(std::filesystem::path target,
                              sample_cache_path(list.files[file], options));
                SA_FAIL_IF(!std::filesystem::exists(target), StatusCode::FileNotFound,
                           "'{}': {}: '{}' holds no file '{}' for the image; it was not "
                           "cached, or it changed since",
                           Detail::display_of(list.files[file]), K_CACHE,
                           Detail::display_of(options.directory),
                           Detail::display_of(target.filename()));

                list.files[file] = std::move(target);
            }

            return Result<SampleList>(std::move(list));
        }, options.directory);
    }
} // namespace SushiAI::Vision::Datasets
