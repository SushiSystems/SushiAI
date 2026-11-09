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
 * @file sample_cache.cpp
 * @brief Implements the raw cache of a sample list: its file names, its writer and its use.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/sample_cache.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiLog/text/text_format.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/hash/fnv1a.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

#include "decode_file.hpp"
#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_CACHE = "sample cache";

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

        /** @brief Writes one image to its cache file through a sink. */
        [[nodiscard]] Status write_entry(const std::filesystem::path& target,
                                         Imaging::ImageView image, const ByteSink& sink,
                                         std::uint64_t& bytes)
        {
            SA_FAIL_IF(image.channels() != Imaging::K_GRAY_CHANNELS &&
                           image.channels() != Imaging::K_RGB_CHANNELS,
                       StatusCode::Unsupported,
                       "'{}': {}: an image of {} channels has no raw image file",
                       Detail::display_of(target), K_CACHE, image.channels());

            std::ostringstream encoded(std::ios::binary);
            SA_TRY(Imaging::write_raw_image(encoded, image,
                                            image.channels() == Imaging::K_GRAY_CHANNELS
                                                ? Imaging::ChannelOrder::Gray
                                                : Imaging::ChannelOrder::RGB));

            const std::string content = encoded.str();
            SA_TRY(sink.write(target, content, K_CACHE));
            bytes = content.size();
            return Status();
        }

        /** @brief Returns the source a call reads through: @p files, else the file system. */
        [[nodiscard]] const ByteSource& source_of(const ByteSource* files) noexcept
        {
            return files != nullptr ? *files : Detail::file_system();
        }

        /** @brief Returns the sink a call writes through: @p sink, else the file system. */
        [[nodiscard]] const ByteSink& sink_of(const ByteSink* sink) noexcept
        {
            return sink != nullptr ? *sink : Detail::file_sink();
        }

        /** @brief Writes the missing cache files; may throw what a path or a stream throws. */
        [[nodiscard]] Result<SampleCacheReport> write_checked(const SampleList& list,
                                                              const SampleCacheOptions& options,
                                                              const ByteSource& files,
                                                              const ByteSink& sink)
        {
            SA_TRY_ASSIGN(const std::vector<bool> is_cached, find_cached_files(list));
            SA_FAIL_IF(options.longest_side > 0 && has_masks(list),
                       StatusCode::InvalidDescription,
                       "{}: a longest side of {} would leave the masks of the list at another "
                       "size than their images; cache a list with masks at its stored size",
                       K_CACHE, options.longest_side);

            Codecs::DecodeOptions decoding;
            decoding.layout = options.layout;

            SampleCacheReport report;
            Imaging::Image resized;

            for (std::size_t file = 0; file < list.files.size(); ++file)
            {
                if (!is_cached[file])
                    continue;

                SA_TRY_ASSIGN(const std::filesystem::path target,
                              sample_cache_path(list.files[file], options, &files));
                if (files.kind_of(target) != PathKind::Missing)
                {
                    ++report.reused;
                    continue;
                }

                SA_TRY_ASSIGN(const Imaging::Image decoded,
                              Detail::decode_file(files, list.files[file], decoding, K_CACHE));

                Imaging::ImageView stored = decoded.view();
                if (options.longest_side > 0 &&
                    std::max(decoded.height(), decoded.width()) > options.longest_side)
                {
                    SA_TRY(Imaging::resize_bilinear(
                        decoded.view(),
                        Imaging::fit_long_side(decoded.size(), options.longest_side), resized));
                    stored = resized.view();
                }

                // The sink makes the directory.
                std::uint64_t bytes = 0;
                SA_TRY(write_entry(target, stored, sink, bytes));
                ++report.written;
                report.bytes_written += bytes;
            }

            return report;
        }
    } // namespace

    Result<std::filesystem::path> sample_cache_path(const std::filesystem::path& source,
                                                    const SampleCacheOptions& options,
                                                    const ByteSource* files) noexcept
    {
        return Detail::at_boundary(K_CACHE, [&]() -> Result<std::filesystem::path>
        {
            const ByteSource& reading = source_of(files);
            SA_TRY_ASSIGN(const std::uint64_t bytes, reading.size_of(source, K_CACHE));
            SA_TRY_ASSIGN(const std::int64_t modified, reading.modified_of(source, K_CACHE));

            const std::string key = SushiLog::format_text(
                "{}|{}|{}|{}|{}", source.generic_u8string(), bytes, modified,
                options.longest_side, static_cast<std::int32_t>(options.layout));

            std::filesystem::path name = source.stem();
            name += "-" + hash_of(key);
            name += std::string(Imaging::K_RAW_IMAGE_EXTENSION);
            return options.directory / name;
        }, source);
    }

    Result<SampleCacheReport> write_sample_cache(const SampleList& list,
                                                 const SampleCacheOptions& options,
                                                 DatasetStorage storage) noexcept
    {
        SA_FAIL_IF(options.directory.empty(), StatusCode::InvalidDescription,
                   "{}: the cache has no directory", K_CACHE);
        SA_FAIL_IF(options.longest_side < 0 || options.longest_side > Imaging::K_MAX_IMAGE_SIDE,
                   StatusCode::InvalidDescription,
                   "{}: a longest side of {} is outside [0, {}]; 0 keeps the stored size",
                   K_CACHE, options.longest_side, Imaging::K_MAX_IMAGE_SIDE);

        return Detail::at_boundary(K_CACHE, [&]() -> Result<SampleCacheReport>
        {
            return write_checked(list, options, source_of(storage.source),
                                 sink_of(storage.sink));
        }, options.directory);
    }

    Result<SampleList> use_sample_cache(SampleList list, const SampleCacheOptions& options,
                                        const ByteSource* files)
    {
        return Detail::at_boundary(K_CACHE, [&]() -> Result<SampleList>
        {
            SA_TRY_ASSIGN(const std::vector<bool> is_cached, find_cached_files(list));

            for (std::size_t file = 0; file < list.files.size(); ++file)
            {
                if (!is_cached[file])
                    continue;

                SA_TRY_ASSIGN(std::filesystem::path target,
                              sample_cache_path(list.files[file], options, files));
                SA_FAIL_IF(source_of(files).kind_of(target) == PathKind::Missing,
                           StatusCode::FileNotFound,
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
