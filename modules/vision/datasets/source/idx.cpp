/**************************************************************************/
/* idx.cpp                                                                */
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
 * @file idx.cpp
 * @brief Implements the reader of a dataset kept as an IDX image file and an IDX label file.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/idx.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiAI/vision/datasets/idx_file.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_idx";

        /** @brief Holds the rank of an image file without a channel dimension. */
        constexpr std::size_t K_GRAY_RANK = 3;

        /** @brief Holds the rank of an image file whose last dimension counts channels. */
        constexpr std::size_t K_CHANNEL_RANK = 4;

        /** @brief Holds the file names one split has in the MNIST distribution. */
        struct StandardFiles
        {
            /** @brief Names the split. */
            std::string_view split;

            /** @brief Names the image file. */
            std::string_view images;

            /** @brief Names the label file. */
            std::string_view labels;
        };

        /** @brief Lists the file names of the splits of the MNIST distribution. */
        constexpr StandardFiles K_STANDARD_FILES[] = {
            {"train", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"},
            {"test", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"},
        };

        /** @brief Holds the paths of the two files of one split. */
        struct FilePair
        {
            /** @brief Holds the path of the image file. */
            std::filesystem::path images;

            /** @brief Holds the path of the label file. */
            std::filesystem::path labels;
        };

        /**
         * @brief Returns the two paths the options name.
         * @return The paths; InvalidDescription when a file name is empty and the split has
         *         no row in K_STANDARD_FILES.
         */
        [[nodiscard]] Result<FilePair> resolve_files(const IDXOptions& options)
        {
            std::string_view images = options.image_file;
            std::string_view labels = options.label_file;

            for (const StandardFiles& standard : K_STANDARD_FILES)
            {
                if (options.split != standard.split)
                    continue;

                if (images.empty())
                    images = standard.images;
                if (labels.empty())
                    labels = standard.labels;
            }

            SA_FAIL_IF(images.empty() || labels.empty(), StatusCode::InvalidDescription,
                       "{}: the split '{}' has no standard file name; name 'train' or 'test', "
                       "or set image_file and label_file",
                       K_READER, options.split);

            return Result<FilePair>(FilePair{options.root / images, options.root / labels});
        }

        /**
         * @brief Returns the place of the first image of an image file.
         * @return The pixels of image 0; InvalidFile for another element type or rank;
         *         Unsupported for a side or a channel count outside the bounds of an image.
         */
        [[nodiscard]] Result<StoredPixels> first_image(const std::filesystem::path& file,
                                                       const IDXHeader& header)
        {
            const std::size_t rank = header.extents.size();
            SA_FAIL_IF(header.element != IDXElement::UInt8, StatusCode::InvalidFile,
                       "'{}': {}: the image file holds elements of code {}; images are unsigned "
                       "bytes, code {}",
                       Detail::display_of(file), K_READER, static_cast<int>(header.element),
                       static_cast<int>(IDXElement::UInt8));
            SA_FAIL_IF(rank != K_GRAY_RANK && rank != K_CHANNEL_RANK, StatusCode::InvalidFile,
                       "'{}': {}: the image file has {} dimensions; it must have {} or {}",
                       Detail::display_of(file), K_READER, rank, K_GRAY_RANK, K_CHANNEL_RANK);

            const std::uint32_t height = header.extents[1];
            const std::uint32_t width = header.extents[2];
            const std::uint32_t channels = rank == K_CHANNEL_RANK ? header.extents[3] : 1u;

            SA_FAIL_IF(!Imaging::is_valid_side(height) || !Imaging::is_valid_side(width) ||
                           !Imaging::is_valid_channel_count(channels),
                       StatusCode::Unsupported,
                       "'{}': {}: {} by {} pixels of {} channels are outside the bounds of an "
                       "image",
                       Detail::display_of(file), K_READER, height, width, channels);

            StoredPixels pixels;
            pixels.offset = header.header_bytes();
            pixels.size = Imaging::ImageSize{static_cast<std::int32_t>(height),
                                             static_cast<std::int32_t>(width)};
            pixels.channels = static_cast<std::int32_t>(channels);
            pixels.layout = PlaneLayout::Interleaved;
            return pixels;
        }

        /** @brief Lists one split; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const IDXOptions& options)
        {
            SA_TRY_ASSIGN(FilePair files, resolve_files(options));
            SA_TRY_ASSIGN(const IDXHeader image_header, read_idx_header(files.images));
            SA_TRY_ASSIGN(const IDXHeader label_header, read_idx_header(files.labels));
            SA_TRY_ASSIGN(StoredPixels pixels, first_image(files.images, image_header));

            SA_FAIL_IF(label_header.element != IDXElement::UInt8 ||
                           label_header.extents.size() != 1,
                       StatusCode::InvalidFile,
                       "'{}': {}: the label file has {} dimensions of element code {}; it must "
                       "hold one dimension of unsigned bytes",
                       Detail::display_of(files.labels), K_READER, label_header.extents.size(),
                       static_cast<int>(label_header.element));

            const std::uint32_t count = image_header.extents[0];
            SA_FAIL_IF(count != label_header.extents[0], StatusCode::InvalidFile,
                       "'{}': {}: the file holds {} images and '{}' holds {} labels",
                       Detail::display_of(files.images), K_READER, count,
                       Detail::display_of(files.labels), label_header.extents[0]);
            SA_FAIL_IF(count == 0, StatusCode::InvalidFile, "'{}': {}: the file holds no image",
                       Detail::display_of(files.images), K_READER);

            SA_TRY_ASSIGN(const std::vector<std::uint8_t> labels,
                          read_idx_payload(files.labels, label_header));

            SampleList list;
            list.files.push_back(std::move(files.images));
            list.samples.reserve(count);

            const std::uint64_t image_bytes = pixels.byte_count();
            std::int32_t largest = 0;

            for (std::uint32_t image = 0; image < count; ++image)
            {
                const std::int32_t label = labels[image];
                largest = std::max(largest, label);

                Sample sample;
                sample.stored = pixels;
                sample.class_index = label;
                list.samples.push_back(std::move(sample));

                pixels.offset += image_bytes;
            }

            list.class_count = largest + 1;
            return Result<SampleList>(std::move(list));
        }
    } // namespace

    Result<SampleList> read_idx(const IDXOptions& options)
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(options);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
