/**************************************************************************/
/* mask_png.cpp                                                           */
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
 * @file mask_png.cpp
 * @brief Implements the reader of a dataset kept as an image folder and a mask folder.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/mask_png.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/image_extensions.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_mask_png";

        /** @brief Holds the extension of a mask file. */
        constexpr std::string_view K_MASK_EXTENSION = ".png";

        /** @brief Returns the files below a folder that a predicate keeps, sorted by path. */
        template <typename Predicate>
        [[nodiscard]] Result<std::vector<std::filesystem::path>> files_of(
            const std::filesystem::path& folder, Predicate is_kept)
        {
            SA_TRY_ASSIGN(std::vector<std::filesystem::path> files,
                          Detail::list_entries(folder, Detail::EntryKind::File,
                                               Detail::ListDepth::Tree, K_READER));

            files.erase(std::remove_if(files.begin(), files.end(),
                                       [&](const std::filesystem::path& file)
            {
                return !is_kept(file);
            }), files.end());

            return Result<std::vector<std::filesystem::path>>(std::move(files));
        }

        /** @brief Returns the folder of a split below a folder; the folder for no split. */
        [[nodiscard]] std::filesystem::path split_folder(const std::filesystem::path& folder,
                                                         const std::string& split)
        {
            return split.empty() ? folder : folder / split;
        }

        /** @brief Reports whether a text ends in another one; every text ends in none. */
        [[nodiscard]] bool ends_with(std::string_view text, std::string_view suffix) noexcept
        {
            return text.size() >= suffix.size() &&
                   text.substr(text.size() - suffix.size()) == suffix;
        }

        /** @brief Reports whether a path ends in a suffix and ".png", in lower case. */
        [[nodiscard]] bool is_mask_file(const std::filesystem::path& file,
                                        std::string_view suffix)
        {
            return file.extension() == std::filesystem::path(K_MASK_EXTENSION) &&
                   ends_with(file.stem().u8string(), suffix);
        }

        /**
         * @brief Returns the mask an image is paired with.
         * @return The path below @p mask_folder that the image has below @p image_folder,
         *         its stem ending in the mask suffix in place of the image suffix;
         *         InvalidFile for an image whose stem lacks the image suffix.
         */
        [[nodiscard]] Result<std::filesystem::path> mask_of(
            const std::filesystem::path& image, const std::filesystem::path& image_folder,
            const std::filesystem::path& mask_folder, const MaskPNGOptions& options)
        {
            std::string stem = image.stem().u8string();
            SA_FAIL_IF(!ends_with(stem, options.image_suffix), StatusCode::InvalidFile,
                       "'{}': {}: the name of the image does not end in '{}'",
                       Detail::display_of(image), K_READER, options.image_suffix);

            stem.resize(stem.size() - options.image_suffix.size());
            stem.append(options.mask_suffix).append(K_MASK_EXTENSION);

            return mask_folder / image.lexically_relative(image_folder).parent_path() /
                   std::filesystem::u8path(stem);
        }

        /** @brief Lists one split; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const MaskPNGOptions& options)
        {
            SA_FAIL_IF(options.class_count < 1, StatusCode::InvalidDescription,
                       "{}: the class count is {}; it must be at least 1", K_READER,
                       options.class_count);

            const std::filesystem::path image_folder =
                split_folder(options.root / options.image_folder, options.split);
            const std::filesystem::path mask_folder =
                split_folder(options.root / options.mask_folder, options.split);

            SA_TRY_ASSIGN(std::vector<std::filesystem::path> images,
                          files_of(image_folder, has_image_extension));
            SA_TRY_ASSIGN(std::vector<std::filesystem::path> masks,
                          files_of(mask_folder, [&](const std::filesystem::path& file)
            {
                return is_mask_file(file, options.mask_suffix);
            }));
            SA_FAIL_IF(images.empty(), StatusCode::InvalidFile,
                       "'{}': {}: the folder holds no image", Detail::display_of(image_folder),
                       K_READER);

            SampleList list;
            list.class_count = options.class_count;
            list.samples.reserve(images.size());

            // masks is sorted; is_paired[m] says that an image already took masks[m].
            std::vector<bool> is_paired(masks.size(), false);

            for (std::size_t image = 0; image < images.size(); ++image)
            {
                SA_TRY_ASSIGN(const std::filesystem::path mask,
                              mask_of(images[image], image_folder, mask_folder, options));

                const auto found = std::lower_bound(masks.begin(), masks.end(), mask);
                SA_FAIL_IF(found == masks.end() || *found != mask, StatusCode::FileNotFound,
                           "'{}': {}: the image has no mask '{}'",
                           Detail::display_of(images[image]), K_READER, Detail::display_of(mask));

                const std::size_t paired = static_cast<std::size_t>(found - masks.begin());
                SA_FAIL_IF(is_paired[paired], StatusCode::InvalidFile,
                           "'{}': {}: the image shares the mask '{}' with an image of the same "
                           "stem",
                           Detail::display_of(images[image]), K_READER, Detail::display_of(mask));
                is_paired[paired] = true;

                Sample sample;
                sample.file = static_cast<std::uint32_t>(image);
                sample.mask = static_cast<std::uint32_t>(images.size() + paired);
                list.samples.push_back(std::move(sample));
            }

            const auto unpaired = std::find(is_paired.begin(), is_paired.end(), false);
            SA_FAIL_IF(unpaired != is_paired.end(), StatusCode::InvalidFile,
                       "'{}': {}: the mask has no image of its stem below '{}'",
                       Detail::display_of(masks[static_cast<std::size_t>(
                           unpaired - is_paired.begin())]),
                       K_READER, Detail::display_of(image_folder));

            list.files = std::move(images);
            list.files.insert(list.files.end(), std::make_move_iterator(masks.begin()),
                              std::make_move_iterator(masks.end()));
            return Result<SampleList>(std::move(list));
        }
    } // namespace

    Result<SampleList> read_mask_png(const MaskPNGOptions& options)
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(options);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
