/**************************************************************************/
/* image_folder.cpp                                                       */
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
 * @file image_folder.cpp
 * @brief Implements the reader of a dataset kept as one folder of images per class.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/image_folder.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiAI/vision/datasets/image_extensions.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_image_folder";

        /** @brief Lists one split; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const std::filesystem::path& directory)
        {
            SA_TRY_ASSIGN(const std::vector<std::filesystem::path> folders,
                          Detail::list_entries(directory, Detail::EntryKind::Directory,
                                               Detail::ListDepth::Children, K_READER));
            SA_FAIL_IF(folders.empty(), StatusCode::InvalidFile,
                       "'{}': {}: the directory holds no class folder",
                       Detail::display_of(directory), K_READER);

            SampleList list;
            list.class_count = static_cast<std::int32_t>(folders.size());
            list.class_names.reserve(folders.size());

            for (std::size_t folder = 0; folder < folders.size(); ++folder)
            {
                list.class_names.push_back(folders[folder].filename().generic_string());

                SA_TRY_ASSIGN(std::vector<std::filesystem::path> files,
                              Detail::list_entries(folders[folder], Detail::EntryKind::File,
                                                   Detail::ListDepth::Tree, K_READER));

                for (std::filesystem::path& file : files)
                {
                    if (!has_image_extension(file))
                        continue;

                    Sample sample;
                    sample.file = static_cast<std::uint32_t>(list.files.size());
                    sample.class_index = static_cast<std::int32_t>(folder);
                    list.files.push_back(std::move(file));
                    list.samples.push_back(std::move(sample));
                }
            }

            SA_FAIL_IF(list.samples.empty(), StatusCode::InvalidFile,
                       "'{}': {}: the {} class folders hold no image file",
                       Detail::display_of(directory), K_READER, folders.size());

            return Result<SampleList>(std::move(list));
        }
    } // namespace

    Result<SampleList> read_image_folder(const ImageFolderOptions& options) noexcept
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(options.split.empty() ? options.root
                                                      : options.root / options.split);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
