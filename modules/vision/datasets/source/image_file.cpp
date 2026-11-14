/**************************************************************************/
/* image_file.cpp                                                         */
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
 * @file image_file.cpp
 * @brief Implements the reader of a dataset that is one image file.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/image_file.hpp>

#include <string_view>
#include <utility>

#include <SushiAI/vision/datasets/image_extensions.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_image_file";

        /** @brief Lists the one image; may throw what an allocation or a path throws. */
        [[nodiscard]] Result<SampleList> read_checked(const ImageFileOptions& options,
                                                      const ByteSource& source)
        {
            SA_FAIL_IF(options.class_count < 1, StatusCode::InvalidDescription,
                       "{}: the class count is {}; it must be at least 1", K_READER,
                       options.class_count);

            const std::filesystem::path path = options.root / options.file;
            SA_FAIL_IF(!has_image_extension(path), StatusCode::InvalidDescription,
                       "'{}': {}: the file has no image extension", Detail::display_of(path),
                       K_READER);

            const PathKind kind = source.kind_of(path);
            SA_FAIL_IF(kind == PathKind::Missing, StatusCode::FileNotFound,
                       "'{}': {}: the file does not exist", Detail::display_of(path), K_READER);
            SA_FAIL_IF(kind != PathKind::File, StatusCode::InvalidFile,
                       "'{}': {}: the path is no regular file", Detail::display_of(path),
                       K_READER);

            SampleList list;
            list.class_count = options.class_count;
            list.files.push_back(path);
            list.samples.emplace_back();
            return Result<SampleList>(std::move(list));
        }
    } // namespace

    Result<SampleList> read_image_file(const ImageFileOptions& options) noexcept
    {
        return read_image_file(options, Detail::FileSystemSource());
    }

    Result<SampleList> read_image_file(const ImageFileOptions& options,
                                       const ByteSource& source) noexcept
    {
        return Detail::at_boundary(K_READER, [&]()
        {
            return read_checked(options, source);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
