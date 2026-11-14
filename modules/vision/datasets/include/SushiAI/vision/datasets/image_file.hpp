/**************************************************************************/
/* image_file.hpp                                                         */
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
 * @file image_file.hpp
 * @brief Declares the reader of a dataset that is one image file.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what read_image_file reads. */
    struct ImageFileOptions
    {
        /** @brief Holds the directory of the image. */
        std::filesystem::path root;

        /** @brief Names the image file, below root. */
        std::filesystem::path file;

        /** @brief Counts the classes of the model the image is read for; at least 1. */
        std::int32_t class_count = 0;
    };

    /**
     * @brief Lists the one image of root/file as a sample with no object.
     * @return The list; InvalidDescription for a class count below 1 or a file with no image
     *         extension; FileNotFound when the file does not exist; InvalidFile when it is no
     *         regular file; OutOfMemory.
     */
    [[nodiscard]] Result<SampleList> read_image_file(const ImageFileOptions& options) noexcept;

    /**
     * @brief Lists the one image of root/file, found through @p source.
     *
     * read_image_file(options) calls this with the file system; a MemorySource serves files
     * that were never written.
     * @return What the overload above returns.
     */
    [[nodiscard]] Result<SampleList> read_image_file(const ImageFileOptions& options,
                                                     const ByteSource& source) noexcept;
} // namespace SushiAI::Vision::Datasets
