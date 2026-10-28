/**************************************************************************/
/* image_folder.hpp                                                       */
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
 * @file image_folder.hpp
 * @brief Declares the reader of a dataset kept as one folder of images per class.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what read_image_folder reads. */
    struct ImageFolderOptions
    {
        /** @brief Holds the directory of the dataset. */
        std::filesystem::path root;

        /** @brief Names the folder of root that holds the class folders; empty for root. */
        std::string split;
    };

    /**
     * @brief Lists the images of every class folder of one split.
     *
     * The classes are the folders of the split in sorted order. A sample is a file with an
     * image extension anywhere below its class folder; a class lists its samples by path.
     * @return The list; FileNotFound when the split is not a directory; InvalidFile when it
     *         holds no folder or no image, or cannot be listed; OutOfMemory.
     */
    [[nodiscard]] Result<SampleList> read_image_folder(const ImageFolderOptions& options) noexcept;
} // namespace SushiAI::Vision::Datasets
