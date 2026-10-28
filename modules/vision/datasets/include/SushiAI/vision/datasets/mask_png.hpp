/**************************************************************************/
/* mask_png.hpp                                                           */
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
 * @file mask_png.hpp
 * @brief Declares the reader of a dataset kept as an image folder and a mask folder.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what read_mask_png reads. */
    struct MaskPNGOptions
    {
        /** @brief Holds the directory of the dataset. */
        std::filesystem::path root;

        /** @brief Names the folder of each of the two folders that is read; empty for none. */
        std::string split = "train";

        /** @brief Names the folder of root that holds the images. */
        std::filesystem::path image_folder = "images";

        /** @brief Names the folder of root that holds one PNG mask per image. */
        std::filesystem::path mask_folder = "masks";

        /** @brief Counts the classes a mask may name; the reader opens no mask to check it. */
        std::int32_t class_count = 0;

        /** @brief Holds what the stem of every image ends in; cut before the mask is named. */
        std::string image_suffix;

        /** @brief Holds what the stem of every mask ends in; other files are not masks. */
        std::string mask_suffix;
    };

    /**
     * @brief Lists the images of one split and pairs each with the mask of its stem.
     *
     * The mask of root/<image_folder>/<split>/a/b<image_suffix>.jpg is
     * root/<mask_folder>/<split>/a/b<mask_suffix>.png. No file is opened.
     * @return The list, or a failure README.md, section mask_png.hpp, lists.
     */
    [[nodiscard]] Result<SampleList> read_mask_png(const MaskPNGOptions& options) noexcept;
} // namespace SushiAI::Vision::Datasets
