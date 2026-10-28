/**************************************************************************/
/* yolo_txt.hpp                                                           */
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
 * @file yolo_txt.hpp
 * @brief Declares the reader of a dataset kept as images and one label text file per image.
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
    /** @brief Holds what read_yolo_txt reads. */
    struct YOLOTxtOptions
    {
        /** @brief Holds the directory of the dataset. */
        std::filesystem::path root;

        /** @brief Names the image list root/<split>.txt, or the folder root/images/<split>. */
        std::string split = "train";

        /** @brief Counts the classes; every class of a label file must be below it. */
        std::int32_t class_count = 0;
    };

    /**
     * @brief Returns the label file of an image.
     *
     * The last folder named "images" on the path becomes "labels" and the extension becomes
     * .txt. A path with no such folder keeps its folder.
     */
    [[nodiscard]] std::filesystem::path yolo_label_path(const std::filesystem::path& image) noexcept;

    /**
     * @brief Lists the images of one split and parses the label file of each.
     *
     * An image without a label file has no object. Imaging::parse_label_text parses a label
     * file. README.md, section yolo_txt.hpp, gives the two forms of a split and every status.
     * @return The list; a failure that names the label file and its line, or the image list
     *         and its line.
     */
    [[nodiscard]] Result<SampleList> read_yolo_txt(const YOLOTxtOptions& options) noexcept;
} // namespace SushiAI::Vision::Datasets
