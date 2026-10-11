/**************************************************************************/
/* idx.hpp                                                                */
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
 * @file idx.hpp
 * @brief Declares the reader of a dataset kept as an IDX image file and an IDX label file.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what read_idx reads. */
    struct IDXOptions
    {
        /** @brief Holds the directory of the two files. */
        std::filesystem::path root;

        /** @brief Names the split, "train" or "test", when a file name below is empty. */
        std::string split = "train";

        /** @brief Names the image file in root; empty for the split's MNIST file name. */
        std::string image_file;

        /** @brief Names the label file in root; empty for the split's MNIST file name. */
        std::string label_file;
    };

    /**
     * @brief Lists the images of an IDX image file with the classes of its label file.
     *
     * Each sample refers to its pixels inside the image file, and the class count is one
     * more than the largest label. README.md, section idx.hpp, lists the forms of the two
     * files and every status.
     */
    [[nodiscard]] Result<SampleList> read_idx(const IDXOptions& options);
} // namespace SushiAI::Vision::Datasets
