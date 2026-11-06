/**************************************************************************/
/* coco_json.hpp                                                          */
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
 * @file coco_json.hpp
 * @brief Declares the reader of a dataset kept as one annotation JSON and an image folder.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names what the reader does with the "segmentation" of an annotation. */
    enum class COCOSegmentation : std::uint8_t
    {
        Ignored,  /**< The member is not read; an object has a box and no polygon. */
        Polygons, /**< A polygon is kept as the object's outline; a run-length mask fails. */
    };

    /** @brief Holds what read_coco_json reads. */
    struct COCOJSONOptions
    {
        /** @brief Holds the directory of the dataset. */
        std::filesystem::path root;

        /** @brief Names the split in the default paths of the two members below. */
        std::string split = "train";

        /** @brief Names the JSON file; empty for annotations/instances_<split>.json. */
        std::filesystem::path annotation_file;

        /** @brief Names the folder "file_name" is taken from; empty for <split>. */
        std::filesystem::path image_folder;

        /** @brief Selects whether polygons are kept. */
        COCOSegmentation segmentation = COCOSegmentation::Ignored;
    };

    /**
     * @brief Parses one annotation file and lists its images with their objects.
     *
     * A class is the rank of its category id among the ids, in rising order. An annotation
     * with "iscrowd" set is left out. README.md, section coco_json.hpp, has every status.
     * @return The list; a failure that names the file and the index of the entry at fault.
     */
    [[nodiscard]] Result<SampleList> read_coco_json(const COCOJSONOptions& options) noexcept;

    /**
     * @brief Reads the annotation file and checks the image folder of a COCO set served by
     *        @p source.
     *
     * read_coco_json(options) calls this with the file system; a MemorySource serves files
     * that were never written.
     * @return What the overload above returns.
     */
    [[nodiscard]] Result<SampleList> read_coco_json(const COCOJSONOptions& options,
                                                    const ByteSource& source) noexcept;
} // namespace SushiAI::Vision::Datasets
