/**************************************************************************/
/* sample.hpp                                                             */
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
 * @file sample.hpp
 * @brief Declares the sample record and the sample list every format reader returns.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/datasets/value_table.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Holds what one sample's annotation gave and where its pixels are.
     *
     * A sample names no task. A layout that labels the whole image sets class_index; a
     * layout that labels objects fills objects; one that labels pixels sets mask; a reader
     * sets what its layout holds. A sample of a tabular list has its row in SampleList::table.
     */
    struct Sample
    {
        /** @brief Holds the index, in SampleList::files, of the file the pixels are in. */
        std::uint32_t file = 0;

        /** @brief Locates raw pixels inside the file; absent when the file is one image. */
        std::optional<StoredPixels> stored;

        /** @brief Holds the class of the whole image, in [0, SampleList::class_count). */
        std::optional<std::int32_t> class_index;

        /** @brief Holds a class, a box and a polygon per object, in fractions of the image. */
        Imaging::FileLabels objects;

        /** @brief Holds the index, in SampleList::files, of the file of one class per pixel. */
        std::optional<std::uint32_t> mask;
    };

    /** @brief Holds the samples of one root and split, in the order their reader lists them. */
    struct SampleList
    {
        /** @brief Holds every file a sample's pixels or its mask are in, once each. */
        std::vector<std::filesystem::path> files;

        /** @brief Holds one record per sample. */
        std::vector<Sample> samples;

        /** @brief Holds one name per class, or nothing when the layout names no class. */
        std::vector<std::string> class_names;

        /** @brief Counts the classes a sample's class index or object class may name. */
        std::int32_t class_count = 0;

        /** @brief Holds row i for sample i of a tabular list; no row for a list of images. */
        ValueTable table;
    };
} // namespace SushiAI::Vision::Datasets
