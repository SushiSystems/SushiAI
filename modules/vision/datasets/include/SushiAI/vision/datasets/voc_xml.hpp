/**************************************************************************/
/* voc_xml.hpp                                                            */
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
 * @file voc_xml.hpp
 * @brief Declares the reader of a dataset kept as images and one XML annotation per image.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names what the reader does with an object its annotation marks as difficult. */
    enum class VOCDifficult : std::uint8_t
    {
        Dropped, /**< The object is left out of the sample. */
        Kept,    /**< The object is read as any other. */
    };

    /** @brief Holds what read_voc_xml reads. */
    struct VOCXMLOptions
    {
        /** @brief Holds the directory with Annotations, JPEGImages and ImageSets. */
        std::filesystem::path root;

        /** @brief Names the list ImageSets/Main/<split>.txt; empty for every annotation. */
        std::string split = "train";

        /** @brief Names class i at entry i; empty for the names of the split, sorted. */
        std::vector<std::string> class_names;

        /** @brief Selects what becomes of an object marked as difficult. */
        VOCDifficult difficult = VOCDifficult::Dropped;
    };

    /**
     * @brief Lists the images of one split and parses the annotation of each with parse_xml.
     *
     * README.md, section voc_xml.hpp, gives the layout, the box conversion and every status.
     * @return The list; a failure that names the annotation and its line, or the image list
     *         and its line.
     */
    [[nodiscard]] Result<SampleList> read_voc_xml(const VOCXMLOptions& options);
} // namespace SushiAI::Vision::Datasets
