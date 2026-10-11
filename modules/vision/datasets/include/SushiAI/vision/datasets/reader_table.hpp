/**************************************************************************/
/* reader_table.hpp                                                       */
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
 * @file reader_table.hpp
 * @brief Declares the table of the format readers a data section may name.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what a data section states that every format reader can take. */
    struct ReaderRequest
    {
        /** @brief Holds the directory of the dataset. */
        std::filesystem::path root;

        /** @brief Names the split, as the reader's own options take it. */
        std::string split = "train";

        /** @brief Counts the classes the section states; 0 when it states none. */
        std::int32_t class_count = 0;

        /** @brief Holds the section a row reads its own keys from; null when there is none. */
        Config::Value section;

        /** @brief Holds what a diagnostic calls the section. */
        std::string context = "data";
    };

    /**
     * @brief Lists the samples of one root and split in one on-disk layout.
     * @return The list, or the failure of the reader behind the row.
     * @throws Error If a key of the row holds a value of another kind.
     */
    using SampleReader = Result<SampleList> (*)(const ReaderRequest& request);

    /** @brief Holds one row of the reader table. */
    struct ReaderEntry
    {
        /** @brief Names the reader, as a data section spells it. */
        std::string_view name;

        /** @brief Points at what lists the samples; never null. */
        SampleReader read;

        /** @brief Lists the keys the row reads of ReaderRequest::section. */
        SushiRuntime::span<const std::string_view> keys;
    };

    /**
     * @brief Returns the rows of the table.
     *
     * They are yolo_txt, image_folder, idx, cifar, coco_json, voc_xml, mask_png and csv.
     */
    [[nodiscard]] SushiRuntime::span<const ReaderEntry> reader_table() noexcept;

    /** @brief Returns the row named @p name, or null when the table holds none. */
    [[nodiscard]] const ReaderEntry* find_reader(std::string_view name) noexcept;
} // namespace SushiAI::Vision::Datasets
