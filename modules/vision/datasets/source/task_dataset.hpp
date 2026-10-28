/**************************************************************************/
/* task_dataset.hpp                                                       */
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
 * @file task_dataset.hpp
 * @brief Declares the two dataset builders the rows of the task table point at.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>

#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/vision/datasets/reader_table.hpp>
#include <SushiAI/vision/datasets/task_table.hpp>

namespace SushiAI::Vision::Datasets::Detail
{
    /**
     * @brief Builds a ReaderDataset: reads the keys of an image dataset, lists the samples,
     *        writes and uses the cache when one is stated, then builds the collator.
     * @return The dataset; InvalidDescription for a value out of range, a pipeline the
     *         table lacks, fewer classes stated than listed, and a reader of rows of numbers.
     *         The failure of a Config accessor for a key absent or of another kind.
     * @see modules/vision/datasets/README.md, section reader_section.hpp.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> build_image_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context,
        const TaskEntry& task, const ReaderEntry& reader, const ReaderRequest& request);

    /**
     * @brief Builds a TableDataset: reads "batch_size", then the samples, then the collator
     *        of the task.
     * @return The dataset; InvalidDescription for a reader that lists no row of numbers
     *         and for a key only a dataset of images reads.
     *         The failure of a Config accessor when "batch_size" is of another kind.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> build_table_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context,
        const TaskEntry& task, const ReaderEntry& reader, const ReaderRequest& request);
} // namespace SushiAI::Vision::Datasets::Detail
