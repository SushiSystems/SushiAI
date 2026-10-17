/**************************************************************************/
/* task_table.hpp                                                         */
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
 * @file task_table.hpp
 * @brief Declares the table of the tasks a data section may name.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/reader_table.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>

namespace SushiAI::Vision::Datasets
{
    struct TaskEntry;

    /** @brief Holds what a collator factory is told of the dataset it builds a collator for. */
    struct CollatorRequest
    {
        /** @brief Holds the samples the reader listed; read for class names and the table. */
        const SampleList& list;

        /** @brief Counts the classes of the dataset; 0 for a list that names none. */
        std::int32_t class_count;

        /** @brief Holds the input height and width; no size for a task that fits no image. */
        Imaging::ImageSize input;
    };

    /**
     * @brief Builds the collator of one task from a data section.
     * @param section The whole section; the factory reads the keys of its row and no other.
     * @param context What to call the section in a diagnostic.
     * @return The collator, never null, or the failure that kept it from being built.
     */
    using CollatorFactory = Result<std::unique_ptr<const Collator>> (*)(
        const Config::Value& section, std::string_view context, const CollatorRequest& request);

    /**
     * @brief Builds the dataset of one task: reads its keys, the samples, then the collator.
     * @param task The row the builder is in; gives the fit and the collator factory.
     * @param reader The row that lists the samples; called once, after the keys are read.
     * @return The dataset, never null, or the failure of the reader, the collator or the
     *         dataset.
     * @pre @p engine outlives the dataset.
     */
    using DatasetBuilder = Result<std::unique_ptr<Data::Dataset>> (*)(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context,
        const TaskEntry& task, const ReaderEntry& reader, const ReaderRequest& request);

    /** @brief Holds one row of the task table. */
    struct TaskEntry
    {
        /** @brief Names the task, as a data section spells it. */
        std::string_view name;

        /** @brief Selects how an image is fitted to the input size; a table task reads none. */
        SampleFit fit;

        /** @brief Lists the keys the task reads beside the keys every section has. */
        SushiRuntime::span<const std::string_view> keys;

        /** @brief Points at what builds the collator; never null. */
        CollatorFactory make_collator;

        /** @brief Points at what builds the dataset, of images or of a table; never null. */
        DatasetBuilder build;
    };

    /**
     * @brief Returns the rows of the table.
     *
     * They are detection, classification, binary, multi_label, regression and segmentation.
     */
    [[nodiscard]] SushiRuntime::span<const TaskEntry> task_table() noexcept;

    /** @brief Returns the row named @p name, or null when the table holds none. */
    [[nodiscard]] const TaskEntry* find_task(std::string_view name) noexcept;
} // namespace SushiAI::Vision::Datasets
