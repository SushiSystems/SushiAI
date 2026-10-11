/**************************************************************************/
/* pipeline_table.hpp                                                     */
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
 * @file pipeline_table.hpp
 * @brief Declares the table that maps the name of a pipeline to what builds it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>

namespace SushiAI::Vision::Datasets
{
    /** @brief Holds what a pipeline factory is told of the dataset it builds a pipeline for. */
    struct PipelineRequest
    {
        /** @brief Holds the input size and how the task fits an image to it. */
        SampleFitOptions fit{};

        /** @brief Counts the channels of the input. */
        std::int32_t channels = 0;

        /** @brief Holds the seed of the dataset. */
        std::uint64_t seed = 0;
    };

    /**
     * @brief Builds one pipeline for a dataset.
     * @return The pipeline, never null; InvalidDescription for a request it cannot serve.
     */
    using PipelineFactory =
        Result<std::unique_ptr<const SamplePipeline>> (*)(const PipelineRequest& request);

    /** @brief Holds one row of the pipeline table. */
    struct PipelineEntry
    {
        /** @brief Names the pipeline, as a data section spells it. */
        std::string_view name;

        /** @brief Points at what builds the pipeline; never null. */
        PipelineFactory make;
    };

    /**
     * @brief Returns the rows of the table.
     *
     * They are none, which only fits, and default, the training transforms of imaging.
     */
    [[nodiscard]] SushiRuntime::span<const PipelineEntry> pipeline_table() noexcept;

    /** @brief Returns the row named @p name, or null when the table holds none. */
    [[nodiscard]] const PipelineEntry* find_pipeline(std::string_view name) noexcept;
} // namespace SushiAI::Vision::Datasets
