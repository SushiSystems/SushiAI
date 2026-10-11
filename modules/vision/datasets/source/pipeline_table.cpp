/**************************************************************************/
/* pipeline_table.cpp                                                     */
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
 * @file pipeline_table.cpp
 * @brief Implements the table that maps the name of a pipeline to what builds it.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/pipeline_table.hpp>

#include <cstddef>
#include <iterator>
#include <utility>

#include <SushiBLAS/core/format.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/imaging/image.hpp>

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Builds the pipeline that fits and draws nothing. */
        [[nodiscard]] Result<std::unique_ptr<const SamplePipeline>> make_fit_pipeline(
            const PipelineRequest& request)
        {
            SA_TRY_ASSIGN(std::unique_ptr<FitPipeline> pipeline,
                          FitPipeline::make(request.fit, request.channels));

            return std::unique_ptr<const SamplePipeline>(std::move(pipeline));
        }

        /**
         * @brief Builds the pipeline of the training transforms.
         * @return The pipeline; InvalidDescription for an input that is not square or has
         *         another channel count than three, which the transforms do not produce.
         */
        [[nodiscard]] Result<std::unique_ptr<const SamplePipeline>> make_augment_pipeline(
            const PipelineRequest& request)
        {
            SA_FAIL_IF(request.fit.size.height != request.fit.size.width,
                       StatusCode::InvalidDescription,
                       "the pipeline '{}' makes square samples; the input is {} by {}",
                       K_PIPELINE_DEFAULT, request.fit.size.height, request.fit.size.width);
            SA_FAIL_IF(request.channels != Imaging::K_RGB_CHANNELS,
                       StatusCode::InvalidDescription,
                       "the pipeline '{}' makes samples of {} channels; the input has {}",
                       K_PIPELINE_DEFAULT, Imaging::K_RGB_CHANNELS, request.channels);

            AugmentOptions options;
            options.image_size = request.fit.size.height;
            options.seed = request.seed;

            SA_TRY_ASSIGN(std::unique_ptr<AugmentPipeline> pipeline,
                          AugmentPipeline::make(options));

            return std::unique_ptr<const SamplePipeline>(std::move(pipeline));
        }

        /** @brief Holds the rows; the first is what a section without the key gets. */
        constexpr PipelineEntry K_PIPELINES[] = {
            PipelineEntry{K_PIPELINE_NONE, &make_fit_pipeline},
            PipelineEntry{K_PIPELINE_DEFAULT, &make_augment_pipeline},
        };
    } // namespace

    SushiRuntime::span<const PipelineEntry> pipeline_table() noexcept
    {
        return SushiRuntime::span<const PipelineEntry>(K_PIPELINES, std::size(K_PIPELINES));
    }

    const PipelineEntry* find_pipeline(std::string_view name) noexcept
    {
        for (const PipelineEntry& row : pipeline_table())
        {
            if (row.name == name)
                return &row;
        }

        return nullptr;
    }
} // namespace SushiAI::Vision::Datasets
