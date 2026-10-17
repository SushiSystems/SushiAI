/**************************************************************************/
/* sample_pipeline.hpp                                                    */
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
 * @file sample_pipeline.hpp
 * @brief Declares the interface that turns the samples of one batch into fitted samples.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Loads the samples of one batch and gives each the input size and its labels.
     *
     * An implementer keeps no state between two calls, so threads may call produce() at
     * once, each with its own fitted samples.
     * @see modules/vision/datasets/README.md, section "Pipelines".
     */
    class SamplePipeline
    {
        public:
            /** @brief Destroys the pipeline through the interface. */
            virtual ~SamplePipeline() = default;

            /** @brief Returns the name of the pipeline, as a data section spells it. */
            [[nodiscard]] virtual std::string_view name() const noexcept = 0;

            /**
             * @brief Produces one fitted sample per index of @p samples.
             * @param samples Indices into the samples of @p list, one per row of the batch.
             * @param epoch Counted from 0; what a pipeline draws depends on it.
             * @param fitted One entry per index, each as its constructor left it.
             * @return The failure of a load or of a transform, opening with the file.
             */
            [[nodiscard]] virtual Status produce(const SampleList& list,
                                                 SushiRuntime::span<const std::size_t> samples,
                                                 std::int64_t epoch,
                                                 SushiRuntime::span<FittedSample> fitted)
                const = 0;
    };
} // namespace SushiAI::Vision::Datasets
