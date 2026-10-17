/**************************************************************************/
/* augment_pipeline.hpp                                                   */
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
 * @file augment_pipeline.hpp
 * @brief Declares the pipeline that runs the training transforms of imaging on each sample.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the pipeline of the training transforms, in a data section. */
    inline constexpr std::string_view K_PIPELINE_DEFAULT = "default";

    /** @brief Holds what the caller chooses about the training transforms. */
    struct AugmentOptions
    {
        /** @brief Holds the side of the square input, in [1, Imaging::K_MAX_MOSAIC_IMAGE_SIZE]. */
        std::int32_t image_size = 640;

        /** @brief Holds the seed every sample's draws derive from, with its epoch and index. */
        std::uint64_t seed = 0;

        /** @brief Counts the first epochs that compose mosaics; later ones do not. */
        std::int32_t mosaic_epochs = std::numeric_limits<std::int32_t>::max();
    };

    /**
     * @brief Runs Imaging::TrainingPipeline on each sample: mosaic, affine, colour, flip.
     *
     * A sample's draws come from Imaging::make_sample_random(seed, epoch, sample index) and
     * its partners from Imaging::DatasetPartners, so its result does not depend on the batch
     * it is in or on the thread that makes it. A mask takes the geometry of its image.
     */
    class AugmentPipeline final : public SamplePipeline
    {
        public:
            /**
             * @brief Builds the pipeline of one training size and one seed.
             * @return The pipeline, never null; InvalidDescription for a size or an epoch
             *         count outside its bounds.
             */
            [[nodiscard]] static Result<std::unique_ptr<AugmentPipeline>> make(
                const AugmentOptions& options);

            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Produces the three-channel training sample of each index.
             * @return What SamplePipeline::produce states; InvalidDescription for an epoch
             *         outside [0, the largest int32) and for a list of no sample.
             */
            [[nodiscard]] Status produce(const SampleList& list,
                                         SushiRuntime::span<const std::size_t> samples,
                                         std::int64_t epoch,
                                         SushiRuntime::span<FittedSample> fitted) const override;

        private:
            /** @brief Holds the settings make() derived. */
            explicit AugmentPipeline(const Imaging::TrainingSettings& settings) noexcept;

            /// @brief Holds the settings of every Imaging::TrainingPipeline produce() builds.
            Imaging::TrainingSettings settings_;
    };
} // namespace SushiAI::Vision::Datasets
