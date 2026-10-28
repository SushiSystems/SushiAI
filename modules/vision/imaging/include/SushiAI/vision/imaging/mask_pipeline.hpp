/**************************************************************************/
/* mask_pipeline.hpp                                                      */
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
 * @file mask_pipeline.hpp
 * @brief Declares the pipeline that gives a mask the geometry of its training sample.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/mosaic.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Supplies, by index, the mask of each image a SampleSource supplies.
     *
     * A mask has one channel, one class index per pixel, and the size of its image.
     */
    class MaskSource
    {
        public:
            /** @brief Destroys the source through a base pointer. */
            virtual ~MaskSource() = default;

            /**
             * @brief Loads the mask of one image at the image's stored resolution.
             * @param index In [0, the image count of the SampleSource).
             * @return Success, or why the mask could not be read.
             */
            [[nodiscard]] virtual Status load(std::size_t index, Image& mask) const = 0;
    };

    /**
     * @brief Produces the mask of a training sample from the trace of its image.
     *
     * It draws nothing: the mosaic, the warp and the flip are the ones TrainingTrace
     * records, applied with nearest sampling. One pipeline serves one thread.
     */
    class TrainingMaskPipeline final
    {
        public:
            /**
             * @brief Constructs a pipeline over the masks of a training pipeline's images.
             * @param source Must outlive the pipeline.
             * @param settings The settings the TrainingPipeline was constructed with.
             * @param void_value Written where no mask lands: the mosaic gaps and the border.
             */
            TrainingMaskPipeline(const MaskSource& source, const TrainingSettings& settings,
                                 std::uint8_t void_value) noexcept;

            /**
             * @brief Produces the mask of the sample TrainingPipeline::make_sample made.
             * @param trace What make_sample returned for @p index.
             * @param loaded_size Sample::loaded_size of that sample.
             * @param mask Receives image_size by image_size pixels of one channel.
             * @return The failure of a load; Incompatible for a mask of several channels or
             *         of another size than its image; OutOfMemory.
             */
            [[nodiscard]] Status make_mask(std::size_t index, const TrainingTrace& trace,
                                           ImageSize loaded_size, Image& mask) noexcept;

        private:
            /** @brief Loads one mask and resizes its long side as its image was resized. */
            [[nodiscard]] Status load_resized(std::size_t index, Image& loaded);

            /// @brief Points at the source, which the caller keeps alive.
            const MaskSource* source_ = nullptr;

            /// @brief Holds the training size and the mosaic settings.
            TrainingSettings settings_{};

            /// @brief Holds the byte of a pixel no mask covers.
            std::uint8_t void_value_ = 0;

            /// @brief Holds the mask a load returns before it is resized.
            Image decoded_;

            /// @brief Holds the resized mask of the sample and of each mosaic partner.
            std::array<Image, K_MOSAIC_PATCHES> patches_{};

            /// @brief Holds the mosaic canvas.
            Image canvas_;

            /// @brief Holds the warped mask before the flip.
            Image warped_;
    };
} // namespace SushiAI::Vision::Imaging
