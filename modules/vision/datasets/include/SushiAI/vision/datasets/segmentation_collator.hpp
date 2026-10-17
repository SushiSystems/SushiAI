/**************************************************************************/
/* segmentation_collator.hpp                                              */
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
 * @file segmentation_collator.hpp
 * @brief Declares the collator of the semantic segmentation task.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the semantic segmentation task in a training file. */
    inline constexpr std::string_view K_TASK_SEGMENTATION = "segmentation";

    /** @brief Holds the value the role mask has, by default, where no class is known. */
    inline constexpr std::int32_t K_DEFAULT_IGNORE_INDEX = 255;

    /** @brief Names where the collator takes the class of a pixel from. */
    enum class MaskOrigin : std::uint8_t
    {
        File,     /**< FittedSample::mask, fitted from the mask file of the sample. */
        Polygons, /**< The polygons of FittedSample::objects, through paint_polygon_mask. */
    };

    /** @brief Holds what the caller chooses about the role mask. */
    struct SegmentationOptions
    {
        /** @brief Counts the classes, in [1, K_MASK_VOID]. */
        std::int32_t class_count = 0;

        /** @brief Holds the input height and width, H and W. */
        Imaging::ImageSize size{};

        /** @brief Holds what a pixel of K_MASK_VOID becomes; outside [0, class_count). */
        std::int32_t ignore_index = K_DEFAULT_IGNORE_INDEX;

        /** @brief Selects the mask file or the polygons of a sample. */
        MaskOrigin origin = MaskOrigin::File;

        /** @brief Holds the class of a pixel no polygon holds; K_MASK_VOID when absent. */
        std::optional<std::int32_t> background;
    };

    /**
     * @brief Writes the role mask: [N, H, W] int32, a class per pixel or the ignore index.
     *
     * A byte of the mask below the class count is written as it is, and K_MASK_VOID,
     * which is also the padding of fit_mask, as the ignore index. The mask is the fitted
     * mask file of a sample or, under MaskOrigin::Polygons, its polygons painted.
     */
    class SegmentationCollator final : public Collator
    {
        public:
            /**
             * @brief Builds the collator of a class count, an input size and an ignore index.
             * @return The collator, never null; InvalidDescription for a value outside the
             *         bounds SegmentationOptions states, for a background that is no class
             *         and for one stated with MaskOrigin::File.
             */
            [[nodiscard]] static Result<std::unique_ptr<SegmentationCollator>> make(
                const SegmentationOptions& options);

            [[nodiscard]] std::string_view task() const noexcept override;
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /**
             * @brief Writes the inputs and the mask of each sample.
             * @return What Collator::collate states; InvalidDescription for a sample whose
             *         mask is not H by W of one channel, with the row and the column for a
             *         byte that is neither a class nor K_MASK_VOID, and, opening with the
             *         file, what paint_polygon_mask refuses.
             */
            [[nodiscard]] Status collate(const SampleList& list,
                                         SushiRuntime::span<const FittedSample> samples,
                                         Data::Batch& batch) const override;

        private:
            /** @brief Holds the options and the description of mask. */
            explicit SegmentationCollator(const SegmentationOptions& options);

            /// @brief Holds the class count, the input size and the ignore index.
            SegmentationOptions options_;

            /// @brief Holds the description of the mask role.
            std::vector<Data::TargetDescription> targets_;
    };
} // namespace SushiAI::Vision::Datasets
