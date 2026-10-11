/**************************************************************************/
/* detection_collator.hpp                                                 */
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
 * @file detection_collator.hpp
 * @brief Declares the collator of the detection task.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/datasets/collator.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the detection task in a training file. */
    inline constexpr std::string_view K_TASK_DETECTION = "detection";

    /** @brief Holds the coordinate count of one box: x1, y1, x2, y2. */
    inline constexpr std::int64_t K_BOX_COORDINATES = 4;

    /**
     * @brief Writes boxes [N, M, 4] float32, classes [N, M] int32 and valid [N, M] float32.
     *
     * A box is x1, y1, x2, y2 in pixels of the input, the units of Detect::GroundTruthBatch.
     * Slot m of image n holds its object m; valid is 1 there and 0 in a padding slot, whose
     * box and class are 0.
     */
    class DetectionCollator final : public Collator
    {
        public:
            /**
             * @brief Builds the collator of @p class_count classes and M = @p max_objects.
             * @return The collator, never null; InvalidDescription for a value below 1.
             */
            [[nodiscard]] static Result<std::unique_ptr<DetectionCollator>> make(
                std::int32_t class_count, std::int32_t max_objects);

            [[nodiscard]] std::string_view task() const noexcept override;
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /**
             * @brief Writes the inputs and the objects of each sample into the three roles.
             * @return What Collator::collate states; InvalidDescription for an image with
             *         more than M objects or with a class outside [0, class count).
             */
            [[nodiscard]] Status collate(const SampleList& list,
                                         SushiRuntime::span<const FittedSample> samples,
                                         Data::Batch& batch) const override;

        private:
            /** @brief Holds the descriptions of the three roles. */
            DetectionCollator(std::int32_t class_count, std::int32_t max_objects);

            /// @brief Counts the classes an object may name.
            std::int32_t class_count_;

            /// @brief Counts the object slots of one image, M.
            std::int32_t max_objects_;

            /// @brief Holds the descriptions of boxes, classes and valid, in that order.
            std::vector<Data::TargetDescription> targets_;
    };
} // namespace SushiAI::Vision::Datasets
