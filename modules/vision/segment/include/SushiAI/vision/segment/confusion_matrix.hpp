/**************************************************************************/
/* confusion_matrix.hpp                                                   */
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
 * @file confusion_matrix.hpp
 * @brief Declares the pixel counts of predicted classes against true classes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>

/** @brief Holds the mask geometry and the metrics of semantic segmentation. */
namespace SushiAI::Vision::Segment
{
    /** @brief Holds the largest class count a confusion matrix is built for. */
    inline constexpr std::int32_t K_MAX_CONFUSION_CLASSES = 4096;

    /**
     * @brief Counts pixels by true class and predicted class, over as many batches as are added.
     *
     * Every count is an int64, so a sum over batches is exact. A pixel whose true class is the
     * ignore index is counted nowhere.
     */
    class ConfusionMatrix final
    {
        public:
            /**
             * @brief Builds a matrix of zeros over @p classes classes.
             * @return The matrix; InvalidDescription for a count outside
             *         [1, K_MAX_CONFUSION_CLASSES]; OutOfMemory when it cannot be allocated.
             */
            [[nodiscard]] static Result<ConfusionMatrix> make(std::int32_t classes);

            /**
             * @brief Counts every pixel whose true class is not @p ignore_index.
             * @param predicted One class per pixel; in [0, class_count()) where the pixel counts.
             * @param truth One class per pixel, in [0, class_count()) or @p ignore_index.
             * @return Incompatible, with the counts unchanged, for spans of unequal length or
             *         for a class outside those bounds, naming the pixel.
             */
            [[nodiscard]] Status add(SushiRuntime::span<const std::int32_t> predicted,
                                     SushiRuntime::span<const std::int32_t> truth,
                                     std::int32_t ignore_index);

            /**
             * @brief Adds every count of @p other to this matrix.
             * @return Incompatible, with the counts unchanged, for another class count.
             */
            [[nodiscard]] Status merge(const ConfusionMatrix& other);

            /** @brief Sets every count to zero. */
            void clear() noexcept;

            /** @brief Returns the number of classes, C. */
            [[nodiscard]] std::int32_t class_count() const noexcept { return classes_; }

            /**
             * @brief Returns the pixels of true class @p truth predicted as @p predicted.
             * @pre Both are in [0, class_count()); not checked.
             */
            [[nodiscard]] std::int64_t count(std::int32_t truth,
                                             std::int32_t predicted) const noexcept;

            /**
             * @brief Returns the counted pixels whose true class is @p class_index.
             * @pre @p class_index is in [0, class_count()); not checked.
             */
            [[nodiscard]] std::int64_t truth_pixels(std::int32_t class_index) const noexcept;

            /**
             * @brief Returns the counted pixels predicted as @p class_index.
             * @pre @p class_index is in [0, class_count()); not checked.
             */
            [[nodiscard]] std::int64_t predicted_pixels(std::int32_t class_index) const noexcept;

            /** @brief Returns the number of pixels counted. */
            [[nodiscard]] std::int64_t pixel_count() const noexcept;

        private:
            /** @brief Stores a class count make() has checked and C * C zeros. */
            explicit ConfusionMatrix(std::int32_t classes);

            /** @brief Returns the position of the count of (@p truth, @p predicted). */
            [[nodiscard]] std::size_t offset_of(std::int32_t truth,
                                                std::int32_t predicted) const noexcept;

            /// @brief Counts the classes, C.
            std::int32_t classes_ = 0;

            /// @brief Holds the C * C counts; the row is the true class.
            std::vector<std::int64_t> counts_;
    };

} // namespace SushiAI::Vision::Segment
