/**************************************************************************/
/* confusion_matrix.cpp                                                   */
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
 * @file confusion_matrix.cpp
 * @brief Implements the pixel counts of predicted classes against true classes.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/segment/confusion_matrix.hpp>

#include <algorithm>
#include <string_view>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI::Vision::Segment
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_UNIT = "ConfusionMatrix";

        /** @brief Reports whether @p value names one of @p classes classes. */
        [[nodiscard]] constexpr bool is_class(std::int32_t value, std::int32_t classes) noexcept
        {
            return value >= 0 && value < classes;
        }
    } // namespace

    Result<ConfusionMatrix> ConfusionMatrix::make(std::int32_t classes)
    {
        SA_FAIL_IF(classes < 1 || classes > K_MAX_CONFUSION_CLASSES,
                   StatusCode::InvalidDescription,
                   "{}: {} classes; a matrix counts 1 to {}", K_UNIT, classes,
                   K_MAX_CONFUSION_CLASSES);

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [classes]()
        {
            return ConfusionMatrix(classes);
        });
    }

    ConfusionMatrix::ConfusionMatrix(std::int32_t classes)
        : classes_(classes),
          counts_(static_cast<std::size_t>(classes) * static_cast<std::size_t>(classes), 0)
    {

    }

    Status ConfusionMatrix::add(SushiRuntime::span<const std::int32_t> predicted,
                                SushiRuntime::span<const std::int32_t> truth,
                                std::int32_t ignore_index)
    {
        SA_FAIL_IF(predicted.size() != truth.size(), StatusCode::Incompatible,
                   "{}: {} predicted classes against {} true classes; one of each per pixel",
                   K_UNIT, predicted.size(), truth.size());

        for (std::size_t pixel = 0; pixel < truth.size(); ++pixel)
        {
            if (truth[pixel] == ignore_index)
                continue;

            SA_FAIL_IF(!is_class(truth[pixel], classes_), StatusCode::Incompatible,
                       "{}: the true class of pixel {} is {}; a class is below {} and {} names "
                       "none",
                       K_UNIT, pixel, truth[pixel], classes_, ignore_index);
            SA_FAIL_IF(!is_class(predicted[pixel], classes_), StatusCode::Incompatible,
                       "{}: the predicted class of pixel {} is {}; a class is below {}", K_UNIT,
                       pixel, predicted[pixel], classes_);
        }

        // Every pixel was checked, so a refused batch leaves the counts as they were.
        for (std::size_t pixel = 0; pixel < truth.size(); ++pixel)
        {
            if (truth[pixel] != ignore_index)
                ++counts_[offset_of(truth[pixel], predicted[pixel])];
        }

        return Status();
    }

    Status ConfusionMatrix::merge(const ConfusionMatrix& other)
    {
        SA_FAIL_IF(other.classes_ != classes_, StatusCode::Incompatible,
                   "{}: a matrix of {} classes cannot take the counts of one of {}", K_UNIT,
                   classes_, other.classes_);

        for (std::size_t index = 0; index < counts_.size(); ++index)
            counts_[index] += other.counts_[index];

        return Status();
    }

    void ConfusionMatrix::clear() noexcept
    {
        std::fill(counts_.begin(), counts_.end(), std::int64_t{0});
    }

    std::int64_t ConfusionMatrix::count(std::int32_t truth, std::int32_t predicted) const noexcept
    {
        return counts_[offset_of(truth, predicted)];
    }

    std::int64_t ConfusionMatrix::truth_pixels(std::int32_t class_index) const noexcept
    {
        std::int64_t total = 0;
        for (std::int32_t predicted = 0; predicted < classes_; ++predicted)
            total += counts_[offset_of(class_index, predicted)];

        return total;
    }

    std::int64_t ConfusionMatrix::predicted_pixels(std::int32_t class_index) const noexcept
    {
        std::int64_t total = 0;
        for (std::int32_t truth = 0; truth < classes_; ++truth)
            total += counts_[offset_of(truth, class_index)];

        return total;
    }

    std::int64_t ConfusionMatrix::pixel_count() const noexcept
    {
        std::int64_t total = 0;
        for (const std::int64_t value : counts_)
            total += value;

        return total;
    }

    std::size_t ConfusionMatrix::offset_of(std::int32_t truth,
                                           std::int32_t predicted) const noexcept
    {
        return static_cast<std::size_t>(truth) * static_cast<std::size_t>(classes_)
             + static_cast<std::size_t>(predicted);
    }

} // namespace SushiAI::Vision::Segment
