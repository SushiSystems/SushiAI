/**************************************************************************/
/* intersection_over_union.hpp                                            */
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
 * @file intersection_over_union.hpp
 * @brief Declares intersection over union per class and its mean, read from a confusion matrix.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiAI/vision/segment/confusion_matrix.hpp>

/** @brief Holds the mask geometry and the metrics of semantic segmentation. */
namespace SushiAI::Vision::Segment
{
    /**
     * @brief Reports whether a counted pixel has @p class_index as its true class.
     * @pre @p class_index is in [0, matrix.class_count()); not checked.
     */
    [[nodiscard]] bool class_occurs(const ConfusionMatrix& matrix,
                                    std::int32_t class_index) noexcept;

    /**
     * @brief Returns the pixels both true and predicted as a class, over the pixels that are
     *        either.
     * @pre @p class_index is in [0, matrix.class_count()); not checked.
     * @return A value in [0, 1]; 0 for a class no counted pixel holds or is predicted as.
     */
    [[nodiscard]] double class_iou(const ConfusionMatrix& matrix,
                                   std::int32_t class_index) noexcept;

    /**
     * @brief Returns the mean of class_iou() over the classes for which class_occurs() holds.
     * @return A value in [0, 1]; 0 for a matrix that counted no pixel.
     */
    [[nodiscard]] double mean_iou(const ConfusionMatrix& matrix) noexcept;

} // namespace SushiAI::Vision::Segment
