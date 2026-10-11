/**************************************************************************/
/* pixel_accuracy.hpp                                                     */
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
 * @file pixel_accuracy.hpp
 * @brief Declares pixel accuracy, read from a confusion matrix.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/vision/segment/confusion_matrix.hpp>

/** @brief Holds the mask geometry and the metrics of semantic segmentation. */
namespace SushiAI::Vision::Segment
{
    /**
     * @brief Returns the counted pixels predicted as their true class, over the counted pixels.
     * @return A value in [0, 1]; 0 for a matrix that counted no pixel.
     */
    [[nodiscard]] double pixel_accuracy(const ConfusionMatrix& matrix) noexcept;

} // namespace SushiAI::Vision::Segment
