/**************************************************************************/
/* detection_truth_check.cpp                                              */
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

/**
 * @file detection_truth_check.cpp
 * @brief Implements the checks of a ground truth that both detection training steps run.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/detection_truth_check.hpp>

#include <cstddef>

#include <SushiAI/failure/check.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the box values of one truth row. */
        constexpr std::size_t K_BOX_VALUES = 4;
    } // namespace

    Status check_truth_layout(const Detect::GroundTruthBatch& truth, std::int64_t batch,
                              std::string_view declared_for)
    {
        SA_FAIL_IF(truth.batch != batch, StatusCode::Incompatible,
                   "The truth holds {} images, but the {} a batch of {}", truth.batch,
                   declared_for, batch);
        SA_FAIL_IF(truth.max_boxes < 0, StatusCode::Incompatible,
                   "The truth holds {} rows per image; the count cannot be negative",
                   truth.max_boxes);

        const std::size_t rows = static_cast<std::size_t>(batch * truth.max_boxes);
        SA_FAIL_IF(truth.labels.size() != rows || truth.valid.size() != rows
                       || truth.boxes.size() != rows * K_BOX_VALUES,
                   StatusCode::Incompatible,
                   "A truth of {} images and {} rows per image needs {} labels, {} flags and {} "
                   "box values; it holds {}, {} and {}",
                   batch, truth.max_boxes, rows, rows, rows * K_BOX_VALUES, truth.labels.size(),
                   truth.valid.size(), truth.boxes.size());

        return Status();
    }

    Status check_truth_labels(const Detect::GroundTruthBatch& truth, std::int64_t classes,
                              std::string_view holder)
    {
        for (std::size_t row = 0; row < truth.valid.size(); ++row)
        {
            SA_FAIL_IF(truth.valid[row] != 0 && (truth.labels[row] < 0
                                                 || truth.labels[row] >= classes),
                       StatusCode::InvalidDescription,
                       "Truth row {} has the label {}, but {} has {} classes", row,
                       truth.labels[row], holder, classes);
        }

        return Status();
    }

} // namespace SushiAI::Vision::Training
