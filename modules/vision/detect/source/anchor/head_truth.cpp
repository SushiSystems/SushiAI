/**************************************************************************/
/* head_truth.cpp                                                         */
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
 * @file head_truth.cpp
 * @brief Implements the check of the truth rows an anchor head is trained against.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_truth.hpp>

#include <cmath>
#include <cstddef>

#include <SushiAI/vision/detect/box_format.hpp>

#include "box_delta.hpp"
#include "truth_box.hpp"

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Reports whether @p value lies in [0, 1); false when it is not a number. */
            bool is_unit_position(float value) noexcept
            {
                return value >= 0.0f && value < 1.0f;
            }

            /** @brief Checks that the arrays of @p truth hold @p batch images of its rows. */
            Status validate_truth_shape(const Detect::GroundTruthBatch& truth, std::int64_t batch)
            {
                SA_FAIL_IF(batch < 1 || truth.batch != batch || truth.max_boxes < 0,
                           StatusCode::InvalidDescription,
                           "a head of {} images needs their truth; got a batch of {} with {} "
                           "rows per image",
                           batch, truth.batch, truth.max_boxes);

                const std::size_t rows =
                    static_cast<std::size_t>(batch) * static_cast<std::size_t>(truth.max_boxes);
                SA_FAIL_IF(truth.labels.size() != rows || truth.valid.size() != rows ||
                               truth.boxes.size() != rows * K_BOX_VALUES,
                           StatusCode::InvalidDescription,
                           "a truth batch of {} rows needs {} labels, {} flags and {} box "
                           "values; got {}, {} and {}",
                           rows, rows, rows, rows * K_BOX_VALUES, truth.labels.size(),
                           truth.valid.size(), truth.boxes.size());
                return Status();
            }

            /**
             * @brief Checks one valid row: its class, its centre and its size.
             * @param index The row's place in `[B, M]`; @p image and @p row name it.
             */
            Status validate_truth_row(const Detect::GroundTruthBatch& truth, std::size_t index,
                                      std::size_t image, std::size_t row, std::int64_t classes,
                                      NetworkSize network)
            {
                const std::int64_t label = truth.labels[index];
                SA_FAIL_IF(label < 0 || label >= classes, StatusCode::InvalidDescription,
                           "truth row {} of image {} names class {} of a head with {} classes",
                           row, image, label, classes);

                const Detail::CentreBox box =
                    Detail::read_truth_box(truth.boxes.data() + index * K_BOX_VALUES, network);
                SA_FAIL_IF(!is_unit_position(box.x) || !is_unit_position(box.y),
                           StatusCode::InvalidDescription,
                           "truth row {} of image {} has its centre at ({}, {}); each "
                           "coordinate must lie in [0, 1)",
                           row, image, box.x, box.y);
                SA_FAIL_IF(!(box.width > 0.0f) || !(box.height > 0.0f) ||
                               !std::isfinite(box.width) || !std::isfinite(box.height),
                           StatusCode::InvalidDescription,
                           "truth row {} of image {} has the size {} by {}; each side must be "
                           "positive and finite",
                           row, image, box.width, box.height);
                return Status();
            }
        } // namespace

        Status validate_head_truth(const Detect::GroundTruthBatch& truth, std::int64_t batch,
                                   std::int64_t classes, NetworkSize network) noexcept
        {
            SA_TRY(validate_truth_shape(truth, batch));
            SA_FAIL_IF(network.height < 1 || network.width < 1, StatusCode::InvalidDescription,
                       "the truth of an anchor head needs a positive network size; got {} by {}",
                       network.height, network.width);

            const std::size_t images = static_cast<std::size_t>(batch);
            const std::size_t rows = static_cast<std::size_t>(truth.max_boxes);
            for (std::size_t image = 0; image < images; ++image)
            {
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const std::size_t index = image * rows + row;
                    if (truth.valid[index] == 0)
                        continue;

                    SA_TRY(validate_truth_row(truth, index, image, row, classes, network));
                }
            }
            return Status();
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
