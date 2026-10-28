/**************************************************************************/
/* truth_from_roles.hpp                                                   */
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
 * @file truth_from_roles.hpp
 * @brief Declares the roles a detection step reads and their copy into a ground truth.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the values of one truth box: x1, y1, x2, y2. */
    inline constexpr std::int64_t K_TRUTH_BOX_VALUES = 4;

    /**
     * @brief Returns the three roles a detection step reads, as Train::match_roles takes them.
     * @param max_objects Object slots of one image, M; at least 1.
     * @param classes Classes the model's heads hold; 0 accepts a dataset of any count.
     * @return boxes [M, 4] float32, classes [M] int32 over @p classes, valid [M] float32, in
     *         that order; InvalidDescription for an M below 1 or a negative class count.
     */
    [[nodiscard]] Result<std::vector<Data::TargetDescription>> describe_truth_roles(
        std::int64_t max_objects, std::int64_t classes) noexcept;

    /**
     * @brief Copies the boxes, classes and valid roles of @p batch into @p truth.
     * @param truth Receives [N, M] rows; its arrays keep their storage between batches of one
     *        size, and a failure leaves it untouched.
     * @return Success; InvalidDescription naming a role the batch lacks or a valid value that
     *         is neither 0 nor 1; Incompatible naming a role of another rank, extent or
     *         element type, or one whose values the host cannot read in order.
     */
    [[nodiscard]] Status truth_from_roles(const Data::Batch& batch,
                                          Detect::GroundTruthBatch& truth) noexcept;

} // namespace SushiAI::Vision::Training
