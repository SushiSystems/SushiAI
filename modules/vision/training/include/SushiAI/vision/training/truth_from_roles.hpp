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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file truth_from_roles.hpp
 * @brief Declares the roles a detection step reads and their copy into a ground truth.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <SushiAI/dtype/dtype.hpp>
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

    /** @brief Holds one role of a batch as the host reads it: a name, extents, a type and values. */
    struct HostRole
    {
        std::string role;                    /**< Name the batch carries the tensor under. */
        std::vector<std::int64_t> extents;   /**< Extents of the tensor, the batch axis counted. */
        DType dtype = DType::Float32;        /**< Element type of the values. */
        const void* values = nullptr;        /**< First value, or null when the host cannot read
                                                  them in order. */
    };

    /**
     * @brief Copies the boxes, classes and valid roles of @p roles into @p truth.
     * @param roles Hold at least those three roles, read from host memory; others are ignored.
     * @param truth Receives [N, M] rows; a failure leaves it untouched.
     * @return What truth_from_roles() returns for the same roles.
     */
    [[nodiscard]] Status truth_from_host_roles(const std::vector<HostRole>& roles,
                                               Detect::GroundTruthBatch& truth) noexcept;

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
