/**************************************************************************/
/* dice_targets.hpp                                                       */
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
 * @file dice_targets.hpp
 * @brief Declares the fill of the five leaves of the soft Dice loss.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the five leaves a Dice step reads: two INT32, three float32 or float64. */
    struct DiceLeaves
    {
        SushiBLAS::Tensor* score_indices = nullptr; /**< R, INT32: r * C + class, or r * C. */
        SushiBLAS::Tensor* class_indices = nullptr; /**< R, INT32: the class, or 0 if ignored. */
        SushiBLAS::Tensor* kept = nullptr;          /**< R: 1 for a kept pixel, else 0. */
        SushiBLAS::Tensor* class_pixels = nullptr;  /**< C: the kept pixels of each class. */
        SushiBLAS::Tensor* class_weights = nullptr; /**< C: 1 / present, or 0 when absent. */
    };

    /**
     * @brief Writes the five leaves from the R class indices of @p mask; a class is present
     *        when a kept pixel holds it, and a mask with no kept pixel gives zeros.
     * @param leaves Five tensors, none null, of the element counts DiceLeaves states.
     * @return Success, or Incompatible, with the leaves unchanged, for what
     *         validate_pixel_labels() refuses, a leaf that does not fit, or an R * classes
     *         that INT32 cannot index; OutOfMemory.
     */
    [[nodiscard]] Status fill_dice_targets(const SushiBLAS::Tensor& mask, std::int64_t classes,
                                           std::int32_t ignore_index, const DiceLeaves& leaves) noexcept;

} // namespace SushiAI::Vision::Training
