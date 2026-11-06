/**************************************************************************/
/* detection_truth_check.hpp                                              */
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

/**
 * @file detection_truth_check.hpp
 * @brief Declares the checks of a ground truth that both detection training steps run.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Checks that @p truth holds @p batch images and arrays of the size its rows state.
     * @param truth The truth a detection step is about to read.
     * @param batch Images of the step's batch.
     * @param declared_for Names what fixed @p batch in the message: "heads were declared for".
     * @return Success; Incompatible for another batch, a negative row count, or a label, flag
     *         or box array of another size.
     */
    [[nodiscard]] Status check_truth_layout(const Detect::GroundTruthBatch& truth,
                                            std::int64_t batch, std::string_view declared_for) noexcept;

    /**
     * @brief Checks that every valid row of @p truth names a class below @p classes.
     * @param truth A truth that passed check_truth_layout().
     * @param classes Classes the model holds.
     * @param holder Names, in the message, what holds @p classes, such as a head's output.
     * @return Success, or InvalidDescription naming the first valid row whose label is
     *         negative or at least @p classes.
     */
    [[nodiscard]] Status check_truth_labels(const Detect::GroundTruthBatch& truth,
                                            std::int64_t classes, std::string_view holder) noexcept;

} // namespace SushiAI::Vision::Training
