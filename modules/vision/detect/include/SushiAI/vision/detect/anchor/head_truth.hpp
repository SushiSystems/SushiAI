/**************************************************************************/
/* head_truth.hpp                                                         */
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
 * @file head_truth.hpp
 * @brief Declares the check of the truth rows an anchor head is trained against.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /**
         * @brief Checks that a head of @p classes classes can assign every valid truth row.
         * @param batch Images of the head; the truth must hold as many.
         * @param network Size the truth's pixel corners are divided by.
         * @return Success, or InvalidDescription naming the row and its value: arrays that do
         *         not hold their shape, a network side below 1, a class outside the head's, a
         *         centre outside [0, 1) on either axis, a width or height that is not positive.
         */
        [[nodiscard]] Status validate_head_truth(const Detect::GroundTruthBatch& truth,
                                                 std::int64_t batch, std::int64_t classes,
                                                 NetworkSize network) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
