/**************************************************************************/
/* head_assignment.hpp                                                    */
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
 * @file head_assignment.hpp
 * @brief Declares the assignment of a model's anchors to its heads by area and stride.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /**
         * @brief Returns a layout of runs of anchors, small areas to small strides, for a model
         *        with no cfg mask; a Darknet `mask=` is kept as stated.
         * @param anchors Kept in their order; a head's slots name them in ascending area.
         * @param strides One per head, above zero; head `k` of the layout has `strides[k]`.
         * @param per_head One count per stride; empty splits evenly, the rest to small strides.
         * @return InvalidDescription for a bad input; Incompatible for counts that disagree.
         */
        [[nodiscard]] Result<AnchorLayout>
        assign_anchors_to_heads(SushiRuntime::span<const AnchorSize> anchors,
                                SushiRuntime::span<const std::int64_t> strides,
                                SushiRuntime::span<const std::int64_t> per_head) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
