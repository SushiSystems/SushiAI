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
         * @brief Returns a layout whose heads take runs of anchors, small areas to small strides.
         * @param anchors Kept in their order; a head's slots name them in ascending area.
         * @param strides One per head, above zero; head `k` of the layout has `strides[k]`.
         * @param per_head One count per stride; empty splits evenly, the rest to small strides.
         * @return InvalidDescription for a bad anchor, stride or count; Incompatible when the
         *         counts disagree with the anchors or the heads.
         */
        [[nodiscard]] Result<AnchorLayout>
        assign_anchors_to_heads(SushiRuntime::span<const AnchorSize> anchors,
                                SushiRuntime::span<const std::int64_t> strides,
                                SushiRuntime::span<const std::int64_t> per_head) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
