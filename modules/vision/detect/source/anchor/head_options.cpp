/**************************************************************************/
/* head_options.cpp                                                       */
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
 * @file head_options.cpp
 * @brief Implements the check of an anchor head's options and the counts of its output.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_options.hpp>

#include <cmath>
#include <cstddef>

#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Returns the anchor slots of a head; a head outside the layout is fatal. */
            std::int64_t count_head_slots(const AnchorLayout& layout,
                                          std::int64_t head_index) noexcept
            {
                const std::int64_t heads = static_cast<std::int64_t>(layout.head_anchors.size());
                SA_FATAL_IF(head_index < 0 || head_index >= heads,
                            "head {} is outside an anchor layout of {} heads", head_index, heads);

                const std::size_t head = static_cast<std::size_t>(head_index);
                return static_cast<std::int64_t>(layout.head_anchors[head].size());
            }
        } // namespace

        Status validate_head(const HeadOptions& options, const AnchorLayout& layout,
                             std::int64_t head_index)
        {
            SA_TRY(validate_anchor_layout(layout));

            const std::int64_t heads = static_cast<std::int64_t>(layout.head_anchors.size());
            SA_FAIL_IF(head_index < 0 || head_index >= heads, StatusCode::InvalidDescription,
                       "head {} is outside an anchor layout of {} heads", head_index, heads);
            SA_FAIL_IF(options.num_classes < 1, StatusCode::InvalidDescription,
                       "an anchor head needs at least one class; got {}", options.num_classes);
            SA_FAIL_IF(!std::isfinite(options.scale_xy) || !(options.scale_xy > 0.0f),
                       StatusCode::InvalidDescription,
                       "scale_xy of an anchor head must be finite and above zero; got {}",
                       options.scale_xy);
            return Status();
        }

        std::int64_t count_head_channels(const HeadOptions& options, const AnchorLayout& layout,
                                         std::int64_t head_index) noexcept
        {
            return count_head_slots(layout, head_index) *
                   (K_FIRST_CLASS_ENTRY + options.num_classes);
        }

        std::int64_t count_head_rows(const HeadShape& shape, const AnchorLayout& layout,
                                     std::int64_t head_index) noexcept
        {
            return shape.height * shape.width * count_head_slots(layout, head_index);
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
