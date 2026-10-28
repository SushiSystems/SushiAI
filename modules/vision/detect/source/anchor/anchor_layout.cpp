/**************************************************************************/
/* anchor_layout.cpp                                                      */
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
 * @file anchor_layout.cpp
 * @brief Implements the check of an anchor layout.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>

#include <cmath>
#include <cstddef>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Reports whether @p side is a finite length above zero. */
            bool is_positive_side(float side) noexcept
            {
                return std::isfinite(side) && side > 0.0f;
            }
        } // namespace

        Status validate_anchor_layout(const AnchorLayout& layout) noexcept
        {
            SA_FAIL_IF(layout.anchors.empty(), StatusCode::InvalidDescription,
                       "an anchor layout needs at least one anchor");
            SA_FAIL_IF(layout.head_anchors.empty(), StatusCode::InvalidDescription,
                       "an anchor layout needs at least one head");

            for (std::size_t index = 0; index < layout.anchors.size(); ++index)
            {
                const AnchorSize& anchor = layout.anchors[index];
                SA_FAIL_IF(!is_positive_side(anchor.width) || !is_positive_side(anchor.height),
                           StatusCode::InvalidDescription,
                           "anchor {} must have finite sides above zero; got {} by {}", index,
                           anchor.width, anchor.height);
            }

            const std::int64_t anchor_count = static_cast<std::int64_t>(layout.anchors.size());
            for (std::size_t head = 0; head < layout.head_anchors.size(); ++head)
            {
                const std::vector<std::int64_t>& slots = layout.head_anchors[head];
                SA_FAIL_IF(slots.empty(), StatusCode::InvalidDescription,
                           "head {} of an anchor layout uses no anchor", head);

                for (std::size_t slot = 0; slot < slots.size(); ++slot)
                {
                    SA_FAIL_IF(slots[slot] < 0 || slots[slot] >= anchor_count,
                               StatusCode::InvalidDescription,
                               "slot {} of head {} names anchor {}; the layout has {} anchors",
                               slot, head, slots[slot], anchor_count);
                }
            }
            return Status();
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
