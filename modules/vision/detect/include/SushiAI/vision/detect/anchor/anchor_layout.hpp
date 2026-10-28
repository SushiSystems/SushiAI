/**************************************************************************/
/* anchor_layout.hpp                                                      */
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
 * @file anchor_layout.hpp
 * @brief Declares the anchors of a model, the anchors each head uses, and their check.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds every anchor of a model and, per head, the anchors that head uses. */
        struct AnchorLayout
        {
            std::vector<AnchorSize> anchors; /**< Every anchor, as fractions of the input. */
            /// @brief Holds, per head, indices into anchors; slot `a` of a head is entry `a`.
            std::vector<std::vector<std::int64_t>> head_anchors;
        };

        /**
         * @brief Checks that a layout has anchors, heads, and head indices that name anchors.
         * @return Success, or InvalidDescription naming the first anchor, head or index that
         *         fails. Two heads may share an anchor and an anchor may belong to no head.
         */
        [[nodiscard]] Status validate_anchor_layout(const AnchorLayout& layout) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
