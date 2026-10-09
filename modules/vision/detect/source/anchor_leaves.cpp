/**************************************************************************/
/* anchor_leaves.cpp                                                      */
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
 * @file anchor_leaves.cpp
 * @brief Implements the anchor points and strides a detection graph reads as leaves.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor_leaves.hpp>

#include <algorithm>
#include <cstddef>

#include <SushiAI/core/common.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Copies the first @p image_values of @p values into each later image. */
            void repeat_first_image(std::vector<float>& values, std::size_t image_values)
            {
                for (std::size_t at = image_values; at < values.size(); at += image_values)
                    std::copy_n(values.begin(), image_values, values.begin() + at);
            }
        } // namespace

        void build_anchor_leaves(SushiRuntime::span<const FeatureLevel> levels,
                                 std::int64_t batch,
                                 AnchorLeaves& leaves)
        {
            SA_THROW_IF(batch <= 0 || levels.empty(),
                        "anchor leaves need a positive batch and at least one level; got batch "
                        "{} and {} levels",
                        batch, levels.size());

            const std::size_t anchors = static_cast<std::size_t>(count_anchors(levels));
            const std::size_t images = static_cast<std::size_t>(batch);
            std::vector<float> strides(anchors);
            leaves.points.resize(images * anchors * K_POINT_VALUES);
            leaves.strides.resize(images * anchors * K_BOX_VALUES);
            make_anchors(levels,
                         SushiRuntime::span<float>(leaves.points.data(), anchors * K_POINT_VALUES),
                         strides);

            for (std::size_t anchor = 0; anchor < anchors; ++anchor)
                std::fill_n(leaves.strides.begin() + anchor * K_BOX_VALUES, K_BOX_VALUES,
                            strides[anchor]);

            repeat_first_image(leaves.points, anchors * K_POINT_VALUES);
            repeat_first_image(leaves.strides, anchors * K_BOX_VALUES);
        }

    } // namespace Vision::Detect
} // namespace SushiAI
