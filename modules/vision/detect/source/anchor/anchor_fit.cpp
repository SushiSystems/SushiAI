/**************************************************************************/
/* anchor_fit.cpp                                                         */
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
 * @file anchor_fit.cpp
 * @brief Implements the report of how well an anchor layout fits a set of label boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>

#include <algorithm>
#include <cstddef>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Holds the anchor that fits a box best and the shape_iou it reaches. */
            struct BestAnchor
            {
                std::size_t index = 0; /**< Index into the layout's anchors. */
                float iou = 0.0f;      /**< shape_iou of the box with that anchor. */
            };

            /**
             * @brief Returns the anchor of largest shape_iou with @p box, the lower index on a tie.
             * @pre @p anchors is not empty.
             */
            BestAnchor find_best_anchor(AnchorSize box,
                                        const std::vector<AnchorSize>& anchors) noexcept
            {
                BestAnchor best{0, shape_iou(box, anchors[0])};
                for (std::size_t index = 1; index < anchors.size(); ++index)
                {
                    const float iou = shape_iou(box, anchors[index]);
                    if (iou > best.iou)
                        best = BestAnchor{index, iou};
                }
                return best;
            }

            /**
             * @brief Returns, per anchor, whether at least one head lists it.
             * @pre validate_anchor_layout accepts @p layout.
             */
            std::vector<bool> find_owned_anchors(const AnchorLayout& layout)
            {
                std::vector<bool> owned(layout.anchors.size(), false);
                for (const std::vector<std::int64_t>& slots : layout.head_anchors)
                {
                    for (const std::int64_t anchor : slots)
                        owned[static_cast<std::size_t>(anchor)] = true;
                }
                return owned;
            }

            /**
             * @brief Returns the value at rank ceil(n / 10) of @p values in ascending order.
             * @pre @p values is not empty; its order is not kept.
             */
            double take_tenth_percentile(std::vector<double>& values) noexcept
            {
                const std::size_t rank = (values.size() + 9) / 10;
                const auto position = values.begin() + static_cast<std::ptrdiff_t>(rank - 1);

                std::nth_element(values.begin(), position, values.end());
                return *position;
            }
        } // namespace

        Result<AnchorFit> measure_anchor_fit(SushiRuntime::span<const AnchorSize> boxes,
                                             const AnchorLayout& layout,
                                             const FitOptions& options) noexcept
        {
            SA_TRY(validate_anchor_layout(layout));
            SA_FAIL_IF(!(options.recall_iou >= 0.0f && options.recall_iou <= 1.0f),
                       StatusCode::InvalidDescription,
                       "recall_iou must lie in [0, 1]; got {}", options.recall_iou);
            SA_FAIL_IF(boxes.empty(), StatusCode::Incompatible,
                       "an anchor fit needs at least one box; the layout has {} anchors",
                       layout.anchors.size());

            const std::vector<bool> owned = find_owned_anchors(layout);
            std::vector<double> best_ious;
            best_ious.reserve(boxes.size());
            std::vector<std::int64_t> preferred(layout.anchors.size(), 0);
            double iou_sum = 0.0;
            std::int64_t recallable = 0;
            std::int64_t unowned = 0;

            for (std::size_t index = 0; index < boxes.size(); ++index)
            {
                const BestAnchor best = find_best_anchor(boxes[index], layout.anchors);

                best_ious.push_back(static_cast<double>(best.iou));
                iou_sum += static_cast<double>(best.iou);
                ++preferred[best.index];
                if (!owned[best.index])
                    ++unowned;
                else if (best.iou >= options.recall_iou)
                    ++recallable;
            }

            const double count = static_cast<double>(boxes.size());
            AnchorFit fit;
            fit.boxes = static_cast<std::int64_t>(boxes.size());
            fit.mean_best_iou = iou_sum / count;
            fit.best_possible_recall = static_cast<double>(recallable) / count;
            fit.unowned_share = static_cast<double>(unowned) / count;
            fit.best_iou_tenth_percentile = take_tenth_percentile(best_ious);
            fit.anchor_share.reserve(preferred.size());
            for (const std::int64_t boxes_of_anchor : preferred)
                fit.anchor_share.push_back(static_cast<double>(boxes_of_anchor) / count);

            return fit;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI

